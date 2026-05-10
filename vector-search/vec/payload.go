package vec

import (
	"time"

	"github.com/buger/jsonparser"
)

// FromPayload vectorizes a fraud-score request payload (raw JSON) into a
// 14-D float64 vector following the dim order defined by the challenge
// specification.
//
// The payload is expected to follow the schema documented in
// docs/br/REGRAS_DE_DETECCAO.md. The output array is rewritten in place;
// no allocations on the hot path beyond what jsonparser does internally.
func FromPayload(payload []byte, norm *Norm, mcc MccRisk, out *[Dim]float64) error {
	*out = [Dim]float64{}

	if err := parseTransaction(payload, norm, out); err != nil {
		return err
	}
	requestedAt, err := parseRequestedAt(payload, out)
	if err != nil {
		return err
	}
	if err := parseAmountVsAvg(payload, norm, out); err != nil {
		return err
	}
	parseLastTransaction(payload, norm, requestedAt, out)
	if err := parseTerminal(payload, norm, out); err != nil {
		return err
	}
	if err := parseCustomer(payload, norm, out); err != nil {
		return err
	}
	if err := parseMerchant(payload, norm, mcc, out); err != nil {
		return err
	}
	return nil
}

// parseTransaction fills dims 0 (amount) and 1 (installments).
func parseTransaction(payload []byte, norm *Norm, out *[Dim]float64) error {
	amount, err := getFloat(payload, "transaction", "amount")
	if err != nil {
		return err
	}
	out[0] = clamp(amount / norm.MaxAmount)

	installments, err := getFloat(payload, "transaction", "installments")
	if err != nil {
		return err
	}
	out[1] = clamp(installments / norm.MaxInstallments)
	return nil
}

// parseAmountVsAvg fills dim 2 (transaction.amount / customer.avg_amount).
// Reads transaction.amount and customer.avg_amount independently — both must
// be in the payload. Caller is responsible for parseTransaction running first
// (we re-read amount here for locality; the cost is one jsonparser walk).
func parseAmountVsAvg(payload []byte, norm *Norm, out *[Dim]float64) error {
	amount, err := getFloat(payload, "transaction", "amount")
	if err != nil {
		return err
	}
	custAvg, err := getFloat(payload, "customer", "avg_amount")
	if err != nil {
		return err
	}
	if custAvg > 0 {
		out[2] = clamp((amount / custAvg) / norm.AmountVsAvgRatio)
	}
	return nil
}

// parseRequestedAt parses transaction.requested_at, fills dim 3 (hour) and
// dim 4 (day_of_week, Mon=0..Sun=6), and returns the parsed timestamp for
// later use by parseLastTransaction.
func parseRequestedAt(payload []byte, out *[Dim]float64) (time.Time, error) {
	raw, _, _, err := jsonparser.Get(payload, "transaction", "requested_at")
	if err != nil {
		return time.Time{}, err
	}
	t, err := time.Parse(time.RFC3339, string(raw))
	if err != nil {
		return time.Time{}, err
	}
	t = t.UTC()
	out[3] = float64(t.Hour()) / 23.0
	// Go's Weekday: Sunday=0..Saturday=6. Spec wants Monday=0..Sunday=6.
	weekday := (int(t.Weekday()) + 6) % 7
	out[4] = float64(weekday) / 6.0
	return t, nil
}

// parseLastTransaction fills dims 5 (minutes_since) and 6 (km_from_current).
// When last_transaction is null/absent, both dims receive the Sentinel value.
func parseLastTransaction(payload []byte, norm *Norm, requestedAt time.Time, out *[Dim]float64) {
	lastTxObj, valueType, _, _ := jsonparser.Get(payload, "last_transaction")
	if valueType != jsonparser.Object || len(lastTxObj) == 0 {
		out[5] = Sentinel
		out[6] = Sentinel
		return
	}

	hasAny := false
	if tsRaw, _, _, err := jsonparser.Get(lastTxObj, "timestamp"); err == nil {
		if lastTxAt, err := time.Parse(time.RFC3339, string(tsRaw)); err == nil {
			diffMinutes := requestedAt.UTC().Sub(lastTxAt.UTC()).Minutes()
			if diffMinutes < 0 {
				diffMinutes = 0
			}
			out[5] = clamp(diffMinutes / norm.MaxMinutes)
			hasAny = true
		}
	}
	if kmFromLast, err := getFloat(lastTxObj, "km_from_current"); err == nil {
		out[6] = clamp(kmFromLast / norm.MaxKm)
		hasAny = true
	}
	if !hasAny {
		out[5] = Sentinel
		out[6] = Sentinel
	}
}

// parseTerminal fills dims 7 (km_from_home), 9 (is_online), 10 (card_present).
func parseTerminal(payload []byte, norm *Norm, out *[Dim]float64) error {
	kmFromHome, err := getFloat(payload, "terminal", "km_from_home")
	if err != nil {
		return err
	}
	out[7] = clamp(kmFromHome / norm.MaxKm)

	if isOnline, _ := jsonparser.GetBoolean(payload, "terminal", "is_online"); isOnline {
		out[9] = 1
	}
	if cardPresent, _ := jsonparser.GetBoolean(payload, "terminal", "card_present"); cardPresent {
		out[10] = 1
	}
	return nil
}

// parseCustomer fills dim 8 (tx_count_24h). dim 11 (unknown_merchant) needs
// the merchant id, so it lives in parseMerchant.
func parseCustomer(payload []byte, norm *Norm, out *[Dim]float64) error {
	txCount, err := getFloat(payload, "customer", "tx_count_24h")
	if err != nil {
		return err
	}
	out[8] = clamp(txCount / norm.MaxTxCount24h)
	return nil
}

// parseMerchant fills dims 11 (unknown_merchant), 12 (mcc_risk),
// 13 (merchant.avg_amount).
func parseMerchant(payload []byte, norm *Norm, mcc MccRisk, out *[Dim]float64) error {
	merchantID, _, _, err := jsonparser.Get(payload, "merchant", "id")
	if err != nil {
		return err
	}

	known := false
	jsonparser.ArrayEach(payload, func(value []byte, _ jsonparser.ValueType, _ int, _ error) {
		if !known && bytesEqual(value, merchantID) {
			known = true
		}
	}, "customer", "known_merchants")
	if !known {
		out[11] = 1
	}

	mccCode, _, _, err := jsonparser.Get(payload, "merchant", "mcc")
	if err != nil {
		return err
	}
	if risk, ok := mcc[string(mccCode)]; ok {
		out[12] = risk
	} else {
		out[12] = DefaultMccRisk
	}

	merchantAvg, err := getFloat(payload, "merchant", "avg_amount")
	if err != nil {
		return err
	}
	out[13] = clamp(merchantAvg / norm.MaxMerchantAvgAmount)
	return nil
}

// --- helpers ---

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func getFloat(data []byte, keys ...string) (float64, error) {
	return jsonparser.GetFloat(data, keys...)
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
