package vec

import (
	"errors"

	"github.com/buger/jsonparser"
)

// Path indexes into the EachKey paths slice below. Used by the callback to
// route each matched value to the right field of the running raw struct.
const (
	pAmount = iota
	pInstallments
	pRequestedAt
	pCustAvg
	pTxCount
	pKnownMerchants
	pMerchantID
	pMcc
	pMerchantAvg
	pIsOnline
	pCardPresent
	pKmHome
	pLastTransaction
)

// payloadPaths is shared across requests (read-only). EachKey walks the JSON
// once and dispatches each match by its position in this slice.
var payloadPaths = [][]string{
	pAmount:          {"transaction", "amount"},
	pInstallments:    {"transaction", "installments"},
	pRequestedAt:     {"transaction", "requested_at"},
	pCustAvg:         {"customer", "avg_amount"},
	pTxCount:         {"customer", "tx_count_24h"},
	pKnownMerchants:  {"customer", "known_merchants"},
	pMerchantID:      {"merchant", "id"},
	pMcc:             {"merchant", "mcc"},
	pMerchantAvg:     {"merchant", "avg_amount"},
	pIsOnline:        {"terminal", "is_online"},
	pCardPresent:     {"terminal", "card_present"},
	pKmHome:          {"terminal", "km_from_home"},
	pLastTransaction: {"last_transaction"},
}

// rawFields is the bag of values captured during the single EachKey pass.
// All byte slices reference into the request payload buffer (lifetime tied
// to the request handler).
type rawFields struct {
	amount         []byte
	installments   []byte
	requestedAt    []byte
	custAvg        []byte
	txCount        []byte
	knownMerchants []byte
	merchantID     []byte
	mcc            []byte
	merchantAvg    []byte
	isOnline       []byte
	cardPresent    []byte
	kmHome         []byte
	lastTx         []byte
	lastTxType     jsonparser.ValueType
}

// FromPayload vectorizes a fraud-score request payload (raw JSON) into a
// 14-D float64 vector following the dim order defined by the challenge
// specification.
//
// All required fields are collected in a single jsonparser.EachKey walk
// (replacing the ~13 separate Get calls of the previous version), then
// post-processed. No allocations on the hot path beyond what jsonparser does
// internally for the EachKey state machine.
func FromPayload(payload []byte, norm *Norm, mcc MccRisk, out *[Dim]float64) error {
	*out = [Dim]float64{}

	var f rawFields
	jsonparser.EachKey(payload, func(idx int, value []byte, vt jsonparser.ValueType, _ error) {
		switch idx {
		case pAmount:
			f.amount = value
		case pInstallments:
			f.installments = value
		case pRequestedAt:
			f.requestedAt = value
		case pCustAvg:
			f.custAvg = value
		case pTxCount:
			f.txCount = value
		case pKnownMerchants:
			f.knownMerchants = value
		case pMerchantID:
			f.merchantID = value
		case pMcc:
			f.mcc = value
		case pMerchantAvg:
			f.merchantAvg = value
		case pIsOnline:
			f.isOnline = value
		case pCardPresent:
			f.cardPresent = value
		case pKmHome:
			f.kmHome = value
		case pLastTransaction:
			f.lastTx = value
			f.lastTxType = vt
		}
	}, payloadPaths...)

	if f.amount == nil || f.installments == nil || f.requestedAt == nil ||
		f.custAvg == nil || f.txCount == nil ||
		f.merchantID == nil || f.mcc == nil || f.merchantAvg == nil ||
		f.kmHome == nil {
		return errors.New("vec.FromPayload: missing required field")
	}

	amount, err := jsonparser.ParseFloat(f.amount)
	if err != nil {
		return err
	}
	installments, err := jsonparser.ParseFloat(f.installments)
	if err != nil {
		return err
	}
	custAvg, err := jsonparser.ParseFloat(f.custAvg)
	if err != nil {
		return err
	}
	txCount, err := jsonparser.ParseFloat(f.txCount)
	if err != nil {
		return err
	}
	merchantAvg, err := jsonparser.ParseFloat(f.merchantAvg)
	if err != nil {
		return err
	}
	kmHome, err := jsonparser.ParseFloat(f.kmHome)
	if err != nil {
		return err
	}

	out[0] = clamp(amount / norm.MaxAmount)
	out[1] = clamp(installments / norm.MaxInstallments)
	if custAvg > 0 {
		out[2] = clamp((amount / custAvg) / norm.AmountVsAvgRatio)
	}

	ry, rmo, rd, rh, rmi, rs, err := parseISO8601Z(f.requestedAt)
	if err != nil {
		return err
	}
	out[3] = float64(rh) / 23.0
	out[4] = float64(dayOfWeekMonZero(ry, rmo, rd)) / 6.0

	fillLastTransaction(&f, norm, ry, rmo, rd, rh, rmi, rs, out)

	out[7] = clamp(kmHome / norm.MaxKm)
	out[8] = clamp(txCount / norm.MaxTxCount24h)
	if bytesEqual(f.isOnline, []byte("true")) {
		out[9] = 1
	}
	if bytesEqual(f.cardPresent, []byte("true")) {
		out[10] = 1
	}

	out[11] = computeUnknownMerchant(f.knownMerchants, f.merchantID)
	if risk, ok := mcc[string(f.mcc)]; ok {
		out[12] = risk
	} else {
		out[12] = DefaultMccRisk
	}
	out[13] = clamp(merchantAvg / norm.MaxMerchantAvgAmount)

	return nil
}

// fillLastTransaction handles dims 5 (minutes_since) and 6 (km_from_current).
// When `last_transaction` is null/absent, both dims receive the Sentinel.
func fillLastTransaction(f *rawFields, norm *Norm, ry, rmo, rd, rh, rmi, rs int, out *[Dim]float64) {
	if f.lastTxType != jsonparser.Object || len(f.lastTx) == 0 {
		out[5] = Sentinel
		out[6] = Sentinel
		return
	}

	// Sub-object — one more EachKey walk over a small object (~60 bytes).
	var tsRaw, kmRaw []byte
	jsonparser.EachKey(f.lastTx, func(idx int, value []byte, _ jsonparser.ValueType, _ error) {
		switch idx {
		case 0:
			tsRaw = value
		case 1:
			kmRaw = value
		}
	}, []string{"timestamp"}, []string{"km_from_current"})

	hasAny := false
	if tsRaw != nil {
		if ly, lmo, ld, lh, lmi, ls, err := parseISO8601Z(tsRaw); err == nil {
			diffMinutes := minutesBetween(ly, lmo, ld, lh, lmi, ls, ry, rmo, rd, rh, rmi, rs)
			if diffMinutes < 0 {
				diffMinutes = 0
			}
			out[5] = clamp(diffMinutes / norm.MaxMinutes)
			hasAny = true
		}
	}
	if kmRaw != nil {
		if km, err := jsonparser.ParseFloat(kmRaw); err == nil {
			out[6] = clamp(km / norm.MaxKm)
			hasAny = true
		}
	}
	if !hasAny {
		out[5] = Sentinel
		out[6] = Sentinel
	}
}

// computeUnknownMerchant returns 1 if `merchantID` is NOT present in the
// `known_merchants` array bytes, 0 otherwise.
func computeUnknownMerchant(knownMerchantsArr, merchantID []byte) float64 {
	if len(knownMerchantsArr) == 0 || merchantID == nil {
		return 1
	}
	known := false
	jsonparser.ArrayEach(knownMerchantsArr, func(value []byte, _ jsonparser.ValueType, _ int, _ error) {
		if !known && bytesEqual(value, merchantID) {
			known = true
		}
	})
	if known {
		return 0
	}
	return 1
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
