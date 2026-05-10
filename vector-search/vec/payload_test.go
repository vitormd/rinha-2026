package vec

import (
	"math"
	"testing"
)

// stdNorm matches the values shipped in dataset/normalization.json. Kept
// inline so tests don't depend on filesystem state.
func stdNorm() *Norm {
	return &Norm{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
}

func stdMcc() MccRisk {
	return MccRisk{
		"5411": 0.15,
		"5812": 0.30,
		"5912": 0.20,
		"7995": 0.85,
	}
}

const samplePayloadNoLast = `{
	"id": "tx-test",
	"transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
	"customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
	"merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 29.2331036248},
	"last_transaction": null
}`

const samplePayloadWithLast = `{
	"id": "tx-test",
	"transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
	"customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
	"merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 13.7090520965},
	"last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774}
}`

func TestFromPayloadNoLastTransaction(t *testing.T) {
	var got [Dim]float64
	if err := FromPayload([]byte(samplePayloadNoLast), stdNorm(), stdMcc(), &got); err != nil {
		t.Fatalf("FromPayload: %v", err)
	}

	// Spot-check a few dims with hand-computed expected values.
	cases := []struct {
		dim  int
		want float64
		desc string
	}{
		{0, 41.12 / 10000.0, "amount"},
		{1, 2.0 / 12.0, "installments"},
		{2, (41.12 / 82.24) / 10.0, "amount_vs_avg"},
		{3, 18.0 / 23.0, "hour"},
		// 2026-03-11 is a Wednesday → spec dow=2 → 2/6
		{4, 2.0 / 6.0, "weekday"},
		{5, Sentinel, "minutes_since_last (no history)"},
		{6, Sentinel, "km_from_last (no history)"},
		{7, 29.2331036248 / 1000.0, "km_from_home"},
		{8, 3.0 / 20.0, "tx_count_24h"},
		{9, 0, "is_online"},
		{10, 1, "card_present"},
		{11, 0, "unknown_merchant (MERC-016 IS known)"},
		{12, 0.15, "mcc_risk 5411"},
		{13, 60.25 / 10000.0, "merchant_avg"},
	}
	for _, tc := range cases {
		if !floatNearlyEqual(got[tc.dim], tc.want, 1e-9) {
			t.Errorf("dim %d (%s): got %v want %v", tc.dim, tc.desc, got[tc.dim], tc.want)
		}
	}
}

func TestFromPayloadWithLastTransaction(t *testing.T) {
	var got [Dim]float64
	if err := FromPayload([]byte(samplePayloadWithLast), stdNorm(), stdMcc(), &got); err != nil {
		t.Fatalf("FromPayload: %v", err)
	}

	// 20:23:35 - 14:58:35 = 5h25m = 325 min exactly (data-generator's
	// last_ts is requested_at - mins_back*60s, so the diff has zero seconds).
	wantMinutes := 325.0 / 1440.0
	if !floatNearlyEqual(got[5], wantMinutes, 1e-9) {
		t.Errorf("dim 5 (minutes_since): got %v want %v", got[5], wantMinutes)
	}
	wantKm := 18.8626479774 / 1000.0
	if !floatNearlyEqual(got[6], wantKm, 1e-9) {
		t.Errorf("dim 6 (km_from_last): got %v want %v", got[6], wantKm)
	}
	if got[11] != 0 { // MERC-001 IS in known_merchants
		t.Errorf("dim 11 (unknown_merchant): got %v want 0", got[11])
	}
	if got[12] != 0.20 { // mcc 5912
		t.Errorf("dim 12 (mcc_risk): got %v want 0.20", got[12])
	}
}

func TestFromPayloadUnknownMerchant(t *testing.T) {
	payload := `{
		"id":"x",
		"transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},
		"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":["MERC-A","MERC-B"]},
		"merchant":{"id":"MERC-X","mcc":"5411","avg_amount":50},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":1.5},
		"last_transaction":null
	}`
	var got [Dim]float64
	if err := FromPayload([]byte(payload), stdNorm(), stdMcc(), &got); err != nil {
		t.Fatalf("FromPayload: %v", err)
	}
	if got[11] != 1 {
		t.Errorf("dim 11 (unknown_merchant): MERC-X is NOT in known list, got %v want 1", got[11])
	}
}

func TestFromPayloadDefaultMccRisk(t *testing.T) {
	payload := `{
		"id":"x",
		"transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},
		"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":[]},
		"merchant":{"id":"M","mcc":"9999","avg_amount":50},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":1.5},
		"last_transaction":null
	}`
	var got [Dim]float64
	if err := FromPayload([]byte(payload), stdNorm(), stdMcc(), &got); err != nil {
		t.Fatalf("FromPayload: %v", err)
	}
	if got[12] != DefaultMccRisk {
		t.Errorf("dim 12 (mcc_risk for unknown 9999): got %v want %v", got[12], DefaultMccRisk)
	}
}

func TestClampBoundaries(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{-0.1, 0},
		{0, 0},
		{0.5, 0.5},
		{1, 1},
		{1.5, 1},
	}
	for _, tc := range cases {
		got := clamp(tc.in)
		if got != tc.want {
			t.Errorf("clamp(%v): got %v want %v", tc.in, got, tc.want)
		}
	}
}

func floatNearlyEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}
