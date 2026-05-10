// Package vec converts a fraud-score request payload into the 14-dimensional
// vector expected by the IVF index and exposes the JSON config types loaded
// once at startup.
//
// All math runs in float64 to match the data-generator's reference
// computation precision; the int16 quantization happens later in package
// quant.
package vec

// Dim is the fixed dimensionality of the fraud-score vector.
const Dim = 14

// Sentinel is the value placed in dims 5 and 6 when last_transaction is
// null. Any reference vector with the same null state will share these
// values, naturally clustering "no-history" cases together in vector space.
const Sentinel float64 = -1.0

// DefaultMccRisk is used when an MCC code is absent from the risk table.
const DefaultMccRisk float64 = 0.5

// Norm holds the normalization constants loaded from normalization.json.
type Norm struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

// MccRisk maps each MCC code to its risk score in [0, 1].
type MccRisk map[string]float64
