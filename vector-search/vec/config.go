package vec

import (
	"encoding/json"
	"errors"
	"os"
)

// LoadNorm reads normalization.json and validates that no constant is zero
// (a zero would produce divide-by-zero or always-clamped values downstream).
func LoadNorm(path string) (*Norm, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	norm := &Norm{}
	if err := json.Unmarshal(raw, norm); err != nil {
		return nil, err
	}
	if norm.MaxAmount == 0 ||
		norm.MaxInstallments == 0 ||
		norm.AmountVsAvgRatio == 0 ||
		norm.MaxMinutes == 0 ||
		norm.MaxKm == 0 ||
		norm.MaxTxCount24h == 0 ||
		norm.MaxMerchantAvgAmount == 0 {
		return nil, errors.New("normalization.json has zero constants")
	}
	return norm, nil
}

// LoadMccRisk reads mcc_risk.json into a string→float64 map.
func LoadMccRisk(path string) (MccRisk, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	parsed := map[string]float64{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	out := make(MccRisk, len(parsed))
	for code, risk := range parsed {
		out[code] = risk
	}
	return out, nil
}
