#include "textflag.h"

// func distBlock(query *[16]int32, block *[112]int16, out *[8]int64, threshold int64)
//
// Computes 8 squared L2 int64 distances. Per dim d:
//   refs   <- VPMOVSXWD block + d*16 (Y1)              -- 8 i16 -> 8 i32
//   diff   <- VPSUBD VPBROADCASTD query+d*4 (Y2), Y1    -- (ref - q) i32
//   sq32   <- VPMULLD diff * diff (Y1)                  -- (ref - q)^2 as i32 (max 4e8)
//   acc_lo <- VPADDQ acc_lo, VPMOVZXDQ low(sq32)        -- accumulate slots 0..3 (i64)
//   acc_hi <- VPADDQ acc_hi, VPMOVZXDQ high(sq32)       -- accumulate slots 4..7 (i64)
//
// After the first 8 dims, if every slot's partial sum already exceeds the
// running top-5 worst (threshold), the remaining 6 dims are skipped — adding
// only non-negative squared diffs can't bring the distance back below the
// threshold. The caller sees the partial sums and rejects them naturally
// because partial >= threshold is not strictly less than threshold.
//
// Y0  = acc_lo (i64, slots 0..3)
// Y15 = acc_hi (i64, slots 4..7)
// Y14 = threshold broadcast to 4 i64 lanes
TEXT ·DistBlock(SB), NOSPLIT, $0-32
	MOVQ query+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ out+16(FP), CX
	MOVQ threshold+24(FP), DX

	// Broadcast threshold to 4 i64 lanes in Y14.
	VMOVQ        DX, X14
	VPBROADCASTQ X14, Y14

	VPXOR Y0, Y0, Y0
	VPXOR Y15, Y15, Y15

#define DIM(off_block, off_query) \
	VPMOVSXWD off_block(BX), Y1 \
	VPBROADCASTD off_query(AX), Y2 \
	VPSUBD Y2, Y1, Y1 \
	VPMULLD Y1, Y1, Y1 \
	VPMOVZXDQ X1, Y3 \
	VEXTRACTI128 $1, Y1, X1 \
	VPMOVZXDQ X1, Y4 \
	VPADDQ Y3, Y0, Y0 \
	VPADDQ Y4, Y15, Y15

	// First 8 dims (offsets 0..7 of 14).
	DIM(0,   0)
	DIM(16,  4)
	DIM(32,  8)
	DIM(48,  12)
	DIM(64,  16)
	DIM(80,  20)
	DIM(96,  24)
	DIM(112, 28)

	// Threshold prune: if every lane (acc > threshold) → skip the rest.
	VPCMPGTQ  Y14, Y0, Y5
	VPCMPGTQ  Y14, Y15, Y6
	VMOVMSKPD Y5, DX
	VMOVMSKPD Y6, SI
	SHLQ      $4, SI
	ORQ       SI, DX
	CMPQ      DX, $0xFF
	JE        store

	// Remaining 6 dims.
	DIM(128, 32)
	DIM(144, 36)
	DIM(160, 40)
	DIM(176, 44)
	DIM(192, 48)
	DIM(208, 52)

#undef DIM

store:
	VMOVDQU Y0,  0(CX)
	VMOVDQU Y15, 32(CX)
	VZEROUPPER
	RET
