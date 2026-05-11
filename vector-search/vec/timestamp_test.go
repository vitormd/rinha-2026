package vec

import (
	"testing"
	"time"
)

func TestParseISO8601Z(t *testing.T) {
	cases := []struct {
		in           string
		y, mo, d, h, mi, s int
	}{
		{"2026-03-11T18:45:53Z", 2026, 3, 11, 18, 45, 53},
		{"2026-03-23T15:33:12Z", 2026, 3, 23, 15, 33, 12},
		{"2026-01-01T00:00:00Z", 2026, 1, 1, 0, 0, 0},
		{"1970-01-01T00:00:00Z", 1970, 1, 1, 0, 0, 0},
	}
	for _, tc := range cases {
		y, mo, d, h, mi, s, err := parseISO8601Z([]byte(tc.in))
		if err != nil {
			t.Errorf("%s: %v", tc.in, err)
			continue
		}
		if y != tc.y || mo != tc.mo || d != tc.d || h != tc.h || mi != tc.mi || s != tc.s {
			t.Errorf("%s: got %d-%d-%dT%d:%d:%d, want %d-%d-%dT%d:%d:%d",
				tc.in, y, mo, d, h, mi, s, tc.y, tc.mo, tc.d, tc.h, tc.mi, tc.s)
		}
	}
}

func TestParseISO8601ZBad(t *testing.T) {
	// We validate the fixed length and the structural separators (the
	// only things that can change format) but not that each digit slot
	// holds an ASCII digit — that would cost an extra branch per byte on
	// the hot path, and the data-generator only emits valid timestamps.
	bad := []string{"", "x", "2026-03-11T18:45:53", "2026/03/11T18:45:53Z"}
	for _, s := range bad {
		if _, _, _, _, _, _, err := parseISO8601Z([]byte(s)); err == nil {
			t.Errorf("%q: expected error", s)
		}
	}
}

func TestDayOfWeekAgainstStdlib(t *testing.T) {
	// Cover March 2026 (test-data range) and a few other dates.
	dates := []string{
		"2026-03-01T00:00:00Z", "2026-03-11T00:00:00Z", "2026-03-23T00:00:00Z",
		"2026-01-01T00:00:00Z", "2024-02-29T00:00:00Z", "2000-01-01T00:00:00Z",
		"1999-12-31T00:00:00Z", "1970-01-01T00:00:00Z",
	}
	for _, s := range dates {
		want := time.Time{}
		want, _ = time.Parse(time.RFC3339, s)
		stdlibDow := (int(want.Weekday()) + 6) % 7 // Sun=0 → shift to Mon=0
		y, mo, d, _, _, _, _ := parseISO8601Z([]byte(s))
		got := dayOfWeekMonZero(y, mo, d)
		if got != stdlibDow {
			t.Errorf("%s: got dow=%d, stdlib=%d", s, got, stdlibDow)
		}
	}
}

func TestMinutesBetween(t *testing.T) {
	// Generator subtracts mins_back * 60 seconds from req_epoch — so the
	// diff is always an exact integer number of minutes.
	cases := []struct {
		req, last string
		want      float64
	}{
		{"2026-03-11T18:45:53Z", "2026-03-11T18:30:53Z", 15.0},
		{"2026-03-11T20:23:35Z", "2026-03-11T14:58:35Z", 325.0},
		{"2026-03-12T00:00:00Z", "2026-03-11T23:00:00Z", 60.0},
		// Cross month
		{"2026-04-01T00:00:00Z", "2026-03-31T23:00:00Z", 60.0},
	}
	for _, tc := range cases {
		y1, mo1, d1, h1, mi1, s1, _ := parseISO8601Z([]byte(tc.last))
		y2, mo2, d2, h2, mi2, s2, _ := parseISO8601Z([]byte(tc.req))
		got := minutesBetween(y1, mo1, d1, h1, mi1, s1, y2, mo2, d2, h2, mi2, s2)
		if got != tc.want {
			t.Errorf("%s - %s: got %v want %v", tc.req, tc.last, got, tc.want)
		}
	}
}
