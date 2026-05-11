package vec

import "errors"

// parseISO8601Z parses a `YYYY-MM-DDTHH:MM:SSZ` UTC timestamp. The format
// is the exact one produced by the challenge's data-generator
// (`sprintf("...%04d-%02d-%02dT%02d:%02d:%02dZ", ...)`). Bypasses
// time.Parse's general-purpose parser to save ~1µs per call.
func parseISO8601Z(b []byte) (y, mo, d, h, mi, s int, err error) {
	if len(b) != 20 ||
		b[4] != '-' || b[7] != '-' ||
		b[10] != 'T' ||
		b[13] != ':' || b[16] != ':' ||
		b[19] != 'Z' {
		return 0, 0, 0, 0, 0, 0, errors.New("vec: bad iso8601 timestamp")
	}
	y = digit(b[0])*1000 + digit(b[1])*100 + digit(b[2])*10 + digit(b[3])
	mo = digit(b[5])*10 + digit(b[6])
	d = digit(b[8])*10 + digit(b[9])
	h = digit(b[11])*10 + digit(b[12])
	mi = digit(b[14])*10 + digit(b[15])
	s = digit(b[17])*10 + digit(b[18])
	return
}

// dayOfWeekMonZero returns the day of week with Monday=0..Sunday=6 (the
// spec's convention). Uses Tomohiko Sakamoto's algorithm — same as the
// data-generator's `day_of_week()` in main.c, so we don't drift on edge
// dates.
func dayOfWeekMonZero(y, m, d int) int {
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if m < 3 {
		y--
	}
	sun0 := (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7
	// sun0: 0=Sun..6=Sat. Convert to Mon=0..Sun=6.
	return (sun0 + 6) % 7
}

// daysSinceEpoch returns the number of days from 1970-01-01 to the given
// proleptic-Gregorian date. Howard Hinnant's branchless formula —
// handles leap years and centuries correctly.
func daysSinceEpoch(y, m, d int) int {
	if m <= 2 {
		y--
	}
	era := y / 400
	if y < 0 && y%400 != 0 {
		era--
	}
	yoe := y - era*400
	mp := m
	if m > 2 {
		mp -= 3
	} else {
		mp += 9
	}
	doy := (153*mp+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}

// minutesBetween returns minutes from t1 → t2 (positive if t2 is later).
// Each component is YYYY-MM-DD HH:MM:SS (UTC).
func minutesBetween(y1, mo1, d1, h1, mi1, s1, y2, mo2, d2, h2, mi2, s2 int) float64 {
	days := daysSinceEpoch(y2, mo2, d2) - daysSinceEpoch(y1, mo1, d1)
	secs := int64(days)*86400 +
		int64(h2-h1)*3600 +
		int64(mi2-mi1)*60 +
		int64(s2-s1)
	return float64(secs) / 60.0
}

func digit(b byte) int { return int(b - '0') }
