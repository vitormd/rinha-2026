package ivf

import (
	"os"
	"testing"
)

// tempFile creates a temp file that's automatically removed when the test
// finishes.
func tempFile(t *testing.T) (*os.File, error) {
	t.Helper()
	f, err := os.CreateTemp("", "ivf-test-*.bin")
	if err != nil {
		return nil, err
	}
	t.Cleanup(func() { os.Remove(f.Name()) })
	return f, nil
}
