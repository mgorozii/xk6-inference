package inference

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func BenchmarkConcurrentInitialization(b *testing.B) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprintf(w, `{"max_batch_size": 8, "input": [{"name": "i", "data_type": "TYPE_FP32", "dims": [1]}], "output": []}`); err != nil {
			b.Fatal("write response:", err)
		}
	}))
	defer s.Close()

	mod := &module{}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			c, err := mod.Connect(s.URL, "localhost:8033")
			if err != nil {
				b.Fatal(err)
			}
			m, err := c.Model("bench")
			if err != nil {
				b.Fatal(err)
			}
			if m.config.MaxBatchSize != 8 {
				b.Fatal("cfg fail")
			}
		}
	})
}
