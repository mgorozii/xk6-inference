package inference

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	tgrpc "github.com/Trendyol/go-triton-client/client/grpc"
	thttp "github.com/Trendyol/go-triton-client/client/http"
	"github.com/Trendyol/go-triton-client/base"
	"go.k6.io/k6/js/modules"
	"go.k6.io/k6/metrics"
)

type rootModule struct{}

func (*rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
	reg := vu.InitEnv().Registry
	tm := &InferenceMetrics{}
	tm.Reqs, _ = reg.NewMetric("inference_reqs", metrics.Counter)
	tm.Duration, _ = reg.NewMetric("inference_req_duration", metrics.Trend, metrics.Time)

	return &module{
		vu:      vu,
		metrics: tm,
	}
}

type module struct {
	vu      modules.VU
	metrics *InferenceMetrics
}

func (m *module) Exports() modules.Exports {
	return modules.Exports{
		Named: map[string]any{
			"connect": m.Connect,
		},
	}
}

func getEnvInt(key string, def int) int {
	if val := os.Getenv(key); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil {
			return parsed
		}
	}
	return def
}

var (
	connCache = make(map[cacheKey]*connections)
	connMu    sync.Mutex
)

type connections struct {
	hc base.Client
	gc base.Client
}

func (m *module) Connect(httpURL, grpcURL string) (*Client, error) {
	k := cacheKey{httpURL, grpcURL}
	connMu.Lock()
	defer connMu.Unlock()

	conns, ok := connCache[k]
	if !ok {
		maxIdleConns := getEnvInt("INFERENCE_MAX_IDLE_CONNS", 100)
		maxOpenConns := getEnvInt("INFERENCE_MAX_OPEN_CONNS", 500)
		conns = &connections{}

		if httpURL != "" {
			cleanHTTP := httpURL
			ssl := false
			if strings.HasPrefix(httpURL, "http://") {
				cleanHTTP = strings.TrimPrefix(httpURL, "http://")
			} else if strings.HasPrefix(httpURL, "https://") {
				cleanHTTP = strings.TrimPrefix(httpURL, "https://")
				ssl = true
			}

			hc, err := thttp.NewClient(cleanHTTP, false, float64(maxIdleConns), float64(maxOpenConns), ssl, true, nil, nil)
			if err != nil {
				return nil, fmt.Errorf("failed to create inference HTTP client: %w", err)
			}
			conns.hc = hc
		}

		if grpcURL != "" {
			gc, err := tgrpc.NewClient(grpcURL, false, float64(maxIdleConns), float64(maxOpenConns), false, true, nil, log.Default())
			if err != nil {
				return nil, fmt.Errorf("failed to create inference gRPC client: %w", err)
			}
			conns.gc = gc
		}
		connCache[k] = conns
	}

	return &Client{
		httpURL: httpURL,
		vu:      m.vu,
		metrics: m.metrics,
		hc:      conns.hc,
		gc:      conns.gc,
	}, nil
}

var _ modules.Module = (*rootModule)(nil)
