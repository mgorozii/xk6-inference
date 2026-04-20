package inference

import (
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Trendyol/go-triton-client/base"
	tgrpc "github.com/Trendyol/go-triton-client/client/grpc"
	thttp "github.com/Trendyol/go-triton-client/client/http"
	"go.k6.io/k6/js/modules"
	"go.k6.io/k6/metrics"
)

type InferenceMetrics struct {
	Reqs     *metrics.Metric
	Duration *metrics.Metric
}

type Client struct {
	httpURL string
	vu      modules.VU
	metrics *InferenceMetrics
	hc      base.Client
	gc      base.Client
}

type modelConfig struct {
	MaxBatchSize int `json:"max_batch_size"`
	Input        []struct {
		Name     string  `json:"name"`
		DataType string  `json:"data_type"`
		Dims     []int64 `json:"dims"`
	} `json:"input"`
	Output []struct {
		Name     string `json:"name"`
		DataType string `json:"data_type"`
	} `json:"output"`
}

type Model struct {
	c      *Client
	name   string
	config *modelConfig
}

type cacheKey struct{ a, b string }

var (
	configCache = make(map[cacheKey]*modelConfig)
	configMu    sync.RWMutex
)

func (c *Client) Model(name string) (*Model, error) {
	m := &Model{c: c, name: name}
	if c.httpURL == "" {
		m.config = &modelConfig{}
		return m, nil
	}

	k := cacheKey{c.httpURL, name}
	configMu.RLock()
	m.config = configCache[k]
	configMu.RUnlock()

	if m.config == nil {
		cfg, err := (&AutoDetectProvider{}).GetModelConfig(http.DefaultClient, c.httpURL, name)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch model config: %w", err)
		}
		configMu.Lock()
		configCache[k] = cfg
		configMu.Unlock()
		m.config = cfg
	}
	return m, nil
}

func (m *Model) Http(data ...any) (any, error) {
	return m.infer("http", data...)
}

func (m *Model) Grpc(data ...any) (any, error) {
	return m.infer("grpc", data...)
}

func (m *Model) normalizeType(t string) string {
	res := strings.ToUpper(strings.TrimPrefix(t, "TYPE_"))
	if res == "STRING" {
		return "BYTES"
	}
	return res
}

func (m *Model) getValLen(val any) int {
	switch v := val.(type) {
	case []float32:
		return len(v)
	case []int64:
		return len(v)
	case []int8:
		return len(v)
	case []byte:
		return len(v)
	case []string:
		return len(v)
	case []any:
		return len(v)
	}
	return 0
}

func (m *Model) calculateShape(configDims []int64, val any) []int64 {
	if len(configDims) == 0 {
		return []int64{1}
	}

	actualShape := make([]int64, len(configDims))
	copy(actualShape, configDims)

	totalElements := m.getValLen(val)

	multiplier := int64(1)
	dynamicIndex := -1
	for i, d := range configDims {
		if d == -1 {
			if dynamicIndex == -1 {
				dynamicIndex = i
			}
		} else {
			multiplier *= d
		}
	}

	if dynamicIndex != -1 {
		if multiplier > 0 {
			actualShape[dynamicIndex] = int64(totalElements) / multiplier
			if actualShape[dynamicIndex] == 0 {
				actualShape[dynamicIndex] = 1
			}
		} else {
			actualShape[dynamicIndex] = int64(totalElements)
		}
	}

	return actualShape
}

func (m *Model) infer(proto string, data ...any) (any, error) {
	var client base.Client
	if proto == "http" {
		client = m.c.hc
	} else {
		client = m.c.gc
	}

	if client == nil {
		return nil, fmt.Errorf("inference %s client not initialized", proto)
	}

	inputs := make([]base.InferInput, 0, len(m.config.Input))
	var inputData map[string]any
	var positionalData []any
	var sentBytes int

	if len(data) > 0 {
		switch d := data[0].(type) {
		case map[string]any:
			inputData = d
		case []any:
			positionalData = d
		}
	}

	for i, in := range m.config.Input {
		var val any
		if inputData != nil {
			if d, ok := inputData[in.Name]; ok {
				var err error
				val, err = m.convertData(d, in.DataType)
				if err != nil {
					return nil, fmt.Errorf("failed to convert input %s: %w", in.Name, err)
				}
			}
		} else if positionalData != nil && i < len(positionalData) {
			var err error
			val, err = m.convertData(positionalData[i], in.DataType)
			if err != nil {
				return nil, fmt.Errorf("failed to convert positional input %d: %w", i, err)
			}
		}

		if val == nil {
			val = m.generateDummyData(in.DataType, in.Dims)
		}

		var inferInput base.InferInput
		dt := m.normalizeType(in.DataType)
		actualShape := m.calculateShape(in.Dims, val)
		if proto == "http" {
			inferInput = thttp.NewInferInput(in.Name, dt, actualShape, nil)
		} else {
			inferInput = tgrpc.NewInferInput(in.Name, dt, actualShape, nil)
		}

		if err := inferInput.SetData(val, false); err != nil {
			return nil, fmt.Errorf("failed to set data for input %s: %w", in.Name, err)
		}
		inputs = append(inputs, inferInput)
		sentBytes += m.getDataBytes(val)
	}

	outputs := make([]base.InferOutput, 0, len(m.config.Output))
	for _, out := range m.config.Output {
		var inferOutput base.InferOutput
		if proto == "http" {
			inferOutput = thttp.NewInferOutput(out.Name, nil)
		} else {
			inferOutput = tgrpc.NewInferOutput(out.Name, nil)
		}
		outputs = append(outputs, inferOutput)
	}

	start := time.Now()
	res, err := client.Infer(m.c.vu.Context(), m.name, "", inputs, outputs, nil)
	duration := time.Since(start)

	var recvBytes int
	var result map[string]any

	if err == nil {
		result = make(map[string]any)
		for _, out := range m.config.Output {
			var val any
			var resErr error
			switch out.DataType {
			case "TYPE_FP32":
				val, resErr = res.AsFloat32Slice(out.Name)
			case "TYPE_INT64":
				val, resErr = res.AsInt64Slice(out.Name)
			case "TYPE_STRING":
				val, resErr = res.AsByteSlice(out.Name)
			default:
				resErr = fmt.Errorf("unsupported output type: %s", out.DataType)
			}
			if resErr == nil {
				result[out.Name] = val
				recvBytes += m.getDataBytes(val)
			}
		}
	}

	m.reportMetrics(proto, duration, err == nil, sentBytes, recvBytes)

	if err != nil {
		return nil, err
	}

	return result, nil
}

func (m *Model) convertData(val any, dataType string) (any, error) {
	slice, ok := val.([]any)
	if !ok {
		// If it's already a native slice, return as is (for tests)
		switch dataType {
		case "TYPE_FP32":
			if _, ok := val.([]float32); ok {
				return val, nil
			}
		case "TYPE_INT64":
			if _, ok := val.([]int64); ok {
				return val, nil
			}
		case "TYPE_INT8":
			if _, ok := val.([]int8); ok {
				return val, nil
			}
		}
		return val, nil
	}

	switch dataType {
	case "TYPE_FP32":
		res := make([]float32, len(slice))
		for i, v := range slice {
			switch t := v.(type) {
			case float64:
				res[i] = float32(t)
			case int64:
				res[i] = float32(t)
			case int:
				res[i] = float32(t)
			default:
				return nil, fmt.Errorf("cannot convert %T to float32", v)
			}
		}
		return res, nil
	case "TYPE_INT64":
		res := make([]int64, len(slice))
		for i, v := range slice {
			switch t := v.(type) {
			case float64:
				res[i] = int64(t)
			case int64:
				res[i] = t
			case int:
				res[i] = int64(t)
			default:
				return nil, fmt.Errorf("cannot convert %T to int64", v)
			}
		}
		return res, nil
	default:
		return val, nil
	}
}

func (m *Model) generateDummyData(dataType string, dims []int64) any {
	size := 1
	for _, d := range dims {
		if d > 0 {
			size *= int(d)
		} else {
			size *= 1
		}
	}

	switch dataType {
	case "TYPE_FP32":
		res := make([]float32, size)
		for i := range res {
			// #nosec G404
			res[i] = rand.Float32()
		}
		return res
	case "TYPE_INT64":
		res := make([]int64, size)
		for i := range res {
			// #nosec G404
			res[i] = rand.Int63()
		}
		return res
	default:
		return make([]int8, size)
	}
}

func (m *Model) getDataBytes(val any) int {
	switch v := val.(type) {
	case []float32:
		return len(v) * 4
	case []int64:
		return len(v) * 8
	case []int8:
		return len(v) * 1
	case []byte:
		return len(v) * 1
	case []string:
		total := 0
		for _, s := range v {
			total += len(s)
		}
		return total
	case []any:
		return len(v) * 8 // Assuming references or arbitrary structs
	}
	return 0
}

func (m *Model) reportMetrics(proto string, duration time.Duration, success bool, sentBytes, recvBytes int) {
	state := m.c.vu.State()
	if state == nil {
		return
	}

	tags := state.Tags.GetCurrentValues().Tags.With("model_name", m.name).With("protocol", proto)
	now := time.Now()

	// Custom metrics
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: m.c.metrics.Reqs, Tags: tags},
		Value:      1,
		Time:       now,
	}
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: m.c.metrics.Duration, Tags: tags},
		Value:      metrics.D(duration),
		Time:       now,
	}

	// Standard k6 metrics (for compatibility with thresholds)
	reqs := state.BuiltinMetrics.HTTPReqs
	reqDuration := state.BuiltinMetrics.HTTPReqDuration
	reqFailed := state.BuiltinMetrics.HTTPReqFailed

	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: reqs, Tags: tags},
		Value:      1,
		Time:       now,
	}
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: reqDuration, Tags: tags},
		Value:      metrics.D(duration),
		Time:       now,
	}

	failedValue := 0.0
	if !success {
		failedValue = 1.0
	}
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: reqFailed, Tags: tags},
		Value:      failedValue,
		Time:       now,
	}

	// Data sent/received (approximate)
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: state.BuiltinMetrics.DataSent, Tags: tags},
		Value:      float64(sentBytes),
		Time:       now,
	}
	state.Samples <- metrics.Sample{
		TimeSeries: metrics.TimeSeries{Metric: state.BuiltinMetrics.DataReceived, Tags: tags},
		Value:      float64(recvBytes),
		Time:       now,
	}
}
