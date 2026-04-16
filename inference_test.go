package inference

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"

	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/options"
	"github.com/grafana/sobek"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"go.k6.io/k6/js/modules"
	"go.k6.io/k6/lib"
	"go.k6.io/k6/metrics"
)

type mockTritonClient struct {
	base.Client
	inferFn func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error)
}

func (m *mockTritonClient) Infer(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
	if m.inferFn != nil {
		return m.inferFn(ctx, modelName, modelVersion, inputs, outputs, options)
	}
	return nil, errors.New("not implemented")
}

type mockInferResult struct {
	base.InferResult
	float32Fn func(name string) ([]float32, error)
	int64Fn   func(name string) ([]int64, error)
	byteFn    func(name string) ([]string, error)
}

func (m *mockInferResult) AsFloat32Slice(name string) ([]float32, error) {
	if m.float32Fn != nil {
		return m.float32Fn(name)
	}
	return nil, nil
}

func (m *mockInferResult) AsInt64Slice(name string) ([]int64, error) {
	if m.int64Fn != nil {
		return m.int64Fn(name)
	}
	return nil, nil
}

func (m *mockInferResult) AsByteSlice(name string) ([]string, error) {
	if m.byteFn != nil {
		return m.byteFn(name)
	}
	return nil, nil
}

type mockVU struct {
	modules.VU
	state *lib.State
	ctx   context.Context
}

func (m *mockVU) State() *lib.State {
	return m.state
}

func (m *mockVU) Context() context.Context {
	return m.ctx
}

func (m *mockVU) Runtime() *sobek.Runtime {
	return nil
}

var _ = Describe("Inference", func() {
	var (
		client  *Client
		model   *Model
		mv      *mockVU
		samples chan metrics.SampleContainer
	)

	BeforeEach(func() {
		samples = make(chan metrics.SampleContainer, 100)
		registry := metrics.NewRegistry()
		tm := &InferenceMetrics{}
		tm.Reqs, _ = registry.NewMetric("inference_reqs", metrics.Counter)
		tm.Duration, _ = registry.NewMetric("inference_req_duration", metrics.Trend, metrics.Time)

		state := &lib.State{
			Samples:        samples,
			BuiltinMetrics: metrics.RegisterBuiltinMetrics(registry),
			Tags:           lib.NewVUStateTags(registry.RootTagSet()),
		}
		mv = &mockVU{
			state: state,
			ctx:   context.Background(),
		}
		client = &Client{
			vu:      mv,
			metrics: tm,
		}
	})

	Context("Model configuration", func() {
		It("should fetch model config correctly", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				Expect(r.URL.Path).To(Equal("/v2/models/test-model/config"))
				_, _ = w.Write([]byte(`{"input":[{"name":"input0","data_type":"TYPE_FP32","dims":[1,224,224,3]}],"output":[{"name":"output0","data_type":"TYPE_FP32"}]}`))
			}))
			defer server.Close()

			client.httpURL = server.URL
			m, err := client.Model("test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(m.name).To(Equal("test-model"))
			Expect(m.config.Input).To(HaveLen(1))
			Expect(m.config.Input[0].Name).To(Equal("input0"))
		})

		It("should return error if server is unreachable", func() {
			client.httpURL = "http://non-existent-server:1234"
			_, err := client.Model("test-model")
			Expect(err).To(HaveOccurred())
		})

		It("should return error if config is invalid JSON", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(`invalid-json`))
			}))
			defer server.Close()

			client.httpURL = server.URL
			_, err := client.Model("test-model")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("Inference", func() {
		BeforeEach(func() {
			model = &Model{
				c:      client,
				name:   "test-model",
				config: &modelConfig{},
			}
			model.config.Input = []struct {
				Name     string  `json:"name"`
				DataType string  `json:"data_type"`
				Dims     []int64 `json:"dims"`
			}{
				{Name: "input0", DataType: "TYPE_FP32", Dims: []int64{1, 2}},
			}
			model.config.Output = []struct {
				Name     string `json:"name"`
				DataType string `json:"data_type"`
			}{
				{Name: "output0", DataType: "TYPE_FP32"},
			}
		})

		It("should perform HTTP inference", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					Expect(modelName).To(Equal("test-model"))
					Expect(inputs).To(HaveLen(1))
					Expect(inputs[0].GetName()).To(Equal("input0"))

					return &mockInferResult{
						float32Fn: func(name string) ([]float32, error) {
							return []float32{0.5, 0.5}, nil
						},
					}, nil
				},
			}
			client.hc = mockTriton

			res, err := model.Http(map[string]any{"input0": []any{1.0, 2.0}})
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(HaveKey("output0"))
			Expect(res.(map[string]any)["output0"]).To(Equal([]float32{0.5, 0.5}))

			// Check metrics
			Eventually(samples).Should(Receive()) // inference_reqs
			Eventually(samples).Should(Receive()) // inference_req_duration
			Eventually(samples).Should(Receive()) // http_reqs
			Eventually(samples).Should(Receive()) // http_req_duration
			Eventually(samples).Should(Receive()) // http_req_failed
			Eventually(samples).Should(Receive()) // data_sent
			Eventually(samples).Should(Receive()) // data_received
		})

		It("should return error if inference returns error", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					return nil, errors.New("inference error")
				},
			}
			client.hc = mockTriton

			_, err := model.Http()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("inference error"))
		})

		It("should perform gRPC inference", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					return &mockInferResult{
						float32Fn: func(name string) ([]float32, error) {
							return []float32{1.0}, nil
						},
					}, nil
				},
			}
			client.gc = mockTriton

			res, err := model.Grpc([]any{[]any{1.0, 2.0}}) // Testing []any positional input
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(HaveKey("output0"))
		})

		It("should handle dummy data generation when input is missing", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					return &mockInferResult{}, nil
				},
			}
			client.hc = mockTriton

			_, err := model.Http() // No data provided
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return error if client is not initialized", func() {
			_, err := model.Http()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not initialized"))
		})
	})
})

var _ = Describe("Inference Helper Functions", func() {
	var model *Model
	BeforeEach(func() {
		model = &Model{}
	})
	Context("convertData", func() {
		It("should return unchanged if already the correct slice type", func() {
			val := []float32{1.0, 2.0}
			res, err := model.convertData(val, "TYPE_FP32")
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(Equal(val))

			val64 := []int64{1, 2}
			res64, err := model.convertData(val64, "TYPE_INT64")
			Expect(err).NotTo(HaveOccurred())
			Expect(res64).To(Equal(val64))

			val8 := []int8{1, 2}
			res8, err := model.convertData(val8, "TYPE_INT8")
			Expect(err).NotTo(HaveOccurred())
			Expect(res8).To(Equal(val8))
		})

		It("should convert float64 slice to float32 slice for TYPE_FP32", func() {
			val := []any{1.0, 2.0}
			res, err := model.convertData(val, "TYPE_FP32")
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(BeAssignableToTypeOf([]float32{}))
			Expect(res).To(Equal([]float32{1.0, 2.0}))
		})

		It("should convert float64 or int64 slice to int64 slice for TYPE_INT64", func() {
			val := []any{1.0, int64(2)}
			res, err := model.convertData(val, "TYPE_INT64")
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(BeAssignableToTypeOf([]int64{}))
			Expect(res).To(Equal([]int64{1, 2}))
		})

		It("should return original value for unknown type or non-slice input", func() {
			res1, err1 := model.convertData("test", "TYPE_FP32")
			Expect(err1).NotTo(HaveOccurred())
			Expect(res1).To(Equal("test"))

			res2, err2 := model.convertData(123, "TYPE_INT64")
			Expect(err2).NotTo(HaveOccurred())
			Expect(res2).To(Equal(123))
		})

		It("should return error for unsupported element types", func() {
			val := []any{"unsupported"}
			_, err := model.convertData(val, "TYPE_FP32")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("cannot convert string to float32"))
		})
	})

	Context("generateDummyData", func() {
		It("should generate float32 dummy data", func() {
			res := model.generateDummyData("TYPE_FP32", []int64{2, 2})
			Expect(res).To(BeAssignableToTypeOf([]float32{}))
			Expect(res).To(HaveLen(4))
		})

		It("should generate int64 dummy data", func() {
			res := model.generateDummyData("TYPE_INT64", []int64{3})
			Expect(res).To(BeAssignableToTypeOf([]int64{}))
			Expect(res).To(HaveLen(3))
		})

		It("should generate int8 dummy data as default", func() {
			res := model.generateDummyData("TYPE_STRING", []int64{10})
			Expect(res).To(BeAssignableToTypeOf([]int8{}))
			Expect(res).To(HaveLen(10))
		})

		It("should handle -1 in dims", func() {
			res := model.generateDummyData("TYPE_FP32", []int64{-1, 5})
			Expect(res).To(HaveLen(5))
		})
	})

	Context("getDataBytes", func() {
		It("should return correct lengths for supported slice types", func() {
			Expect(model.getDataBytes([]float32{1, 2, 3})).To(Equal(12))
			Expect(model.getDataBytes([]int64{1, 2})).To(Equal(16))
			Expect(model.getDataBytes([]int8{1})).To(Equal(1))
			Expect(model.getDataBytes([]any{1, 2, 3, 4})).To(Equal(32))
		})

		It("should return 0 for unsupported types", func() {
			Expect(model.getDataBytes("test")).To(Equal(0))
			Expect(model.getDataBytes(123)).To(Equal(0))
		})
	})
})
