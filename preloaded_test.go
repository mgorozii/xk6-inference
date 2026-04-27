package inference

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/options"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"go.k6.io/k6/lib"
	"go.k6.io/k6/metrics"
)

var _ = Describe("Preloaded Inference", func() {
	var (
		client  *Client
		model   *Model
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
		mv := &mockVU{state: state, ctx: context.Background()}
		client = &Client{vu: mv, metrics: tm}
		model = &Model{
			c:    client,
			name: "test-model",
			config: &modelConfig{
				Input: []struct {
					Name     string  `json:"name"`
					DataType string  `json:"data_type"`
					Dims     []int64 `json:"dims"`
				}{
					{Name: "input0", DataType: "TYPE_FP32", Dims: []int64{1, 2}},
				},
				Output: []struct {
					Name     string `json:"name"`
					DataType string `json:"data_type"`
				}{
					{Name: "output0", DataType: "TYPE_FP32"},
				},
			},
		}
	})

	Context("LoadPayload", func() {
		It("should parse named map payload", func() {
			payload := map[string]any{"input0": []any{1.0, 2.0}}
			raw, _ := json.Marshal(payload)
			err := model.LoadPayload(string(raw))
			Expect(err).NotTo(HaveOccurred())
			Expect(model.preloadedHttp).To(HaveLen(1))
			Expect(model.preloadedHttp[0].GetName()).To(Equal("input0"))
			Expect(model.preloadedBytes).To(BeNumerically(">", 0))
		})

		It("should parse Triton V2 body payload", func() {
			payload := map[string]any{
				"inputs": []any{
					map[string]any{
						"name":     "input0",
						"datatype": "FP32",
						"data":     []any{3.0, 4.0},
					},
				},
			}
			raw, _ := json.Marshal(payload)
			err := model.LoadPayload(string(raw))
			Expect(err).NotTo(HaveOccurred())
			Expect(model.preloadedHttp).To(HaveLen(1))
		})

		It("should parse positional array payload", func() {
			payload := []any{[]any{5.0, 6.0}}
			raw, _ := json.Marshal(payload)
			err := model.LoadPayload(string(raw))
			Expect(err).NotTo(HaveOccurred())
			Expect(model.preloadedHttp).To(HaveLen(1))
		})

		It("should return error for invalid JSON", func() {
			err := model.LoadPayload("not-json")
			Expect(err).To(HaveOccurred())
		})

		It("should return error for missing inputs", func() {
			payload := map[string]any{"wrong_name": []any{1.0}}
			raw, _ := json.Marshal(payload)
			err := model.LoadPayload(string(raw))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("missing input"))
		})

		It("should infer a single dynamic dimension when payload size matches", func() {
			model.config.Input[0].Dims = []int64{-1, 2}
			err := model.LoadPayload(`{"input0": [1.0, 2.0, 3.0, 4.0]}`)
			Expect(err).NotTo(HaveOccurred())
			Expect(model.preloadedHttp).To(HaveLen(1))
		})

		It("should return explicit error when dynamic shape cannot be inferred exactly", func() {
			model.config.Input[0].Dims = []int64{-1, 3}
			err := model.LoadPayload(`{"input0": [1.0, 2.0]}`)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("cannot infer shape"))
		})

		It("should return explicit error for multiple dynamic dims", func() {
			model.config.Input[0].Dims = []int64{-1, -1, 3}
			err := model.LoadPayload(`{"input0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}`)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("multiple dynamic dims"))
		})

		It("should return error when dims is empty in model config", func() {
			model.config.Input[0].Dims = []int64{}
			err := model.LoadPayload(`{"input0": [1.0]}`)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("empty dims"))
		})

		It("should generate dummy inputs when no payload provided", func() {
			err := model.LoadPayload("")
			Expect(err).NotTo(HaveOccurred())
			Expect(model.preloadedHttp).To(HaveLen(1))
		})
	})

	Context("HttpPreloaded / GrpcPreloaded", func() {
		It("should use preloaded inputs for HTTP", func() {
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

			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())

			res, err := model.HttpPreloaded()
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(HaveKey("output0"))
			Expect(res.(map[string]any)["output0"]).To(Equal([]float32{0.5, 0.5}))

			Eventually(samples).Should(Receive())
			Eventually(samples).Should(Receive())
		})

		It("should use preloaded inputs for gRPC", func() {
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

			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())

			res, err := model.GrpcPreloaded()
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(HaveKey("output0"))
		})

		It("should return error if payload not loaded", func() {
			client.hc = &mockTritonClient{}
			_, err := model.HttpPreloaded()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not loaded"))
		})

		It("should return error if client not initialized", func() {
			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())
			_, err := model.HttpPreloaded()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not initialized"))
		})

		It("should propagate inference errors", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					return nil, errors.New("server error")
				},
			}
			client.hc = mockTriton

			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())

			_, err := model.HttpPreloaded()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("server error"))
		})

		It("should use dummy data for preloaded when empty payload", func() {
			mockTriton := &mockTritonClient{
				inferFn: func(ctx context.Context, modelName string, modelVersion string, inputs []base.InferInput, outputs []base.InferOutput, options *options.InferOptions) (base.InferResult, error) {
					Expect(inputs[0].GetData()).NotTo(BeNil())
					return &mockInferResult{}, nil
				},
			}
			client.hc = mockTriton

			Expect(model.LoadPayload("")).To(Succeed())

			_, err := model.HttpPreloaded()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return error on unsupported output type", func() {
			model.config.Output[0].DataType = "TYPE_BOOL"
			client.hc = &mockTritonClient{
				inferFn: func(_ context.Context, _ string, _ string, _ []base.InferInput, _ []base.InferOutput, _ *options.InferOptions) (base.InferResult, error) {
					return &mockInferResult{}, nil
				},
			}
			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())
			_, err := model.HttpPreloaded()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("unsupported output type"))
		})

		It("should return error on output decode failure", func() {
			client.hc = &mockTritonClient{
				inferFn: func(_ context.Context, _ string, _ string, _ []base.InferInput, _ []base.InferOutput, _ *options.InferOptions) (base.InferResult, error) {
					return &mockInferResult{
						float32Fn: func(name string) ([]float32, error) {
							return nil, errors.New("decode error")
						},
					}, nil
				},
			}
			Expect(model.LoadPayload(`{"input0": [1.0, 2.0]}`)).To(Succeed())
			_, err := model.HttpPreloaded()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("decode error"))
		})
	})
})
