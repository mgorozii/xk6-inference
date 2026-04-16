package inference

import (
	"net/http"
	"net/http/httptest"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Model Providers", func() {
	var (
		client *http.Client
	)

	BeforeEach(func() {
		client = http.DefaultClient
	})

	Context("TritonConfigProvider", func() {
		It("should fetch triton config correctly", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				Expect(r.URL.Path).To(Equal("/v2/models/test-model/config"))
				_, _ = w.Write([]byte(`{"input":[{"name":"input0","data_type":"TYPE_FP32","dims":[1,224,224,3]}],"output":[{"name":"output0","data_type":"TYPE_FP32"}]}`))
			}))
			defer server.Close()

			provider := &TritonConfigProvider{}
			config, err := provider.GetModelConfig(client, server.URL, "test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(config.Input).To(HaveLen(1))
			Expect(config.Input[0].Name).To(Equal("input0"))
			Expect(config.Input[0].DataType).To(Equal("TYPE_FP32"))
		})
	})

	Context("KServeMetadataProvider", func() {
		It("should fetch kserve metadata and convert to modelConfig", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				Expect(r.URL.Path).To(Equal("/v2/models/test-model"))
				_, _ = w.Write([]byte(`{
					"name": "test-model",
					"inputs": [{"name": "in0", "datatype": "FP32", "shape": [1, 10]}],
					"outputs": [{"name": "out0", "datatype": "FP32", "shape": [1, 1]}]
				}`))
			}))
			defer server.Close()

			provider := &KServeMetadataProvider{}
			config, err := provider.GetModelConfig(client, server.URL, "test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(config.Input).To(HaveLen(1))
			Expect(config.Input[0].Name).To(Equal("in0"))
			Expect(config.Input[0].DataType).To(Equal("TYPE_FP32"))
			Expect(config.Input[0].Dims).To(Equal([]int64{1, 10}))
		})

		It("should accept stringified shapes from modelmesh metadata", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				Expect(r.URL.Path).To(Equal("/v2/models/test-model"))
				_, _ = w.Write([]byte(`{
					"name": "test-model",
					"inputs": [{"name": "in0", "datatype": "FP32", "shape": ["-1", "3", "224", "224"]}],
					"outputs": [{"name": "out0", "datatype": "FP32", "shape": ["-1", "1000"]}]
				}`))
			}))
			defer server.Close()

			provider := &KServeMetadataProvider{}
			config, err := provider.GetModelConfig(client, server.URL, "test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(config.Input).To(HaveLen(1))
			Expect(config.Input[0].Dims).To(Equal([]int64{-1, 3, 224, 224}))
		})
	})

	Context("AutoDetectProvider", func() {
		It("should try Triton first then KServe", func() {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.URL.Path == "/v2/models/test-model/config" {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if r.URL.Path == "/v2/models/test-model" {
					_, _ = w.Write([]byte(`{"inputs": [{"name": "in0", "datatype": "FP32"}]}`))
					return
				}
			}))
			defer server.Close()

			provider := &AutoDetectProvider{}
			config, err := provider.GetModelConfig(client, server.URL, "test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(config.Input[0].Name).To(Equal("in0"))
			Expect(calls).To(Equal(2))
		})

		It("should fallback to KServe when metadata shape is stringified", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v2/models/test-model/config":
					w.WriteHeader(http.StatusNotFound)
				case "/v2/models/test-model":
					_, _ = w.Write([]byte(`{
						"inputs": [{"name": "in0", "datatype": "FP32", "shape": ["-1", "3", "224", "224"]}],
						"outputs": [{"name": "out0", "datatype": "FP32", "shape": ["-1", "1000"]}]
					}`))
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()

			provider := &AutoDetectProvider{}
			config, err := provider.GetModelConfig(client, server.URL, "test-model")
			Expect(err).NotTo(HaveOccurred())
			Expect(config.Input[0].Dims).To(Equal([]int64{-1, 3, 224, 224}))
		})
	})
})
