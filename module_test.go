package inference

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"go.k6.io/k6/js/common"
	"go.k6.io/k6/js/modules"
	"go.k6.io/k6/lib"
	"go.k6.io/k6/metrics"
)

var _ = Describe("Module", func() {
	var (
		root *rootModule
		mv   *mockVU
	)

	BeforeEach(func() {
		root = &rootModule{}
		registry := metrics.NewRegistry()
		state := &lib.State{
			BuiltinMetrics: metrics.RegisterBuiltinMetrics(registry),
		}
		initEnv := &common.InitEnvironment{
			TestPreInitState: &lib.TestPreInitState{
				Registry: registry,
			},
		}
		mv = &mockVU{
			state: state,
		}
		mv.VU = &stubVU{initEnv: initEnv}
	})

	It("should create a new module instance", func() {
		instance := root.NewModuleInstance(mv)
		Expect(instance).NotTo(BeNil())

		m, ok := instance.(*module)
		Expect(ok).To(BeTrue())
		Expect(m.metrics.Reqs).NotTo(BeNil())
		Expect(m.metrics.Duration).NotTo(BeNil())
	})

	It("should connect to inference server", func() {
		instance := root.NewModuleInstance(mv).(*module)
		client, err := instance.Connect("http://localhost:8000", "localhost:8001")
		Expect(err).NotTo(HaveOccurred())
		Expect(client).NotTo(BeNil())
		Expect(client.httpURL).To(Equal("http://localhost:8000"))
	})
})

type stubVU struct {
	modules.VU
	initEnv *common.InitEnvironment
}

func (s *stubVU) InitEnv() *common.InitEnvironment {
	return s.initEnv
}
