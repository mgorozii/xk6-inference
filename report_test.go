package inference

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Integration Report", func() {
	var (
		server *http.Server
		url    string
		tmpDir string
	)

	BeforeEach(func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/v2/models/simple/config", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{
				"name": "simple",
				"input": [{"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [1]}],
				"output": [{"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [1]}]
			}`))
		})
		mux.HandleFunc("/v2/models/simple/infer", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{
				"model_name": "simple",
				"outputs": [{"name": "OUTPUT0", "datatype": "FP32", "shape": [1], "data": [1.0]}]
			}`))
		})

		listener, err := net.Listen("tcp", "127.0.0.1:0")
		Expect(err).NotTo(HaveOccurred())
		url = fmt.Sprintf("http://%s", listener.Addr().String())

		server = &http.Server{Handler: mux}
		go func() {
			_ = server.Serve(listener)
		}()
		tmpDir = GinkgoT().TempDir()
	})

	AfterEach(func() {
		_ = server.Close()
	})

	It("should generate a report with valid metrics", func() {
		if _, err := os.Stat("./k6"); os.IsNotExist(err) {
			Skip("k6 binary not found, skipping integration test")
		}

		script := `
import inference from 'k6/x/inference';
import { check } from 'k6';

export const options = { vus: 1, iterations: 1 };

export default function () {
    const client = inference.connect("` + url + `", "");
    const model = client.model('simple');
    const res = model.http({ "INPUT0": [1.0] });
    check(res, { 'status is ok': (r) => r !== null && r.OUTPUT0 !== undefined });
}
`
		scriptPath := filepath.Join(tmpDir, "test_report.js")
		reportPath := filepath.Join(tmpDir, "report.json")

		err := os.WriteFile(scriptPath, []byte(script), 0600)
		Expect(err).NotTo(HaveOccurred())

		cmd := exec.Command("./k6", "run", "--summary-export="+reportPath, scriptPath)
		output, err := cmd.CombinedOutput()
		if err != nil {
			GinkgoWriter.Printf("k6 output: %s\n", string(output))
		}
		Expect(err).NotTo(HaveOccurred())

		data, err := os.ReadFile(reportPath)
		Expect(err).NotTo(HaveOccurred())

		var report struct {
			Metrics map[string]struct {
				Count float64 `json:"count"`
				Avg   float64 `json:"avg"`
				Value float64 `json:"value"`
			} `json:"metrics"`
		}
		err = json.Unmarshal(data, &report)
		Expect(err).NotTo(HaveOccurred(), "Failed to unmarshal report: %s", string(data))

		Expect(report.Metrics).To(HaveKey("inference_reqs"), "Missing inference_reqs in %v", report.Metrics)
		Expect(report.Metrics["inference_reqs"].Count).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("inference_req_duration"))
		Expect(report.Metrics["inference_req_duration"].Avg).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("http_reqs"))
		Expect(report.Metrics["http_reqs"].Count).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("http_req_duration"))
		Expect(report.Metrics["http_req_duration"].Avg).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("http_req_failed"))
		Expect(report.Metrics["http_req_failed"].Value).To(BeNumerically("==", 0))

		Expect(report.Metrics).To(HaveKey("data_sent"))
		Expect(report.Metrics["data_sent"].Count).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("data_received"), "Missing data_received in %v", report.Metrics)
		Expect(report.Metrics["data_received"].Count).To(BeNumerically(">", 0))

		Expect(report.Metrics).To(HaveKey("checks"))
		Expect(report.Metrics["checks"].Value).To(BeNumerically("==", 1))
	})
})
