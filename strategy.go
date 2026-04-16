package inference

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

type ModelProvider interface {
	GetModelConfig(client *http.Client, baseURL string, modelName string) (*modelConfig, error)
}

type TritonConfigProvider struct{}

func (p *TritonConfigProvider) GetModelConfig(client *http.Client, baseURL string, modelName string) (*modelConfig, error) {
	url := fmt.Sprintf("%s/v2/models/%s/config", baseURL, modelName)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("triton config status %d", resp.StatusCode)
	}

	var config modelConfig
	if err := json.NewDecoder(resp.Body).Decode(&config); err != nil {
		return nil, err
	}
	return &config, nil
}

type KServeMetadataProvider struct{}

type shapeDim int64

func (d *shapeDim) UnmarshalJSON(data []byte) error {
	var intValue int64
	if err := json.Unmarshal(data, &intValue); err == nil {
		*d = shapeDim(intValue)
		return nil
	}

	var stringValue string
	if err := json.Unmarshal(data, &stringValue); err != nil {
		return err
	}

	parsed, err := strconv.ParseInt(stringValue, 10, 64)
	if err != nil {
		return err
	}

	*d = shapeDim(parsed)
	return nil
}

func (p *KServeMetadataProvider) GetModelConfig(client *http.Client, baseURL string, modelName string) (*modelConfig, error) {
	url := fmt.Sprintf("%s/v2/models/%s", baseURL, modelName)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("kserve metadata status %d", resp.StatusCode)
	}

	var meta struct {
		Inputs []struct {
			Name     string     `json:"name"`
			Datatype string     `json:"datatype"`
			Shape    []shapeDim `json:"shape"`
		} `json:"inputs"`
		Outputs []struct {
			Name     string     `json:"name"`
			Datatype string     `json:"datatype"`
			Shape    []shapeDim `json:"shape"`
		} `json:"outputs"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return nil, err
	}

	config := &modelConfig{}
	for _, in := range meta.Inputs {
		dims := make([]int64, len(in.Shape))
		for index, dim := range in.Shape {
			dims[index] = int64(dim)
		}
		config.Input = append(config.Input, struct {
			Name     string  `json:"name"`
			DataType string  `json:"data_type"`
			Dims     []int64 `json:"dims"`
		}{
			Name:     in.Name,
			DataType: "TYPE_" + in.Datatype,
			Dims:     dims,
		})
	}
	for _, out := range meta.Outputs {
		config.Output = append(config.Output, struct {
			Name     string `json:"name"`
			DataType string `json:"data_type"`
		}{
			Name:     out.Name,
			DataType: "TYPE_" + out.Datatype,
		})
	}

	return config, nil
}

type AutoDetectProvider struct {
	Triton TritonConfigProvider
	KServe KServeMetadataProvider
}

func (p *AutoDetectProvider) GetModelConfig(client *http.Client, baseURL string, modelName string) (*modelConfig, error) {
	config, err := p.Triton.GetModelConfig(client, baseURL, modelName)
	if err == nil {
		return config, nil
	}
	kserveConfig, kserveErr := p.KServe.GetModelConfig(client, baseURL, modelName)
	if kserveErr == nil {
		return kserveConfig, nil
	}
	return nil, fmt.Errorf("failed to get model config: triton error: %v, kserve error: %v", err, kserveErr)
}
