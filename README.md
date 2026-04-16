# xk6-inference

k6 extension for performance testing [Inference Servers](https://github.com/triton-inference-server/server) like NVIDIA Triton and KServe.

## Features

- **Auto-Config**: Automatically fetches model configuration (shapes, datatypes) from Triton's `/config` or KServe's metadata endpoint.
- **Dual Protocol**: Supports both HTTP and gRPC for inference.
- **Metrics**: Built-in metrics for request count, duration, and data throughput.
- **Concise API**: Simple and idiomatic JavaScript API.

## Installation

To build a custom k6 binary with this extension:

```bash
xk6 build --with github.com/mgorozii/xk6-inference
```

## Usage

```javascript
import inference from 'k6/x/inference';
import { check } from 'k6';

export default function () {
    // Connect to Inference Server (HTTP and gRPC)
    const client = inference.connect('http://localhost:8000', 'localhost:8001');

    // Load model (automatically fetches config/metadata via HTTP)
    const model = client.model('simple');

    // Perform inference via HTTP
    let res = model.http({
        "INPUT0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "INPUT1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    });
    
    // Perform inference via gRPC
    res = model.grpc({
        "INPUT0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "INPUT1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    });

    check(res, {
        'success': (r) => r.OUTPUT0 !== undefined
    });
}
```

## API

### `inference.connect(httpUrl, grpcUrl)`
Creates an inference client.
- `httpUrl`: Required for fetching configuration and HTTP inference.
- `grpcUrl`: Required for gRPC inference.

### `client.model(name)`
Fetches model configuration/metadata and returns a `Model` object.

### `model.http(data)`
Performs inference using HTTP.
- `data` (optional): Object where keys are input names and values are data arrays. If omitted, dummy data is generated based on model config.

### `model.grpc(data)`
Performs inference using gRPC.
- `data` (optional): Same as `model.http`.

## Metrics

The extension emits the following metrics:

- `inference_reqs`: Counter of inference requests.
- `inference_req_duration`: Trend of request duration.
- `data_sent`: Built-in k6 metric for total bytes sent.
- `data_received`: Built-in k6 metric for total bytes received.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
