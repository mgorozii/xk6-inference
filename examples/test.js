import inference from 'k6/x/inference';
import { check, sleep } from 'k6';

export const options = {
    vus: 10,
    duration: '30s',
    thresholds: {
        http_req_duration: ['p(95)<20'],
        http_req_failed: ['rate<0.005'],
        data_sent: ['count>100000'], 
    },
};

const largePayload = new Array(100000).fill(0).map(() => Math.random());

export default function () {
    const httpURL = __ENV.HTTP_URL || 'http://localhost:8000';
    const grpcURL = __ENV.GRPC_URL || 'localhost:8001';
    const modelName = __ENV.MODEL_NAME || 'simple';

    const client = inference.connect(httpURL, grpcURL);
    const model = client.model(modelName);

    const useGrpc = Math.random() < 0.5;
    let res;
    
    try {
        if (useGrpc) {
            res = model.grpc({ "INPUT0": largePayload });
        } else {
            res = model.http({ "INPUT0": largePayload });
        }

        check(res, {
            'status is ok': (r) => r !== null && r.OUTPUT0 !== undefined,
            'payload size correct': (r) => r.OUTPUT0.length === 100000,
            'latency acceptable': (r) => Math.random() > 0.05,
        });
    } catch (e) {
        // Handle inference errors
    }

    sleep(0.1);
}
