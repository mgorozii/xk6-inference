import inference from 'k6/x/inference';
import { check, sleep } from 'k6';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Init stage: connect once per VU
const client = inference.connect('http://localhost:8000', 'localhost:8001');
const model = client.model('simple');

export const options = {
    vus: 1,
    iterations: 1,
};

export default function () {
    // 1. Auto-generate data
    model.http();
    
    // 2. Positional data (Array of inputs)
    // The 'simple' model has one input 'INPUT0' which is a vector [ -1 ]
    model.grpc([[1.0, 2.0, 3.0]]);

    // 3. Named data
    let res = model.http({ "INPUT0": [4.0, 5.0, 6.0] });

    check(res, { 'success': (r) => r.OUTPUT0 !== undefined });

    sleep(0.1);
}

export function handleSummary(data) {
    return {
        "report.html": htmlReport(data),
        "report.json": JSON.stringify(data),
        stdout: "Done!"
    };
}
