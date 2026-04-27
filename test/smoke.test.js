import inference from 'k6/x/inference';
import { check } from 'k6';

// No external dependencies
const client = inference.connect('', '');
const model = client.model('dummy');

export const options = { vus: 1, iterations: 1 };

export default function () {
    check(client, { 'client ok': (c) => c !== null });
    check(model, { 'model ok': (m) => m !== null });

    let errOk = false;
    try {
        model.http();
    } catch (e) {
        // Expected: no HTTP URL
        if (e.message.includes('not initialized')) errOk = true;
    }
    check(errOk, { 'http throws without server': (e) => e === true });
}
