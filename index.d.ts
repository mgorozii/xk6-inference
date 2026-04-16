/**
 * **Inference k6 extension**
 *
 * @module inference
 */
export as namespace inference;

/**
 * Connect to Inference Server (Triton or KServe).
 * @param httpUrl The HTTP URL of the server (e.g. "http://localhost:8000")
 * @param grpcUrl The gRPC URL of the server (e.g. "localhost:8001")
 */
export function connect(httpUrl: string, grpcUrl: string): Client;

/**
 * Inference client.
 */
export interface Client {
    /**
     * Get a model by name.
     * @param name The name of the model.
     */
    model(name: string): Model;
}

/**
 * Inference model.
 */
export interface Model {
    /**
     * Perform inference via HTTP.
     * @param data Optional input data (map or positional array).
     */
    http(data?: any): any;

    /**
     * Perform inference via gRPC.
     * @param data Optional input data (map or positional array).
     */
    grpc(data?: any): any;
}
