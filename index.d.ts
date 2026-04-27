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
     * @param data Optional input data (map, positional array, or Triton/KServe V2 body).
     */
    http(data?: any): any;

    /**
     * Perform inference via gRPC.
     * @param data Optional input data (map, positional array, or Triton/KServe V2 body).
     */
    grpc(data?: any): any;

    /**
     * Parse and cache input payload once during init.
     * Pass empty string to generate dummy data.
     * @param jsonPayload JSON string of input data.
     */
    loadPayload(jsonPayload: string): void;

    /**
     * Perform inference via HTTP using preloaded payload.
     */
    httpPreloaded(): any;

    /**
     * Perform inference via gRPC using preloaded payload.
     */
    grpcPreloaded(): any;
}
