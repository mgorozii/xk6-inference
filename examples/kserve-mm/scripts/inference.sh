#!/bin/bash

# Configuration
NAMESPACE="modelmesh-serving"
MODEL_NAME="triton-model"
SERVICE_PORT=8008

echo "Port-forwarding to service..."
pkill -f "kubectl port-forward service/modelmesh-serving $SERVICE_PORT" || true
kubectl port-forward service/modelmesh-serving $SERVICE_PORT:$SERVICE_PORT -n $NAMESPACE > /dev/null 2>&1 &
PF_PID=$!

# Give it a second to start
sleep 2

# Input3 is the correct name for this ONNX model
INPUT_NAME="Input3"

echo "Sending inference request to $MODEL_NAME (Input: $INPUT_NAME)..."

# Generate dummy data (784 floats)
DATA=$(python3 -c "import json; print(json.dumps([0.5]*784))")

# V2 Inference Protocol request
RESPONSE=$(curl -s -X POST "http://localhost:$SERVICE_PORT/v2/models/$MODEL_NAME/infer" \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"$INPUT_NAME\",
        \"shape\": [1, 1, 28, 28],
        \"datatype\": \"FP32\",
        \"data\": $DATA
      }
    ]
  }")

echo "Response:"
echo "$RESPONSE" | jq .

# Kill port-forward
kill $PF_PID
