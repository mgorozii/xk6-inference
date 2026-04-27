#!/bin/bash
# Universal script for KServe V2 REST API calls (ModelMesh optimized)

NAMESPACE="modelmesh-serving"
MODEL_NAME="triton-model"
SERVICE_PORT=8008

COMMAND=$1
shift

# Check if port-forward is running
pkill -f "kubectl port-forward service/modelmesh-serving $SERVICE_PORT" || true
kubectl port-forward service/modelmesh-serving $SERVICE_PORT:$SERVICE_PORT -n $NAMESPACE > /dev/null 2>&1 &
PF_PID=$!
sleep 2

case $COMMAND in
    "server-info")
        echo "--- Server Info (via Model Metadata) ---"
        curl -s "http://localhost:$SERVICE_PORT/v2/models/$MODEL_NAME" | jq '{name, versions, platform}'
        ;;
    "model-metadata")
        echo "--- Model Metadata ($MODEL_NAME) ---"
        curl -s "http://localhost:$SERVICE_PORT/v2/models/$MODEL_NAME" | jq .
        ;;
    "model-ready")
        echo "--- Model Readiness ($MODEL_NAME) ---"
        # In ModelMesh, if metadata returns 200, it's generally ready
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$SERVICE_PORT/v2/models/$MODEL_NAME")
        if [ "$HTTP_CODE" == "200" ]; then
            echo "HTTP 200 OK"
        else
            echo "HTTP $HTTP_CODE"
        fi
        ;;
    *)
        echo "Usage: $0 {server-info|model-metadata|model-ready}"
        ;;
esac

kill $PF_PID
