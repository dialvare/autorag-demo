curl -s -X POST "http://llamastack-custom-distribution-service-llamastack.apps.ocp.c257c.sandbox1891.opentlc.com/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "vllm-inference/redhataiqwen3-8b-fp8-dynamic",
           "messages": [
             {
               "role": "user",
               "content": "What is OpenShift?"
             }
           ]
         }'