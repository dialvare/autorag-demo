curl -s -X POST "http://localhost:8321/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "vllm-inference/redhataiqwen3-8b-fp8-dynamic",
           "messages": [
             {
               "role": "system",
               "content": "You are the official assistant for Pizza Bank. Respond in English, clearly, and based SOLELY on the following context extracted from our database. If the answer is not in the context, say that you do not know.\n\nCONTEXT:<INSERT RAG RETRIEVAL HERE>"
             },
             {
               "role": "user",
               "content": "What are the requirements for the Youth Account and how much does it cost to renew the Gold Card?"
             }
           ]
         }'
