# Pizza Bank RAG Demo with OGX on OpenShift

End-to-end Retrieval-Augmented Generation (RAG) demo for a fictional bank ("Pizza Bank") running on OpenShift. It combines OGX as the AI gateway, AutoRAG for document ingestion and evaluation, Milvus as the vector database, an MCP server for live database queries, and a Streamlit chatbot as the user-facing application.

## Prerequisites

- OpenShift cluster with OpenShift AI installed
- OGX operator installed
- `oc` CLI authenticated to the cluster
- LLM and embedding models served via vLLM (e.g., `redhataiqwen3-8b-fp8-dynamic` and `redhataiembeddinggemma-300m`)
- `podman` for building the chatbot image

## Repository Structure

```
2-storage/
├── milvus-setup.yaml          # Milvus standalone + etcd + secret
├── postgresql-setup.yaml      # PostgreSQL for OGX state
├── minio-setup.yaml           # MinIO for document storage
├── minio-job.yaml             # Job to upload documents to MinIO
├── input_data/                # RAG source documents (accounts.txt, cards.txt)
└── benchmark_data.json        # Evaluation benchmark dataset

4-ogx/
└── deployment.yaml            # OGXServer CR

5-autorag/
├── knowledge-connection.yaml  # S3 connection secret for AutoRAG
└── pipeline-server.yaml       # Data Science Pipelines Application (DSPA)

6-mcp/
├── mariadb.yaml               # MariaDB deployment + seed data + credentials
├── mariadb-mcp.yaml           # MCPServer CR (MariaDB MCP server)
├── mariadb-mcp-logs-patch.yaml
└── mcp-lifecycle-operator.yaml # MCP Lifecycle Operator CRD + controller

7-app/
├── deployment-mariadb.yaml    # Chatbot Deployment + ConfigMap + Service + Route
└── ogx-mariadb-connection.yaml # Merge patch to add MCP_SERVER_URL to OGX

source-code-chatbot/
├── app.py                     # Streamlit chatbot application
├── requirements.txt           # Python dependencies
└── Dockerfile                 # Container build (UBI9 + Python 3.12)
```

## Building the Chatbot Image

```bash
cd source-code-chatbot
podman build --platform linux/amd64 -t quay.io/<your-user>/chatbot:latest .
podman push quay.io/<your-user>/chatbot:latest
```

Update the image reference in `7-app/deployment-mariadb.yaml` accordingly.

## Chatbot Features

- RAG over Pizza Bank documents via AutoRAG + Milvus (vector and hybrid search)
- MCP tool calling for live MariaDB queries (client accounts, transactions)
- Configurable sidebar: model selection, temperature, vector store, search mode, ranker settings
- OpenAI-compatible `/v1/responses` API
