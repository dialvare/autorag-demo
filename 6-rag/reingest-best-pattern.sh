#!/usr/bin/env bash
# Declarative path (preferred for workshop recreate):
#   oc apply -f 6-rag/best-pattern-ingest.yaml
#   oc wait -n llamastack --for=condition=complete job/best-pattern-rag-ingest --timeout=300s
#   oc apply -f 8-app/deployment.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-llamastack}"

oc delete job best-pattern-rag-ingest -n "$NAMESPACE" --ignore-not-found
oc apply -f "${SCRIPT_DIR}/best-pattern-ingest.yaml"
oc wait -n "$NAMESPACE" --for=condition=complete job/best-pattern-rag-ingest --timeout=300s
oc logs -n "$NAMESPACE" job/best-pattern-rag-ingest

echo
echo "App ConfigMap uses VECTOR_STORE_ID=pizza-bank-best-pattern (8-app/deployment.yaml)."
