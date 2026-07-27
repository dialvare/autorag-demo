import os
import re

import requests
import streamlit as st

HOST = os.getenv("LLAMA_STACK_HOST", "llama-stack-demo")
PORT = os.getenv("LLAMA_STACK_PORT", "8321")
SCHEME = (
    "https"
    if os.getenv("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
    else "http"
)
BASE_URL = f"{SCHEME}://{HOST}:{PORT}/v1"
API_KEY = (
    os.getenv("LLAMA_STACK_CLIENT_API_KEY", "").strip()
    or os.getenv("LLAMA_STACK_API_KEY", "").strip()
)
DEFAULT_MODEL = os.getenv("INFERENCE_MODEL", "redhataiqwen3-8b-fp8-dynamic")
# Optional preferred AutoRAG vs_* id; leave empty so the UI picks the first listed store.
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE_ID", "").strip()
MCP_SERVER_LABEL = os.getenv("MCP_SERVER_LABEL", "openshift-mcp-server")
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://openshift-mcp-deployment.llamastack.svc.cluster.local:8080/mcp",
)
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-service.llamastack.svc.cluster.local")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
REQUEST_TIMEOUT = int(os.getenv("LLAMA_STACK_REQUEST_TIMEOUT", "30"))
TURN_TIMEOUT = int(os.getenv("LLAMA_STACK_TURN_TIMEOUT", "300"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_STACK_MAX_OUTPUT_TOKENS", "2048"))
# MCP tool schemas alone can consume ~2k tokens; keep headroom under small
# model context windows (e.g. VLLM_MAX_TOKENS=4096).
MCP_MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_STACK_MCP_MAX_OUTPUT_TOKENS", "1024"))
MAX_RESPONSE_CONTINUATIONS = int(os.getenv("LLAMA_STACK_MAX_RESPONSE_CONTINUATIONS", "2"))
RAG_MAX_RESULTS = int(os.getenv("LLAMA_STACK_RAG_MAX_RESULTS", "5"))
# Used only for the OpenAI file_search path (stores with indexed files).
RAG_RANKER = os.getenv("LLAMA_STACK_RAG_RANKER", "weighted").strip()
RAG_RANKER_ALPHA = float(os.getenv("LLAMA_STACK_RAG_RANKER_ALPHA", "0.5"))
# Restrict MCP schemas so prompt+answer fit; override with comma-separated names.
MCP_ALLOWED_TOOLS = [
    t.strip()
    for t in os.getenv(
        "MCP_ALLOWED_TOOLS",
        "pods_list_in_namespace,pods_get,pods_list,nodes_top,events_list,namespaces_list",
    ).split(",")
    if t.strip()
]
RAG_INSTRUCTIONS = (
    "Answer Pizza Bank product and policy questions using only information found in "
    "the retrieved context. Cite the retrieved documents. If the retrieved context "
    "has no relevant chunks, say the knowledge base has no matching information."
)
MCP_INSTRUCTIONS = (
    "For OpenShift/Kubernetes questions, call the matching MCP tool immediately "
    "(e.g. pods_list_in_namespace for pods in a namespace, nodes_top for nodes). "
    "Do not narrate your plan. For pods_list_in_namespace pass ONLY the namespace "
    "unless the user asks to filter; never set fieldSelector to bare 'status.phase'. "
    "If a tool errors, retry once with simpler arguments. After tool results arrive, "
    "prefer a short summary: counts by phase, then list only Running/Pending/Failed/"
    "Error pods (skip long Completed job pods unless asked)."
)
COMBINED_TOOL_INSTRUCTIONS = (
    "Choose the right tool for each question. Use retrieved knowledge-base context "
    "only for Pizza Bank products, accounts, cards, fees, and policies. Use OpenShift "
    "MCP tools for Kubernetes or cluster infrastructure questions (nodes, pods, events, "
    "cluster status). Never answer infrastructure questions from the knowledge base "
    "or claim it lacks node or cluster data—call MCP tools instead."
)
BASE_INSTRUCTIONS = (
    "You are a corporate assistant for Pizza Bank. Reply directly to the user in "
    "plain language. Never include internal reasoning, <think> blocks, or step-by-step "
    "planning. If a tool is needed, call it immediately with no preamble. "
    "Keep answers concise but complete. /no_think"
)


def _build_instructions(*, enable_rag, enable_mcp, retrieved_context=None):
    parts = [BASE_INSTRUCTIONS]
    if enable_rag and enable_mcp:
        parts.append(COMBINED_TOOL_INSTRUCTIONS)
        parts.append(MCP_INSTRUCTIONS)
    elif enable_rag:
        parts.append(RAG_INSTRUCTIONS)
    elif enable_mcp:
        parts.append(MCP_INSTRUCTIONS)
    if retrieved_context:
        parts.append(
            "Retrieved knowledge-base context:\n"
            f"{retrieved_context}\n"
            "Use only this context for Pizza Bank product/policy answers."
        )
    return " ".join(parts)


_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
    re.DOTALL | re.IGNORECASE,
)
# Qwen often gets truncated mid-thought when max_output_tokens is low.
_INCOMPLETE_THINKING_RE = re.compile(
    r"<(?:think|redacted_thinking)>.*$",
    re.DOTALL | re.IGNORECASE,
)

st.set_page_config(page_title="Llama Stack Agent", page_icon="🦙", layout="wide")


def _headers():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _request_error_detail(exc):
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = exc.response.text.strip()
        if "maximum context length" in body:
            return (
                "The model context window was exceeded. With MCP enabled, tool schemas "
                "alone can use ~2k tokens — if `VLLM_MAX_TOKENS` is 4096 and "
                "`max_output_tokens` is too high, even *hi* fails. "
                "Turn MCP off for simple chat, or raise `VLLM_MAX_TOKENS` (e.g. 8192+) "
                "and keep MCP output tokens low. For cluster questions, ask something "
                "narrow such as *How are my nodes doing?*"
            )
        if body:
            return f"{exc} — {body[:300]}"
    return str(exc)


def _ls_request(method, path, *, json=None, timeout=REQUEST_TIMEOUT):
    return requests.request(
        method,
        f"{BASE_URL}{path}",
        json=json,
        headers=_headers(),
        timeout=timeout,
    )


def _extract_id(item):
    return item.get("id") or item.get("identifier")


def _resolve_model_id(model_id, available_models):
    if model_id in available_models:
        return model_id
    for candidate in available_models:
        if candidate.endswith(f"/{model_id}") or candidate.endswith(model_id):
            return candidate
    return model_id


def _default_index(items, preferred):
    resolved = _resolve_model_id(preferred, items)
    if resolved in items:
        return items.index(resolved)
    for index, item in enumerate(items):
        if preferred in item:
            return index
    return 0


def _store_embedding_model(store):
    if not store:
        return None
    metadata = store.get("metadata") or {}
    return (metadata.get("embedding_model") or "").strip() or None


def _store_file_count(store):
    if not store:
        return 0
    counts = store.get("file_counts") or {}
    return int(counts.get("completed") or counts.get("total") or store.get("completed_files") or 0)


def _milvus_collection_name(vs_id):
    return vs_id.replace("-", "_")


def _embed_query(model, text):
    res = _ls_request(
        "POST",
        "/embeddings",
        json={"model": model, "input": text},
        timeout=REQUEST_TIMEOUT,
    )
    res.raise_for_status()
    data = res.json().get("data") or []
    if not data or not data[0].get("embedding"):
        raise RuntimeError(f"No embedding returned for model {model}")
    return data[0]["embedding"]


def _milvus_search(collection, vector, top_k):
    from pymilvus import Collection, connections, utility

    alias = "chatbot"
    if not connections.has_connection(alias):
        connections.connect(alias=alias, host=MILVUS_HOST, port=str(MILVUS_PORT))
    if not utility.has_collection(collection, using=alias):
        raise RuntimeError(
            f"Milvus collection `{collection}` not found on {MILVUS_HOST}:{MILVUS_PORT}"
        )
    col = Collection(collection, using=alias)
    col.load()
    hits = col.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["chunk_id", "content"],
    )
    results = []
    for hit_group in hits:
        for hit in hit_group:
            content = hit.entity.get("content") or ""
            if isinstance(content, str) and content.strip():
                results.append(
                    {
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": content.strip(),
                        "score": float(hit.score),
                    }
                )
    return results


def _retrieve_rag_context(vs_id, query, store):
    """Embed with the store's model and search AutoRAG Milvus chunks."""
    embedding_model = _store_embedding_model(store)
    if not embedding_model:
        raise RuntimeError(
            f"Vector store `{vs_id}` has no metadata.embedding_model; "
            "cannot retrieve AutoRAG chunks."
        )
    vector = _embed_query(embedding_model, query)
    collection = _milvus_collection_name(vs_id)
    chunks = _milvus_search(collection, vector, RAG_MAX_RESULTS)
    return {
        "embedding_model": embedding_model,
        "collection": collection,
        "chunks": chunks,
        "context_text": "\n\n---\n\n".join(c["content"] for c in chunks),
    }


def _uses_file_search(store):
    """OpenAI file_search only works when Llama Stack has indexed files."""
    return _store_file_count(store) > 0


@st.cache_data(ttl=60)
def get_models():
    try:
        res = _ls_request("GET", "/models")
        res.raise_for_status()
        models = [_extract_id(m) for m in res.json().get("data", []) if _extract_id(m)]
        llm_models = [
            m
            for m in models
            if "embed" not in m.lower() and "embedding" not in m.lower()
        ]
        models = llm_models or models
        if models:
            return models, None
        return [_resolve_model_id(DEFAULT_MODEL, [])], "Llama Stack returned no models."
    except Exception as exc:
        return [DEFAULT_MODEL], _request_error_detail(exc)


@st.cache_data(ttl=60)
def get_vector_stores():
    fallback = []
    if DEFAULT_VECTOR_STORE:
        fallback = [
            {
                "id": DEFAULT_VECTOR_STORE,
                "name": DEFAULT_VECTOR_STORE,
                "label": DEFAULT_VECTOR_STORE,
                "completed_files": 0,
                "metadata": {},
                "file_counts": {"completed": 0, "total": 0},
            }
        ]
    try:
        res = _ls_request("GET", "/vector_stores")
        res.raise_for_status()
        stores = []
        for item in res.json().get("data", []):
            store_id = _extract_id(item)
            if not store_id:
                continue
            name = (item.get("name") or "").strip()
            metadata = item.get("metadata") or {}
            file_counts = item.get("file_counts") or {}
            completed_files = int(file_counts.get("completed") or 0)
            embedding_model = (metadata.get("embedding_model") or "").strip()
            emb_short = embedding_model.split("/")[-1] if embedding_model else "unknown-emb"
            base = f"{name} ({store_id})" if name else store_id
            label = f"{base} · {emb_short} · {completed_files} files"
            stores.append(
                {
                    "id": store_id,
                    "name": name or store_id,
                    "label": label,
                    "completed_files": completed_files,
                    "metadata": metadata,
                    "file_counts": file_counts,
                }
            )
        # Prefer stores with indexed files, then by name/id.
        stores.sort(key=lambda s: (-s["completed_files"], s["name"], s["id"]))
        if stores:
            return stores, None
        return fallback, "Llama Stack returned no vector stores."
    except Exception as exc:
        return fallback, _request_error_detail(exc)


def _pick_default_vector_store(stores):
    if not stores:
        return None
    if DEFAULT_VECTOR_STORE:
        for store in stores:
            if store["id"] == DEFAULT_VECTOR_STORE or store["name"] == DEFAULT_VECTOR_STORE:
                return store["id"]
    for store in stores:
        if store["completed_files"] > 0:
            return store["id"]
    return stores[0]["id"]


@st.cache_data(ttl=60)
def get_builtin_tools():
    try:
        res = _ls_request("GET", "/tools")
        res.raise_for_status()
        groups = set()
        for tool in res.json().get("data", []):
            toolgroup_id = tool.get("toolgroup_id")
            if toolgroup_id and toolgroup_id.startswith("builtin::"):
                groups.add(toolgroup_id)
        return sorted(groups), None
    except Exception as exc:
        return ["builtin::websearch"], _request_error_detail(exc)


def _build_response_tools(
    *, enable_rag, selected_vstore, selected_store, enable_websearch, enable_mcp
):
    tools = []

    # file_search only when LS has OpenAI-indexed files; AutoRAG stores use Milvus bridge.
    if enable_rag and selected_vstore and _uses_file_search(selected_store):
        file_search = {
            "type": "file_search",
            "vector_store_ids": [selected_vstore],
            "max_num_results": RAG_MAX_RESULTS,
        }
        if RAG_RANKER:
            file_search["ranking_options"] = {
                "ranker": RAG_RANKER,
                "alpha": RAG_RANKER_ALPHA,
            }
        tools.append(file_search)

    if enable_websearch:
        tools.append({"type": "web_search"})

    if enable_mcp and MCP_SERVER_URL:
        mcp_tool = {
            "type": "mcp",
            "server_label": MCP_SERVER_LABEL,
            "server_description": "OpenShift MCP server deployed via the MCP catalog",
            "server_url": MCP_SERVER_URL,
            "require_approval": "never",
        }
        if MCP_ALLOWED_TOOLS:
            mcp_tool["allowed_tools"] = MCP_ALLOWED_TOOLS
        tools.append(mcp_tool)

    return tools


def _clean_model_text(text):
    cleaned = _THINKING_BLOCK_RE.sub("", text)
    cleaned = _INCOMPLETE_THINKING_RE.sub("", cleaned).strip()
    return cleaned


def _extract_message_text(data):
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return _clean_model_text(data["output_text"])

    message_chunks = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                message_chunks.append(content["text"])

    if message_chunks:
        return _clean_model_text("\n".join(message_chunks))
    return ""


def _extract_response_text(data):
    message_text = _extract_message_text(data)
    if message_text:
        return message_text

    mcp_outputs = []
    pending_calls = []

    for item in data.get("output", []):
        item_type = item.get("type")

        if item_type == "mcp_call":
            if item.get("error"):
                mcp_outputs.append(
                    f"**{item.get('name', 'mcp_tool')}** failed: `{item['error']}`"
                )
            elif item.get("output"):
                mcp_outputs.append(
                    f"**{item.get('name', 'mcp_tool')}**:\n```\n{item['output']}\n```"
                )
        elif item_type == "function_call":
            pending_calls.append(item.get("name") or "unknown_tool")

    if mcp_outputs:
        return "Tool results:\n\n" + "\n\n".join(mcp_outputs)

    if pending_calls:
        return (
            "The model requested tools that could not be executed automatically: "
            f"`{', '.join(pending_calls)}`. Try asking in natural language, for "
            "example: *What is the status of my cluster?*"
        )

    if data.get("error"):
        return f"❌ **The server returned an error:** `{data['error']}`"
    if data.get("detail"):
        return f"❌ **API error:** `{data['detail']}`"
    if data.get("status") == "failed":
        return f"❌ **Response failed:** `{data.get('incomplete_details')}`"
    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
        leftover = _clean_model_text(_extract_message_text(data) or "")
        hint = (
            "The model ran out of output tokens (often spent on hidden reasoning). "
            "Click **Apply Changes and Restart Chat**, ask again, or raise "
            "`VLLM_MAX_TOKENS` / `LLAMA_STACK_MCP_MAX_OUTPUT_TOKENS`."
        )
        if leftover:
            return f"{leftover}\n\n⚠️ **Incomplete response** ({reason}). {hint}"
        return f"⚠️ **Incomplete response** ({reason}). {hint}"

    return (
        "⚠️ **No assistant message was returned.**\n"
        f"```json\n{data}\n```"
    )


def _run_response_turn(payload):
    response_data = None
    message_parts = []

    for _ in range(MAX_RESPONSE_CONTINUATIONS + 1):
        response_res = _ls_request(
            "POST",
            "/responses",
            json=payload,
            timeout=TURN_TIMEOUT,
        )
        response_res.raise_for_status()
        response_data = response_res.json()

        message_text = _extract_message_text(response_data)
        if message_text:
            message_parts.append(message_text)

        status = response_data.get("status")
        if status != "incomplete":
            break

        incomplete_details = response_data.get("incomplete_details") or {}
        reason = incomplete_details.get("reason")
        if reason not in ("max_output_tokens", "length"):
            break

        # Continuations need a stored response id; skip if store was disabled.
        if payload.get("store") is False:
            break

        response_id = response_data.get("id")
        if not response_id:
            break

        payload = {
            "model": payload["model"],
            "previous_response_id": response_id,
            "max_output_tokens": payload.get("max_output_tokens", MAX_OUTPUT_TOKENS),
            "stream": False,
        }

    if message_parts:
        return "\n\n".join(message_parts)

    return _extract_response_text(response_data)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    st.caption(f"Backend: `{SCHEME}://{HOST}:{PORT}`")
    st.caption("API: OpenAI-compatible `/v1/responses`")

    models, models_error = get_models()
    if models_error:
        st.warning(f"Could not list models. Using `{DEFAULT_MODEL}`.")
        st.caption(models_error)

    selected_model = st.selectbox(
        "🧠 Model",
        models,
        index=_default_index(models, DEFAULT_MODEL),
    )
    selected_model = _resolve_model_id(selected_model, models)
    temperature = st.slider(
        "🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )

    st.divider()

    st.subheader("📚 Knowledge Bases (RAG)")
    vstores, vstores_error = get_vector_stores()
    enable_rag = st.toggle(
        "Enable Vector Search",
        value=bool(vstores) or bool(DEFAULT_VECTOR_STORE),
    )
    if vstores_error:
        st.warning("Could not list vector stores.")
        st.caption(vstores_error)
    store_labels = [store["label"] for store in vstores]
    selected_vstore = None
    selected_store = None
    if enable_rag and vstores:
        default_store_id = _pick_default_vector_store(vstores)
        default_store_label = next(
            store["label"] for store in vstores if store["id"] == default_store_id
        )
        selected_label = st.selectbox(
            "Select Vector Store",
            store_labels,
            index=store_labels.index(default_store_label),
        )
        selected_store = next(
            store for store in vstores if store["label"] == selected_label
        )
        selected_vstore = selected_store["id"]
        emb = _store_embedding_model(selected_store)
        if emb:
            st.caption(f"Embedding (from store metadata): `{emb}`")
        if _uses_file_search(selected_store):
            st.caption("Retrieval: Llama Stack `file_search`")
        else:
            st.caption(
                "Retrieval: AutoRAG Milvus bridge "
                f"(`{_milvus_collection_name(selected_vstore)}`)"
            )
    elif enable_rag:
        st.warning("No vector stores available. Run AutoRAG first.")

    st.divider()

    st.subheader("🛠️ Tools")
    builtin_tools, builtin_error = get_builtin_tools()
    if builtin_error:
        st.caption(f"Built-in tool discovery: {builtin_error}")

    enable_websearch = st.toggle(
        "Enable Web Search",
        value="builtin::websearch" in builtin_tools,
        disabled="builtin::websearch" not in builtin_tools,
    )

    enable_mcp = st.toggle("Enable OpenShift MCP Server", value=bool(MCP_SERVER_URL))
    if enable_mcp:
        st.caption(f"MCP endpoint: `{MCP_SERVER_URL}`")

    st.divider()
    if st.button(
        "🔄 Apply Changes and Restart Chat", type="primary", use_container_width=True
    ):
        st.session_state.clear()
        st.rerun()

st.title("🦙 Intelligent Assistant (Llama Stack)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rag_chunks"):
            with st.expander("Retrieved context", expanded=False):
                st.caption(msg.get("rag_meta", ""))
                for chunk in msg["rag_chunks"]:
                    st.markdown(
                        f"**score={chunk['score']:.3f}**\n\n{chunk['content']}"
                    )

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Querying the AI and its tools..."):
            try:
                use_file_search = enable_rag and selected_vstore and _uses_file_search(
                    selected_store
                )
                use_milvus_bridge = (
                    enable_rag
                    and selected_vstore
                    and selected_store
                    and not use_file_search
                )

                retrieved = None
                retrieved_context = None
                if use_milvus_bridge:
                    retrieved = _retrieve_rag_context(
                        selected_vstore, prompt, selected_store
                    )
                    retrieved_context = retrieved.get("context_text") or None
                    if not retrieved_context:
                        retrieved_context = (
                            "(No matching chunks were retrieved from the "
                            "AutoRAG vector store.)"
                        )

                tools = _build_response_tools(
                    enable_rag=enable_rag,
                    selected_vstore=selected_vstore,
                    selected_store=selected_store,
                    enable_websearch=enable_websearch,
                    enable_mcp=enable_mcp,
                )
                max_output_tokens = (
                    MCP_MAX_OUTPUT_TOKENS if enable_mcp else MAX_OUTPUT_TOKENS
                )

                response_payload = {
                    "model": selected_model,
                    "input": f"{prompt} /no_think" if enable_mcp else prompt,
                    "instructions": _build_instructions(
                        enable_rag=enable_rag,
                        enable_mcp=enable_mcp,
                        retrieved_context=retrieved_context,
                    ),
                    "temperature": 0.2 if enable_mcp else temperature,
                    "max_output_tokens": max_output_tokens,
                    "stream": False,
                    "store": False,
                }
                if tools:
                    response_payload["tools"] = tools
                if use_file_search:
                    response_payload["include"] = ["file_search_call.results"]
                    if not enable_mcp:
                        response_payload["tool_choice"] = {"type": "file_search"}

                bot_reply = _run_response_turn(response_payload)
                message_placeholder.markdown(bot_reply)

                assistant_msg = {"role": "assistant", "content": bot_reply}
                if retrieved and retrieved.get("chunks"):
                    assistant_msg["rag_chunks"] = retrieved["chunks"]
                    assistant_msg["rag_meta"] = (
                        f"embedding=`{retrieved['embedding_model']}` · "
                        f"collection=`{retrieved['collection']}`"
                    )
                    with st.expander("Retrieved context", expanded=False):
                        st.caption(assistant_msg["rag_meta"])
                        for chunk in retrieved["chunks"]:
                            st.markdown(
                                f"**score={chunk['score']:.3f}**\n\n{chunk['content']}"
                            )
                st.session_state.messages.append(assistant_msg)
            except requests.RequestException as exc:
                st.error("Network error querying Llama Stack:")
                st.write(_request_error_detail(exc))
            except Exception as exc:
                st.error("RAG / retrieval error:")
                st.write(str(exc))
