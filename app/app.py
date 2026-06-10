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
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE_ID", "pizza-bank-production")
MCP_SERVER_LABEL = os.getenv("MCP_SERVER_LABEL", "openshift-mcp-server")
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://openshift-mcp-deployment.llamastack.svc.cluster.local:8080/mcp",
)
REQUEST_TIMEOUT = int(os.getenv("LLAMA_STACK_REQUEST_TIMEOUT", "30"))
TURN_TIMEOUT = int(os.getenv("LLAMA_STACK_TURN_TIMEOUT", "300"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_STACK_MAX_OUTPUT_TOKENS", "2048"))
MAX_RESPONSE_CONTINUATIONS = int(os.getenv("LLAMA_STACK_MAX_RESPONSE_CONTINUATIONS", "2"))
RAG_MAX_RESULTS = int(os.getenv("LLAMA_STACK_RAG_MAX_RESULTS", "5"))
RAG_INSTRUCTIONS = (
    "Answer Pizza Bank product and policy questions using only information found in "
    "file_search results. Cite the retrieved documents. If file_search returns no "
    "relevant chunks, say the knowledge base has no matching information."
)
MCP_INSTRUCTIONS = (
    "For OpenShift/Kubernetes cluster questions, use OpenShift MCP tools sparingly: "
    "call nodes_top once and optionally events_list. Do not call pods_list unless the "
    "user asks about specific pods. There is no nodes_list tool; use nodes_top for "
    "node metrics."
)
COMBINED_TOOL_INSTRUCTIONS = (
    "Choose the right tool for each question. Use file_search only for Pizza Bank "
    "products, accounts, cards, fees, and policies. Use OpenShift MCP tools for "
    "Kubernetes or cluster infrastructure questions (nodes, pods, events, cluster "
    "status). Never answer infrastructure questions from file_search or claim the "
    "knowledge base lacks node or cluster data—call MCP tools instead."
)
BASE_INSTRUCTIONS = (
    "You are a corporate assistant for Pizza Bank. Reply directly to the user in "
    "plain language. Never include internal reasoning, thinking tags, or  blocks. "
    "Keep answers concise but complete."
)


def _build_instructions(*, enable_rag, enable_mcp):
    parts = [BASE_INSTRUCTIONS]
    if enable_rag and enable_mcp:
        parts.append(COMBINED_TOOL_INSTRUCTIONS)
        parts.append(MCP_INSTRUCTIONS)
    elif enable_rag:
        parts.append(RAG_INSTRUCTIONS)
    elif enable_mcp:
        parts.append(MCP_INSTRUCTIONS)
    return " ".join(parts)
_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
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
                "The model context window was exceeded. This usually happens when MCP "
                "tools return very large outputs (for example pods_list across all "
                "namespaces). Click **Apply Changes and Restart Chat**, then ask a "
                "more specific question such as *How are my nodes doing?*"
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
    fallback = [
        {
            "id": DEFAULT_VECTOR_STORE,
            "name": DEFAULT_VECTOR_STORE,
            "label": DEFAULT_VECTOR_STORE,
            "completed_files": 0,
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
            completed_files = (item.get("file_counts") or {}).get("completed", 0)
            label = f"{name} ({store_id})" if name else store_id
            stores.append(
                {
                    "id": store_id,
                    "name": name or store_id,
                    "label": label,
                    "completed_files": completed_files,
                }
            )
        if stores:
            return stores, None
        return fallback, "Llama Stack returned no vector stores."
    except Exception as exc:
        return fallback, _request_error_detail(exc)


def _pick_default_vector_store(stores):
    for store in stores:
        if store["id"] == DEFAULT_VECTOR_STORE or store["name"] == DEFAULT_VECTOR_STORE:
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


def _build_response_tools(*, enable_rag, selected_vstore, enable_websearch, enable_mcp):
    tools = []

    if enable_rag and selected_vstore:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [selected_vstore],
                "max_num_results": RAG_MAX_RESULTS,
            }
        )

    if enable_websearch:
        tools.append({"type": "web_search"})

    if enable_mcp and MCP_SERVER_URL:
        tools.append(
            {
                "type": "mcp",
                "server_label": MCP_SERVER_LABEL,
                "server_description": "OpenShift MCP server deployed via the MCP catalog",
                "server_url": MCP_SERVER_URL,
                "require_approval": "never",
            }
        )

    return tools


def _clean_model_text(text):
    cleaned = _THINKING_BLOCK_RE.sub("", text).strip()
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
        return (
            "⚠️ **The response was cut off before completion.** "
            "Try a more specific question or disable Vector Search for cluster queries."
        )

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
        if incomplete_details.get("reason") != "max_output_tokens":
            break

        response_id = response_data.get("id")
        if not response_id:
            break

        payload = {
            "model": payload["model"],
            "previous_response_id": response_id,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
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
    enable_rag = st.toggle(
        "Enable Vector Search",
        value=bool(DEFAULT_VECTOR_STORE),
    )
    vstores, vstores_error = get_vector_stores()
    if vstores_error:
        st.warning(
            f"Could not list vector stores. Using `{DEFAULT_VECTOR_STORE}`."
        )
        st.caption(vstores_error)
    store_labels = [store["label"] for store in vstores]
    default_store_id = _pick_default_vector_store(vstores)
    default_store_label = next(
        store["label"] for store in vstores if store["id"] == default_store_id
    )
    selected_vstore = None
    if enable_rag:
        selected_label = st.selectbox(
            "Select Vector Store",
            store_labels,
            index=store_labels.index(default_store_label),
        )
        selected_store = next(
            store for store in vstores if store["label"] == selected_label
        )
        selected_vstore = selected_store["id"]
        st.caption(
            f"Milvus ID: `{selected_vstore}` · "
            f"indexed files: {selected_store['completed_files']}"
        )

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

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Querying the AI and its tools..."):
            try:
                tools = _build_response_tools(
                    enable_rag=enable_rag,
                    selected_vstore=selected_vstore,
                    enable_websearch=enable_websearch,
                    enable_mcp=enable_mcp,
                )

                response_payload = {
                    "model": selected_model,
                    "input": prompt,
                    "instructions": _build_instructions(
                        enable_rag=enable_rag,
                        enable_mcp=enable_mcp,
                    ),
                    "temperature": temperature,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "stream": False,
                }
                if tools:
                    response_payload["tools"] = tools
                if enable_rag and selected_vstore:
                    response_payload["include"] = ["file_search_call.results"]
                    if not enable_mcp:
                        response_payload["tool_choice"] = {"type": "file_search"}

                bot_reply = _run_response_turn(response_payload)
                message_placeholder.markdown(bot_reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply}
                )
            except requests.RequestException as exc:
                st.error("Network error querying Llama Stack:")
                st.write(_request_error_detail(exc))
