const LLM_ARGS = {
    gemini: [
        {name: "gemini_api_key", label: "Gemini API Key", type: "password", required: true}
    ],
    azure_openai: [
        {name: "azure_openai_api_key", label: "Azure OpenAI API Key", type: "password", required: true},
        {name: "azure_openai_api_version", label: "API Version", type: "text", required: true},
        {name: "azure_openai_api_endpoint", label: "API Endpoint", type: "text", required: true}
    ],
    openai: [
        {name: "openai_api_key", label: "OpenAI API Key", type: "password", required: true}
    ],
    ollama: [
        {name: "ollama_host", label: "Ollama Host", type: "text", required: true}
    ],
    groq: [
        {name: "groq_api_key", label: "Groq API Key", type: "password", required: true}
    ]
};

function renderModelDropdown(selectId, argsDivId, modelDivId, currentSettings) {
    const llm = document.getElementById(selectId).value;
    const argsDiv = document.getElementById(argsDivId);
    const modelDiv = document.getElementById(modelDivId);
    let apiKey = "";
    let endpoint = "";

    // Get API key and endpoint from rendered inputs
    if (LLM_ARGS[llm]) {
        LLM_ARGS[llm].forEach(arg => {
            const input = document.getElementById(`${selectId}-${arg.name}`);
            if (input) {
                if (arg.name.includes("api_key")) apiKey = input.value;
                if (arg.name.includes("endpoint")) endpoint = input.value;
            }
        });
    }

    // Show loading
    modelDiv.innerHTML = `<span>Loading models...</span>`;

    fetch("/api/get_models", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({llm_type: llm, api_key: apiKey, endpoint: endpoint})
    })
    .then(res => res.json().then(data => ({status: res.status, body: data})))
    .then(({status, body}) => {
        if (status !== 200) {
            modelDiv.innerHTML = `<span style="color:red;">${body.error || "Failed to fetch models"}</span>`;
            return;
        }
        const models = body.models || [];
        if (models.length === 0) {
            modelDiv.innerHTML = `<span>No models found for this API key.</span>`;
            return;
        }
        let selected = currentSettings && currentSettings["model-name"] ? currentSettings["model-name"] : "";
        modelDiv.innerHTML = `
            <label for="${selectId}_model-name">Model Name</label>
            <select id="${selectId}_model-name" name="${selectId}-model-name" required>
                ${models.map(m => `<option value="${m}" ${m === selected ? "selected" : ""}>${m}</option>`).join("")}
            </select>
        `;
    })
    .catch(() => {
        modelDiv.innerHTML = `<span style="color:red;">Failed to fetch models.</span>`;
    });
}

function renderLlmArgs(selectId, argsDivId, currentSettings) {
    const select = document.getElementById(selectId);
    const argsDiv = document.getElementById(argsDivId);
    const llm = select.value;
    argsDiv.innerHTML = "";
    if (LLM_ARGS[llm]) {
        LLM_ARGS[llm].forEach(arg => {
            const value = currentSettings && currentSettings[arg.name] ? currentSettings[arg.name] : "";
            argsDiv.innerHTML += `
                <label for="${selectId}-${arg.name}">${arg.label}${arg.required ? ' *' : ''}</label>
                <input type="${arg.type}" id="${selectId}-${arg.name}" name="${selectId}_${arg.name}" value="${value}" ${arg.required ? 'required' : ''} autocomplete="off">
            `;
        });
    }
}

document.addEventListener("DOMContentLoaded", function() {
    // These objects should be rendered from Flask context if you want to prefill values
    const agentLlmSettings = window.agentLlmSettings || {};
    const plannerLlmSettings = window.plannerLlmSettings || {};

    renderLlmArgs("agent-llm", "agent-llm-args", agentLlmSettings);
    renderLlmArgs("planner-llm", "planner-llm-args", plannerLlmSettings);

    document.getElementById("agent-llm").addEventListener("change", function() {
        renderLlmArgs("agent-llm", "agent-llm-args");
    });
    document.getElementById("planner-llm").addEventListener("change", function() {
        renderLlmArgs("planner-llm", "planner-llm-args");
    });
});