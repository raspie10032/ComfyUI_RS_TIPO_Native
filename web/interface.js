import { app } from "../../scripts/app.js";
import { updateSockets } from "./sockets.js";

const panels = new WeakMap();
const advancedNames = new Set(["max_new_tokens", "seed", "control_after_generate", "temperature", "common_weight", "character_weight"]);
const value = (node, name, fallback) => node.widgets?.find(w => w.name === name)?.value ?? fallback;

function refresh(node, sockets = true) {
    const count = Number(value(node, "character_count", 2));
    if (sockets) updateSockets(node, count);
    for (const widget of node.widgets ?? []) {
        if (!advancedNames.has(widget.name)) continue;
        const hide = !value(node, "advanced", false);
        if (hide && !widget.tipoSaved && !widget.type?.startsWith("converted-widget")) {
            widget.tipoSaved = { type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden };
            widget.type = "rs-tipo-hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
        } else if (!hide && widget.tipoSaved) {
            Object.assign(widget, widget.tipoSaved); delete widget.tipoSaved;
        }
    }
    const size = node.computeSize();
    node.setSize([Math.max(390, node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

function addReport(node) {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:8px;box-sizing:border-box;height:100%;background:#18212b;color:#e0e8f3;font:12px sans-serif;";
    const help = document.createElement("div");
    help.textContent = "Shared scene + participant roles · Source protection limits added attributes and known role conflicts. Review results before generation.";
    const status = document.createElement("div");
    status.textContent = "Connect common / character STRINGs, then run. Results appear here.";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "TIPO result slot");
    const text = document.createElement("textarea");
    text.readOnly = true; text.setAttribute("aria-label", "TIPO expansion result");
    text.style.cssText = "flex:1;min-height:100px;resize:none;background:#0f1720;color:#e0e8f3;border:1px solid #40536a;padding:8px;white-space:pre-wrap;";
    let rows = [];
    const render = () => {
        const row = rows[Number(select.value)];
        const rejected = row?.rejected_tags?.map(item => `${item.tag} — ${item.reason}`).join("\n");
        text.value = row ? `Original\n${row.original}\n\nAdded\n${row.added_tags || '(none)'}\n\nExcluded\n${rejected || '(none)'}\n\nExpanded\n${row.output}` : "";
    };
    select.onchange = render;
    root.append(help, status, select, text);
    node.addDOMWidget("tipo_results", "rs-tipo-results", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 245, getMaxHeight: () => 245,
    });
    panels.set(node, report => {
        rows = report.slots ?? [];
        select.replaceChildren(...rows.map((row, i) => {
            const option = document.createElement("option");
            option.value = String(i); option.textContent = row.slot === "common" ? "Common prompt" : `Character ${row.slot}`;
            return option;
        }));
        help.textContent = report.interaction_prompt ? "Shared interaction: " + report.interaction_prompt : "Each character reads the same common scene. Write each participant’s action in its own input; add interaction_prompt for a shared description.";
        status.textContent = `${report.character_count} characters · ${report.seconds}s · Source protection ${report.protect_source === false ? 'OFF' : 'ON'} · ${report.building_pipe_available ? 'RS pipe + STRINGs' : 'STRINGs (RS pack not installed)'}`;
        render();
    });
}

app.registerExtension({
    name: "RS.TIPO.PromptInterface",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RSTIPOPrompt") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            addReport(this);
            for (const widget of this.widgets ?? []) {
                if (!["character_count", "advanced"].includes(widget.name)) continue;
                const callback = widget.callback;
                widget.callback = (...args) => { callback?.apply(widget, args); refresh(this); };
            }
            refresh(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            refresh(this, false);
            queueMicrotask(() => refresh(this));
            return result;
        };
        const graphConfigured = nodeType.prototype.onGraphConfigured;
        nodeType.prototype.onGraphConfigured = function () {
            const result = graphConfigured?.apply(this, arguments); refresh(this); return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            try { panels.get(this)?.(JSON.parse(message.tipo_report?.[0] ?? "{}")); }
            catch { /* Missing/corrupt cached UI data must not stop graph execution. */ }
            return result;
        };
    },
});
