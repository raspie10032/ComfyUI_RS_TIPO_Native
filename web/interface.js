import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { updateSockets } from "./sockets.js";
import { bindDownloadStatus } from "./download-status.js";
import { resultText, modelSummary } from "./report.js";

const panels = new WeakMap();
const downloadPanels = new WeakMap();
bindDownloadStatus(api, app, downloadPanels);
const configuredInputs = new WeakSet();
const scheduled = new WeakSet();
const restoring = new WeakSet();
const advancedNames = new Set(["max_new_tokens", "max_added_tags", "seed", "control_after_generate", "temperature", "common_weight", "character_weight", "quality_era", "language_model", "translation_max_tokens", "generation_max_tokens"]);
const retiredNames = new Set(["common_weight", "character_weight", "max_new_tokens", "input_language", "translation_max_tokens", "protect_source"]);
const value = (node, name, fallback) => node.widgets?.find(w => w.name === name)?.value ?? fallback;
const promptSocket = name => /^(?:common_prompt|character\d+)$/.test(name);

function layoutPromptSockets(node) {
    // Keep the backing widgets required by Primitive STRING, but give every
    // socket a real row. A zero-height hidden widget puts every connection at
    // y=0 in current LiteGraph, including after the next arrange() pass.
    const rows = [];
    for (const input of node.inputs ?? []) {
        if (!promptSocket(input.name)) continue;
        const widget = node.widgets?.find(w => w.name === input.name);
        if (!widget) continue;
        widget.type = "rs-tipo-string-socket";
        widget.hidden = false;
        widget.computeSize = () => [0, 20];
        widget.draw = () => {};
        rows.push(widget);
    }
    rows.sort((a, b) => {
        const rank = name => name === "common_prompt" ? 0 : Number(name.slice(9));
        return rank(a.name) - rank(b.name);
    });
    for (const widget of node.widgets ?? []) {
        if (!promptSocket(widget.name) || rows.includes(widget)) continue;
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
    }
    // Splice in place: frontend widget stores can retain the array identity.
    if (node.widgets) node.widgets.splice(0, node.widgets.length,
        ...rows, ...node.widgets.filter(w => !rows.includes(w)));
}

function scheduleRefresh(node) {
    if (scheduled.has(node)) return;
    scheduled.add(node);
    const run = () => {
        if (app.configuringGraph) { setTimeout(run, 10); return; }
        scheduled.delete(node);
        refresh(node);
        if (restoring.delete(node)) {
            // Legacy PrimitiveNode can create its STRING widget but skip
            // restoring widgets_values. Repair only sources connected to us,
            // once during graph load, never after an interactive edit.
            for (const input of node.inputs ?? []) {
                if (input.type !== "STRING" || input.link == null) continue;
                const link = node.graph?.links?.[input.link];
                const source = link && node.graph.getNodeById(link.origin_id);
                if (source?.type === "PrimitiveNode" && typeof source.widgets_values?.[0] === "string" && source.widgets?.[0]) {
                    source.widgets[0].value = source.widgets_values[0];
                    source.applyToGraph?.();
                }
            }
        }
    };
    queueMicrotask(run);
}

function refresh(node, sockets = true) {
    const count = Number(value(node, "character_count", 2));
    if (sockets) {
        // Retire the prototype's separate caption input only after graph restore.
        // Its source node and saved text remain available for the user to reuse.
        const legacy = node.inputs?.findIndex(s => s.name === "interaction_prompt") ?? -1;
        if (legacy >= 0) node.removeInput(legacy);
        const legacyWidget = node.widgets?.findIndex(w => w.name === "interaction_prompt") ?? -1;
        if (legacyWidget >= 0) {
            const prior = node.widgets[legacyWidget].value;
            if (typeof prior === "string" && prior) {
                node.properties ??= {};
                node.properties.rs_tipo_legacy_interaction = prior;
            }
            node.widgets.splice(legacyWidget, 1);
        }
        updateSockets(node, count, name => node.addInput(name, "STRING", { widget: { name } }));
        // Primitive STRING nodes need the widget configuration, including after
        // a socket is recreated or an older workflow has lost its widget field.
        const setConfig = globalThis.comfyAPI?.widgetInputs?.setWidgetConfig;
        for (const input of node.inputs ?? []) {
            if (!promptSocket(input.name)) continue;
            if (!node.widgets?.some(w => w.name === input.name)) {
                // Frontend 1.28 omits the forceInput backing widget, while
                // PrimitiveNode still requires one to transfer its STRING.
                const saved = node.properties?.rs_tipo_widget_values?.[input.name];
                const w = node.addWidget("converted-widget", input.name,
                    typeof saved === "string" ? saved : "", () => {}, {});
                w.hidden = true;
                w.computeSize = () => [0, -4];
            }
            if (!setConfig || configuredInputs.has(input)) continue;
            input.widget ??= { name: input.name };
            setConfig(input, ["STRING", { multiline: true, forceInput: true }]);
            configuredInputs.add(input);
        }
        layoutPromptSockets(node);
    }
    for (const widget of node.widgets ?? []) {
        if (!advancedNames.has(widget.name) && !retiredNames.has(widget.name) && widget.name !== 'tipo_lora') continue;
        const linked = node.inputs?.find(input => input.name === widget.name)?.link != null;
        const standalone = value(node, 'model_family', 'gemma_tipo') === 'tipo_2_1';
        const hide = (retiredNames.has(widget.name) && !(standalone && widget.name === 'protect_source'))
            || (standalone && ['language_model', 'tipo_lora'].includes(widget.name))
            || (advancedNames.has(widget.name) && !value(node, "advanced", false) && !linked);
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
    root.style.width = "100%";
    const help = document.createElement("div");
    help.textContent = "Choose Gemma TIPO for Korean/English, or TIPO 2.1 for English. Missing files download on first use. Review meaning and roles.";
    const status = document.createElement("div");
    status.textContent = "Connect common / character STRINGs, then run. Results appear here.";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "TIPO result slot");
    const view = document.createElement("select");
    view.setAttribute("aria-label", "TIPO result view");
    for (const [value, label] of [['output', 'Output'], ['source', 'Source'], ['details', 'Model details / raw output']]) {
        const option = document.createElement("option");
        option.value = value; option.textContent = label; view.append(option);
    }
    const copy = document.createElement("button");
    copy.type = "button"; copy.textContent = "Copy";
    copy.setAttribute("aria-label", "Copy TIPO displayed result");
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
    select.style.flex = "1"; view.style.flex = "1";
    controls.append(select, view, copy);
    const text = document.createElement("textarea");
    text.readOnly = true; text.setAttribute("aria-label", "TIPO expansion result");
    text.style.cssText = "flex:1;min-height:100px;resize:none;background:#0f1720;color:#e0e8f3;border:1px solid #40536a;padding:8px;white-space:pre-wrap;";
    let modelReport = null;
    const render = () => {
        text.value = resultText(modelReport, select.value || 'all', view.value || 'output');
        select.disabled = view.value === 'details';
        copy.disabled = !text.value;
    };
    select.onchange = view.onchange = render;
    copy.disabled = true;
    copy.onclick = async () => {
        try {
            await navigator.clipboard.writeText(text.value);
            copy.textContent = 'Copied';
            setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
        } catch {
            text.focus(); text.select();
            copy.textContent = 'Press Ctrl+C';
        }
    };
    root.append(help, status, controls, text);
    // Legacy ComfyUI resets display on the outer DOM widget each frame.
    // Preserve the report's flex layout inside an independent inner element.
    const container = document.createElement("div");
    container.append(root);
    node.addDOMWidget("tipo_results", "rs-tipo-results", container, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 245, getMaxHeight: () => 245,
    });
    panels.set(node, report => {
        modelReport = report;
        const previous = select.value;
        const all = document.createElement("option");
        all.value = 'all'; all.textContent = 'All prompts';
        select.replaceChildren(all, ...(report.slots ?? []).map(row => {
            const option = document.createElement("option");
            option.value = String(row.slot); option.textContent = row.slot === "common" ? "Common prompt" : `Character ${row.slot}`;
            return option;
        }));
        select.value = [...select.options].some(option => option.value === previous) ? previous : 'all';
        help.textContent = report.model_family === 'TIPO 2.1'
            ? 'Standalone TIPO 2.1 · English only. Tags share action context; captions use individual inputs and may invent details. Prefer Gemma TIPO for character prose.'
            : 'One Gemma base + selected TIPO LoRA. Reads all participants jointly; no second model. Korean meaning and tag accuracy remain experimental.';
        status.textContent = `${modelSummary(report)}\n${report.character_count} characters · ${report.seconds}s · ${report.output_format} · ${report.building_pipe_available ? 'RS pipe + STRINGs' : 'STRINGs'}`;
        status.style.whiteSpace = 'pre-wrap';
        render();
    });
}

app.registerExtension({
    name: "RS.TIPO.PromptInterface",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData.name.startsWith("RSTIPO")) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const status = document.createElement("div");
            status.style.cssText = "padding:6px;box-sizing:border-box;font:12px sans-serif;white-space:normal;overflow-wrap:anywhere;color:#b9dcec;";
            status.textContent = "Selected model downloads on first use. Gemma Q4 + adapter ~3.27 GiB; TIPO 2.1 ~1.85 GiB. Cached files are reused.";
            this.addDOMWidget("tipo_download", "rs-tipo-download", status, {
                serialize: false, hideOnZoom: false, getMinHeight: () => 42, getMaxHeight: () => 42,
            });
            downloadPanels.set(this, status);
            return result;
        };
        if (nodeData.name !== "RSTIPOPrompt") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            addReport(this);
            for (const widget of this.widgets ?? []) {
                if (!["character_count", "advanced", "model_family"].includes(widget.name)) continue;
                const callback = widget.callback;
                widget.callback = (...args) => { callback?.apply(widget, args); refresh(this); };
            }
            // Core widgetInputs may convert forceInput widgets after this
            // callback (notably frontend 1.11). Wait before managing sockets.
            refresh(this, false);
            scheduleRefresh(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = configured?.apply(this, arguments);
            restoring.add(this);
            // Older frontends keep forceInput STRINGs in widgets_values;
            // newer ones omit them. Restore by name across both layouts.
            const saved = info?.properties?.rs_tipo_widget_values;
            if (saved && typeof saved === "object") {
                for (const widget of this.widgets ?? []) {
                    if (widget.name !== "tipo_results" && Object.hasOwn(saved, widget.name) &&
                        ["string", "number", "boolean"].includes(typeof saved[widget.name])) widget.value = saved[widget.name];
                }
            }
            refresh(this, false);
            scheduleRefresh(this);
            return result;
        };
        const connectionsChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = connectionsChanged?.apply(this, arguments);
            scheduleRefresh(this);
            return result;
        };
        const drawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = drawForeground?.apply(this, arguments);
            if (this.flags?.collapsed) return result;
            // Modern frontends suppress labels for widget-linked sockets. Our
            // rows are connection-only, so draw their names at the actual port.
            ctx.save();
            ctx.font = "12px Arial";
            ctx.fillStyle = "#c4cbd4";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            for (const [index, input] of (this.inputs ?? []).entries()) {
                if (!promptSocket(input.name)) continue;
                if (!("isWidgetInputSlot" in input)) continue; // Older core draws these labels itself.
                const pos = this.getConnectionPos(true, index);
                ctx.fillText(input.label || input.name, pos[0] - this.pos[0] + 12, pos[1] - this.pos[1]);
            }
            ctx.restore();
            return result;
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = serialize?.apply(this, arguments);
            const named = {};
            for (const w of this.widgets ?? []) {
                if (["string", "number", "boolean"].includes(typeof w.value) && w.name !== "tipo_results") named[w.name] = w.value;
            }
            for (const input of this.inputs ?? []) {
                if (input.type !== "STRING" || input.link == null) continue;
                const link = this.graph?.links?.[input.link];
                const source = link && this.graph.getNodeById(link.origin_id);
                if (source?.type === "PrimitiveNode" && typeof source.widgets?.[0]?.value === "string") named[input.name] = source.widgets[0].value;
            }
            info.properties ??= {};
            info.properties.rs_tipo_widget_values = named;
            return result;
        };
        const graphConfigured = nodeType.prototype.onGraphConfigured;
        nodeType.prototype.onGraphConfigured = function () {
            const result = graphConfigured?.apply(this, arguments);
            // Let Primitive.onAfterGraphConfigured restore saved text before
            // setWidgetConfig can recreate its widget.
            scheduleRefresh(this);
            return result;
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
