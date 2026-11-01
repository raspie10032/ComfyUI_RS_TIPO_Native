// Outputs always retain backend indexes: pipe=0, common=1, character N=N+1.
// Connected inactive sockets stay visible and saved; execution emits empty
// STRINGs for them. Never remove an output before another connected output.
export function updateSockets(node, count, addInput = name => node.addInput(name, "STRING")) {
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        const input = node.inputs[i], match = /^character(\d+)$/.exec(input.name);
        if (!match) continue;
        const inactive = Number(match[1]) > count;
        if (inactive && input.link == null) node.removeInput(i);
        else input.label = input.name + (inactive ? " (inactive)" : "");
    }
    for (let i = 1; i <= count; i++) {
        if (!node.inputs?.some(input => input.name === `character${i}`)) addInput(`character${i}`);
    }
    let last = count + 1;
    for (let i = 2; i < (node.outputs?.length ?? 0); i++) {
        if (node.outputs[i].links?.length) last = Math.max(last, i);
    }
    while ((node.outputs?.length ?? 0) > last + 1) node.removeOutput(node.outputs.length - 1);
    while ((node.outputs?.length ?? 0) <= last) {
        const i = node.outputs.length - 1;
        node.addOutput(`character${i}`, "STRING");
    }
    for (let i = 2; i < node.outputs.length; i++) {
        node.outputs[i].label = `character${i-1}` + (i-1 > count ? " (inactive)" : "");
    }
}
