// Model/user text is data and never becomes HTML.
export function resultText(report, slot = 'all', view = 'output') {
    if (!report) return '';
    if (view === 'details') {
        return JSON.stringify({
            model: report.model_family, base: report.base, adapter: report.adapter,
            adapter_details: report.adapter_details, lora_targets: report.lora_targets,
            model_calls: report.model_calls, seconds: report.seconds,
            limitations: report.limitations, calls: report.calls,
            raw_output: report.raw_output,
        }, null, 2);
    }
    const rows = (report.slots ?? []).filter(row => slot === 'all' || String(row.slot) === slot);
    return rows.map(row => {
        const label = row.slot === 'common' ? 'Common prompt' : `Character ${row.slot}`;
        return `${label}\n${view === 'source' ? row.original : row.output}`;
    }).join('\n\n');
}

export function modelSummary(report) {
    if (!report) return '';
    const detail = report.adapter_details;
    const adapter = report.adapter ? ` · ${report.adapter}` : '';
    const rank = detail?.rank ? ` · rank ${detail.rank}` : '';
    const targets = report.lora_targets ? ` · ${report.lora_targets} targets` : '';
    return `${report.model_family ?? 'Gemma TIPO'}${adapter}${rank}${targets}`;
}
