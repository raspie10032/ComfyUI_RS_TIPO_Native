// Model/user text is data and never becomes HTML.
export function resultText(report, slot = 'all', view = 'output') {
    if (!report) return '';
    if (view === 'details') {
        return JSON.stringify({
            model: report.model_family, base: report.base, adapter: report.adapter,
            adapter_details: report.adapter_details, lora_targets: report.lora_targets,
            lora_strength: report.lora_strength, inference_mode: report.inference_mode,
            effective_patch_strengths: report.effective_patch_strengths,
            model_calls: report.model_calls, seconds: report.seconds,
            limitations: report.limitations, calls: report.calls, warnings: report.warnings,
            raw_output: report.raw_output,
            slots: report.slots,
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
    const strength = report.lora_strength != null ? ` · strength ${report.lora_strength}` : '';
    const rank = detail?.rank ? ` · rank ${detail.rank}` : '';
    const targets = report.lora_targets ? ` · ${report.lora_targets} targets` : '';
    const counts = (report.slots ?? []).filter(row => row.expansion_status)
        .map(row => `${row.slot === 'common' ? 'common' : row.slot}: ${
            row.expansion_status === 'expanded' ? '+' + row.added_tag_count + ' tags' : row.expansion_status}`);
    const expansion = counts.length ? ` · ${counts.join(' / ')}` : '';
    const warnings = report.warnings?.length ? ` · ⚠ ${report.warnings.length} recovery warning(s); see model details` : '';
    return `${report.model_family ?? 'Gemma TIPO'}${adapter}${strength}${rank}${targets}${expansion}${warnings}`;
}
