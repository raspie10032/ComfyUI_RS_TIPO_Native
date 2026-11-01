// Transient status only: never store transfer state in workflow properties.
export function formatDownloadStatus(data) {
    const mib = bytes => (Math.max(0, Number(bytes) || 0) / 1048576).toFixed(1);
    const total = Math.max(0, Number(data.total) || 0);
    const current = Math.min(total, Math.max(0, Number(data.current) || 0));
    const percent = total ? (100 * current / total).toFixed(1) : '0.0';
    const amount = `${percent}% · ${mib(current)} / ${mib(total)} MiB`;
    switch (data.phase) {
        case 'starting': return `Preparing model download · ${amount}`;
        case 'downloading': return `Downloading ${data.filename} · ${amount}`;
        case 'verifying': return `Checking SHA-256: ${data.filename} · ${amount}`;
        case 'ready': return 'Model download verified · Loading TIPO…';
        case 'stopped': return 'Model download stopped · See execution error; run again to resume.';
        default: return '';
    }
}

export function bindDownloadStatus(api, app, panels) {
    api.addEventListener('rs_tipo_download', event => {
        const data = event.detail;
        const node = app.graph?.getNodeById(data?.node);
        const panel = node && panels.get(node);
        if (panel) panel.textContent = formatDownloadStatus(data);
    });
    api.addEventListener('executed', event => {
        const node = app.graph?.getNodeById(event.detail?.node);
        const panel = node && panels.get(node);
        if (panel) panel.textContent = 'TIPO ready · Execution complete.';
    });
}
