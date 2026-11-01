// UI-only status simulation; no inference and no model download.
const fs = require('fs'), assert = require('assert/strict');
const {chromium} = require('playwright');
(async () => {
    const browser = await chromium.launch({headless:true,
        ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {}),
        args:['--no-sandbox']});
    try {
        const page = await browser.newPage({viewport:{width:1280,height:1050}});
        const errors = [];
        page.on('pageerror', e => errors.push(e.message));
        await page.goto(`http://127.0.0.1:${process.argv[2] || 8188}/`);
        await page.waitForFunction(() => globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);
        await page.waitForTimeout(4500); // Let the frontend finish initial workspace restore.
        await page.evaluate(async graph => {
            const {app} = await import('/scripts/app.js');
            await app.loadGraphData(graph);
        }, JSON.parse(fs.readFileSync('examples/shared-interaction.json','utf8')));
        await page.waitForFunction(async () => {
            const {app} = await import('/scripts/app.js');
            return app.graph._nodes.some(n => n.type === 'RSTIPOPrompt');
        });
        const results = await page.evaluate(async () => {
            const {app} = await import('/scripts/app.js');
            const {api} = await import('/scripts/api.js');
            const node = app.graph._nodes.find(n => n.type === 'RSTIPOPrompt');
            const panel = node.widgets.find(w => w.name === 'tipo_download').element;
            const statuses = [];
            for (const phase of ['starting','downloading','verifying','ready','stopped']) {
                api.dispatchEvent(new CustomEvent('rs_tipo_download',{detail:{
                    node:String(node.id),phase,filename:'model.safetensors',current:1048576,total:2097152}}));
                statuses.push(panel.textContent);
            }
            const saved = app.graph.serialize();
            structuredClone(saved);
            const serialized = JSON.stringify(saved);
            await app.loadGraphData(saved);
            const restored = app.graph._nodes.find(n => n.type === 'RSTIPOPrompt');
            return {statuses, serializedStatus:serialized.includes('Model download stopped'),
                restored:restored.widgets.find(w => w.name === 'tipo_download').element.textContent};
        });
        assert.match(results.statuses[1], /50.0%.*1.0 \/ 2.0 MiB/);
        assert.match(results.statuses[2], /SHA-256/);
        assert.match(results.statuses[3], /verified/);
        assert.match(results.statuses[4], /resume/);
        assert.equal(results.serializedStatus,false);
        assert.match(results.restored,/auto-download/);
        assert.deepEqual(errors,[]);
        fs.mkdirSync('validation/compatibility',{recursive:true});
        fs.writeFileSync('validation/compatibility/download-panel.json',JSON.stringify({mode:'synthetic UI status events',...results,errors},null,2));
        console.log('PASS',JSON.stringify(results));
    } finally {await browser.close();}
})().catch(e => {console.error(e);process.exit(1);});
