// Isolated local test browser. Never reuses a user's browser profile.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  assert.equal(process.argv[2], '8196', 'Use the isolated test server');
  const base = 'http://127.0.0.1:8196';
  const out = 'validation/trained-node-20261103';
  const adapter = 'local:scale-20pct-001';
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    args: ['--no-sandbox'] });
  try {
    const context = await browser.newContext({ viewport: { width: 1560, height: 1180 },
      permissions: ['clipboard-read', 'clipboard-write'] });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(base);
    await page.waitForFunction(() => globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);
    // Initial frontend workspace restore can replace a graph loaded too early.
    await page.waitForTimeout(4000);
    await page.evaluate(async graph => {
      const { app } = await import('/scripts/app.js');
      window.tipo = app;
      await app.loadGraphData(graph);
    }, JSON.parse(fs.readFileSync('examples/gemma-tipo-korean.json', 'utf8')));
    await page.waitForTimeout(1000);
    await page.evaluate(settings => {
      const n = window.tipo.graph._nodes.find(n => n.type === 'RSTIPOPrompt');
      for (const [name, val] of Object.entries(settings)) {
        const w = n.widgets.find(w => w.name === name);
        if (!w) throw Error('Missing widget ' + name);
        w.value = val; w.callback?.(val);
      }
      n.title = 'RS TIPO Native — trained Gemma';
    }, { model_family: 'gemma_tipo', tipo_lora: adapter, language_model: 'heretic_ara_q4',
      output_format: 'natural_language', expand_common: true, generation_max_tokens: 1024 });
    await page.waitForTimeout(500);
    const saved = await page.evaluate(async () => {
      const graph = window.tipo.graph.serialize();
      structuredClone(graph);
      await window.tipo.loadGraphData(graph);
      return graph;
    });
    await page.waitForTimeout(1000);
    const state = await page.evaluate(async () => {
      const n = window.tipo.graph._nodes.find(n => n.type === 'RSTIPOPrompt');
      return { inputs: (await window.tipo.graphToPrompt()).output[n.id].inputs,
        ports: n.inputs.map((p, i) => ({name: p.name, y: n.getConnectionPos(true, i)[1]}))
          .filter(p => /^(common_prompt|character\d+|building_pipe)$/.test(p.name)) };
    });
    assert.equal(state.inputs.tipo_lora, adapter);
    assert.equal(state.inputs.character_count, 2);
    assert.equal(state.inputs.common_prompt, '성인 여성 두 명이 오후 카페에서 컵을 주고받는다.');
    assert.equal(new Set(state.ports.map(p => p.y)).size, state.ports.length);
    const response = page.waitForResponse(r => /\/(api\/)?prompt$/.test(new URL(r.url()).pathname)
      && r.request().method() === 'POST');
    await page.evaluate(() => window.tipo.queuePrompt(0, 1));
    const posted = await response;
    assert(posted.ok(), await posted.text());
    const handle = await posted.json();
    const box = page.getByLabel('TIPO expansion result');
    await page.waitForFunction(() => document.querySelector('[aria-label="TIPO expansion result"]')
      ?.value.includes('Common prompt\n'), null, { timeout: 180000 });
    const history = await (await page.request.get(base + '/history/' + handle.prompt_id)).json();
    const record = history[handle.prompt_id];
    assert.equal(record.status.status_str, 'success');
    const report = JSON.parse(Object.values(record.outputs).find(o => o.tipo_report).tipo_report[0]);
    assert.equal(report.adapter, adapter);
    assert.equal(report.adapter_details.rank, 8);
    assert.equal(report.adapter_details.weights_sha256.length, 64);
    assert.equal(report.lora_targets, 190);
    assert.equal(report.slots.length, 3);
    const output = await box.inputValue();
    assert(output.includes(report.slots[1].output));
    await page.getByLabel('TIPO result slot').selectOption('2');
    assert.equal(await box.inputValue(), 'Character 2\n' + report.slots[2].output);
    await page.getByLabel('TIPO result view').selectOption('source');
    assert.equal(await box.inputValue(), 'Character 2\n' + report.slots[2].original);
    await page.getByLabel('TIPO result view').selectOption('details');
    assert((await box.inputValue()).includes(report.adapter_details.weights_sha256));
    await page.getByLabel('Copy TIPO displayed result').click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), await box.inputValue());
    await page.getByLabel('TIPO result view').selectOption('output');
    await page.getByLabel('TIPO result slot').selectOption('all');
    assert.equal(errors.length, 0, errors.join('\n'));
    fs.mkdirSync(out, {recursive: true});
    fs.writeFileSync(out + '/browser.json', JSON.stringify({state, report,
      prompt_id: handle.prompt_id, errors, copied: true}, null, 2));
    fs.writeFileSync('examples/gemma-tipo-trained-local.json', JSON.stringify(saved, null, 2));
    await page.screenshot({path: out + '/browser.png', fullPage: true});
    console.log('Trained weights, UI execution, save/reload, separated sockets, source/output/details and Copy PASS');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
