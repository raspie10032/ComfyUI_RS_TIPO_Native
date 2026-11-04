// Isolated frontend check: named restoration, old workflows, real queue and report.
// Explicit test-server port required; never opens or modifies the user's tabs.
const fs = require('fs'), assert = require('assert/strict');
const {chromium} = require('playwright');
(async () => {
 const port = Number(process.argv[2]); assert(Number.isInteger(port) && port > 0);
 const base = `http://127.0.0.1:${port}`;
 const browser = await chromium.launch({headless:true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{}),args:['--no-sandbox']});
 try {
  const page = await browser.newPage({viewport:{width:1600,height:1100}}), errors=[];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(() => globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);
  await page.waitForTimeout(4500); // let initial workflow restoration finish
  const workflow=JSON.parse(fs.readFileSync('examples/shared-interaction.json','utf8'));
  await page.evaluate(async g => {const {app}=await import('/scripts/app.js');window.metadataApp=app;await app.loadGraphData(g);}, workflow);
  await page.waitForFunction(()=>window.metadataApp.graph._nodes.some(n=>n.type==='RSTIPOPrompt'));
  await page.waitForTimeout(700);
  const before = await page.evaluate(async () => {
   const app=window.metadataApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');
   const data=(await app.graphToPrompt()).output[n.id].inputs;
   const initial=Object.fromEntries(['rating','quality','quality_era'].map(k=>[k,data[k]]));
   for (const [name,value] of Object.entries({rating:'safe',quality:'masterpiece',quality_era:'newest',expand_common:true,max_new_tokens:64,seed:8910})) {
    const w=n.widgets.find(w=>w.name===name);w.value=value;w.callback?.(value);
   }
   const saved=app.graph.serialize();structuredClone(saved);await app.loadGraphData(saved);return initial;
  });
  assert.deepEqual(before,{rating:'unspecified',quality:'unspecified',quality_era:'unspecified'});
  await page.waitForTimeout(700);
  const after=await page.evaluate(async()=>{
   const app=window.metadataApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');
   const inputs=(await app.graphToPrompt()).output[n.id].inputs;
   return {inputs,eraHidden:n.widgets.find(w=>w.name==='quality_era').type==='rs-tipo-hidden',
    ports:n.inputs.map((s,i)=>({name:s.name,y:n.getConnectionPos(true,i)[1]}))
     .filter(s=>s.name==='building_pipe'||/^(common_prompt|character\d+)$/.test(s.name))};
  });
  assert.equal(after.inputs.rating,'safe');assert.equal(after.inputs.quality,'masterpiece');assert.equal(after.inputs.quality_era,'newest');
  assert.equal(after.eraHidden,true);assert(after.inputs.character1.includes('(blonde hair:1.2)'));
  assert.equal(new Set(after.ports.map(s=>s.y)).size,after.ports.length);
  // Queue through the actual frontend and receive the actual websocket report.
  let handle;
  if (process.argv[3]) {
   handle={prompt_id:process.argv[3]};
   const history=await (await page.request.get(base+'/history/'+handle.prompt_id)).json();
   const record=history[handle.prompt_id];assert.equal(record.status.status_str,'success');
   const node=Object.entries(record.outputs).find(([,o])=>o.tipo_report);
   for(const name of ['rating','quality','quality_era','character1','character2'])
    assert.equal(record.prompt[2][node[0]].inputs[name],after.inputs[name]);
   await page.evaluate(output=>window.metadataApp.graph._nodes.find(n=>n.type==='RSTIPOPrompt').onExecuted(output),node[1]);
  } else {
   const queue=await (await page.request.get(base+'/queue')).json();
   assert(!queue.queue_running.length && !queue.queue_pending.length,'Test server is busy; leave existing work alone.');
   const response=page.waitForResponse(r=>new URL(r.url()).pathname.match(/^\/(?:api\/)?prompt$/) && r.request().method()==='POST');
   await page.evaluate(()=>window.metadataApp.queuePrompt(0,1));
   const posted=await response;assert(posted.ok());handle=await posted.json();
  }
  console.log('Observing frontend prompt',handle.prompt_id);
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('textarea')).some(t=>t.value.startsWith('TIPO metadata (guidance)\nquality: masterpiece, newest\nrating: safe')),null,{timeout:180000});
  const reportText=await page.locator('textarea[aria-label="TIPO expansion result"]').inputValue();
  const history=await (await page.request.get(base+'/history/'+handle.prompt_id)).json();
  assert.equal(history[handle.prompt_id].status.status_str,'success');assert.equal(errors.length,0,errors.join('\n'));
  const report=JSON.parse(Object.values(history[handle.prompt_id].outputs).find(o=>o.tipo_report).tipo_report[0]);
  assert.equal(report.slots.length,3);assert.equal(report.metadata.rating,'safe');
  fs.writeFileSync(`validation/metadata-ui-${port}.json`,JSON.stringify({before,after,prompt_id:handle.prompt_id,reportSource:process.argv[3]?'completed history':'live websocket',reportText,errors},null,2));
  await page.screenshot({path:`validation/metadata-ui-${port}.png`});
  console.log('Metadata UI PASS',port);
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
