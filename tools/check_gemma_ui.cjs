// Isolated local test browser; never touches user browser sessions.
const fs=require('fs'), assert=require('assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const port=Number(process.argv[2]);assert(port===8196,'Use isolated test server');
 const base=`http://127.0.0.1:${port}`;
 const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1200}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(()=>globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);
  await page.waitForTimeout(1500);
  await page.evaluate(async g=>{const {app}=await import('/scripts/app.js');window.tipo=app;await app.loadGraphData(g);},JSON.parse(fs.readFileSync('examples/shared-interaction.json','utf8')));
  await page.waitForTimeout(500);
  const state=await page.evaluate(async()=>{
   const app=window.tipo,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');
   for(const [name,value] of Object.entries({output_format:'natural_language',tipo_lora:'pilot-002',expand_common:true,generation_max_tokens:1024})){
    const w=n.widgets.find(w=>w.name===name);w.value=value;w.callback?.(value);
   }
   const strings={common_prompt:'성인 여성 두 명이 오후 카페에서 컵을 주고받는다.',character1:'왼쪽의 금발 파란 눈 여성이 흰 셔츠를 입고 컵을 건넨다.',character2:'오른쪽의 갈색 머리 초록 눈 여성이 파란 드레스를 입고 컵을 받는다.'};
   for(const [name,value] of Object.entries(strings)){
    const input=n.inputs.find(s=>s.name===name),link=app.graph.links[input.link],src=app.graph.getNodeById(link.origin_id);
    src.widgets[0].value=value;src.applyToGraph?.();
   }
   const saved=app.graph.serialize();structuredClone(saved);await app.loadGraphData(saved);
   return {saved,inputs:(await app.graphToPrompt()).output[n.id].inputs};
  });
  await page.waitForTimeout(500);
  assert.equal(state.inputs.output_format,'natural_language');assert.equal(state.inputs.tipo_lora,'pilot-002');
  const ports=await page.evaluate(()=>{const n=window.tipo.graph._nodes.find(n=>n.type==='RSTIPOPrompt');return n.inputs.map((s,i)=>({name:s.name,y:n.getConnectionPos(true,i)[1]})).filter(s=>/^(common_prompt|character\d+|building_pipe)$/.test(s.name));});
  assert.equal(new Set(ports.map(s=>s.y)).size,ports.length);
  const response=page.waitForResponse(r=>/\/(api\/)?prompt$/.test(new URL(r.url()).pathname)&&r.request().method()==='POST');
  await page.evaluate(()=>window.tipo.queuePrompt(0,1));
  const posted=await response;assert(posted.ok(),await posted.text());const handle=await posted.json();
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('textarea')).some(t=>t.value.includes('is on the left')||t.value.includes('on the left')||t.value.includes('The first character')),{},{timeout:180000});
  const history=await(await page.request.get(base+'/history/'+handle.prompt_id)).json();
  const record=history[handle.prompt_id];assert.equal(record.status.status_str,'success');
  const report=JSON.parse(Object.values(record.outputs).find(o=>o.tipo_report).tipo_report[0]);
  assert.equal(report.adapter,'pilot-002');assert.equal(report.lora_targets,190);assert(report.model_calls>=1&&report.model_calls<=2);
  assert.equal(report.slots.length,3);assert.equal(report.output_format,'natural_language');
  assert.equal(errors.length,0,errors.join('\n'));
  fs.writeFileSync('validation/gemma-ui.json',JSON.stringify({ports,report,prompt_id:handle.prompt_id,errors},null,2));
  fs.writeFileSync('examples/gemma-tipo-korean.json',JSON.stringify(state.saved,null,2));
  await page.screenshot({path:'validation/gemma-ui.png',fullPage:true});
  console.log('Live Gemma TIPO UI + Korean natural-language + save/reload + separated sockets PASS',handle.prompt_id);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
