// Real isolated frontend: Korean STRINGs, named save/reload, live queue and result.
const fs=require('fs'),assert=require('assert/strict'),{chromium}=require('playwright');
(async()=>{
 const port=Number(process.argv[2]);assert(port>=8190&&port<=8198);
 const browser=await chromium.launch({headless:true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{}),args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1200}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}`);
  await page.waitForFunction(()=>globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);
  await page.waitForTimeout(4000);
  await page.evaluate(async g=>{const{app}=await import('/scripts/app.js');window.rsLanguageApp=app;await app.loadGraphData(g);},JSON.parse(fs.readFileSync('examples/shared-interaction.json')));
  await page.waitForTimeout(700);
  await page.evaluate(async()=>{
   const app=window.rsLanguageApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');
   const sources={common_prompt:'성인 여성 두 명이 카페에서 컵을 주고받는다.',
     character1:'(금발:1.2), blue eyes, 흰 셔츠, 왼쪽, 컵 건네기',
     character2:'갈색 머리, green eyes, 파란 원피스, 오른쪽, 컵 받기'};
   for(const[name,text]of Object.entries(sources)){
    const socket=n.inputs.find(i=>i.name===name),origin=app.graph.getNodeById(app.graph.links[socket.link].origin_id);
    origin.widgets[0].value=text;origin.applyToGraph?.();
   }
   for(const[name,val]of Object.entries({input_language:'auto',language_model:'auto',translation_max_tokens:4096,
       excluded_tags:'watermark, 서명',rating:'safe',expand_common:true,max_new_tokens:64,seed:42})){
    const w=n.widgets.find(w=>w.name===name);w.value=val;w.callback?.(val);
   }
   const data=app.graph.serialize();structuredClone(data);await app.loadGraphData(data);
  });
  await page.waitForTimeout(700);
  const state=await page.evaluate(async()=>{
   const app=window.rsLanguageApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');
   const p=(await app.graphToPrompt()).output[n.id].inputs;
   return{inputs:p,hiddenModel:n.widgets.find(w=>w.name==='language_model').type==='rs-tipo-hidden',
     ports:n.inputs.map((s,i)=>({name:s.name,y:n.getConnectionPos(true,i)[1]}))
      .filter(s=>s.name==='building_pipe'||/^(common_prompt|character\d+)$/.test(s.name))};
  });
  assert(state.hiddenModel);assert(state.inputs.character1.includes('(금발:1.2)'));
  assert.equal(state.inputs.input_language,'auto');assert.equal(state.inputs.translation_max_tokens,4096);
  assert.equal(new Set(state.ports.map(p=>p.y)).size,state.ports.length);
  const q=await(await page.request.get(`http://127.0.0.1:${port}/queue`)).json();
  assert(!q.queue_running.length&&!q.queue_pending.length,'Test queue busy; leave running work alone.');
  const posted=page.waitForResponse(r=>/^\/(?:api\/)?prompt$/.test(new URL(r.url()).pathname)&&r.request().method()==='POST');
  await page.evaluate(()=>window.rsLanguageApp.queuePrompt(0,1));
  const response=await posted;assert(response.ok());const handle=await response.json();
  fs.writeFileSync('validation/korean-ui.pending.json',JSON.stringify(handle));
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('textarea')).some(t=>t.value.startsWith('Korean source → English TIPO input')),null,{timeout:180000});
  const rendered=await page.locator('textarea[aria-label="TIPO expansion result"]').inputValue();
  const history=await(await page.request.get(`http://127.0.0.1:${port}/history/${handle.prompt_id}`)).json();
  const record=history[handle.prompt_id];assert.equal(record.status.status_str,'success');
  const report=JSON.parse(Object.values(record.outputs).find(o=>o.tipo_report).tipo_report[0]);
  assert(report.language.applied);assert.equal(report.language.converted.excluded_tags,'watermark, signature');
  assert(report.language.converted.characters[0].startsWith('(blonde hair:1.2), blue eyes'));
  assert.equal(errors.length,0,errors.join('\n'));
  fs.writeFileSync('validation/korean-ui.json',JSON.stringify({state,prompt_id:handle.prompt_id,rendered,report,errors},null,2));
  await page.screenshot({path:'validation/korean-ui.png'});fs.unlinkSync('validation/korean-ui.pending.json');
  console.log('Korean frontend PASS',handle.prompt_id);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
