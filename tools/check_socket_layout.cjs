const fs=require('fs'),assert=require('assert/strict');
const{chromium}=require('playwright');
// Isolated browser test: does not queue inference or modify user tabs.
fs.mkdirSync('validation/compatibility',{recursive:true});
(async()=>{
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{}),args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1700,height:1200}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
await page.goto(`http://127.0.0.1:${process.argv[2]}/`);await page.waitForFunction(()=>globalThis.LiteGraph?.registered_node_types?.RSTIPOPrompt);await page.waitForTimeout(4500);
await page.evaluate(async g=>{const{app}=await import('/scripts/app.js');await app.loadGraphData(g);window.testApp=app;},JSON.parse(fs.readFileSync('examples/shared-interaction.json','utf8')));
await page.waitForFunction(()=>window.testApp.graph._nodes.some(n=>n.type==='RSTIPOPrompt'));await page.waitForTimeout(800);
const states=[];
for(const count of [2,10,1,0,3]){
 await page.evaluate(count=>{const n=window.testApp.graph._nodes.find(n=>n.type==='RSTIPOPrompt');const w=n.widgets.find(w=>w.name==='character_count');w.value=count;w.callback(count);},count);
 await page.waitForTimeout(250);
 states.push(await page.evaluate(()=>{const n=window.testApp.graph._nodes.find(n=>n.type==='RSTIPOPrompt');return {count:n.widgets.find(w=>w.name==='character_count').value,ports:n.inputs.map((s,i)=>({name:s.name,link:s.link,y:n.getConnectionPos(true,i)[1]})).filter(s=>s.name==='building_pipe'||/^(common_prompt|interaction_prompt|character\d+)$/.test(s.name))}}));
 const ports=states.at(-1).ports;assert(!ports.some(p=>p.name==='interaction_prompt'));assert.equal(new Set(ports.map(s=>s.y)).size,ports.length,`overlapping sockets at count ${count}`);
}
const roundtrip=await page.evaluate(async()=>{const app=window.testApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');n.widgets.find(w=>w.name==='excluded_tags').value='long_hair, white shirt';const advanced=n.widgets.find(w=>w.name==='advanced');advanced.value=true;advanced.callback(true);if(!n.inputs.some(s=>s.name==='seed')){const {convertToInput}=await import('/extensions/core/widgetInputs.js');convertToInput(n,n.widgets.find(w=>w.name==='seed'),['INT',{default:0,min:0,max:18446744073709551615}]);}const seed=LiteGraph.createNode('PrimitiveNode');app.graph.add(seed);seed.connect(0,n,n.inputs.findIndex(s=>s.name==='seed'));seed.widgets[0].value=42;seed.applyToGraph?.();advanced.value=false;advanced.callback(false);const s=app.graph.serialize();structuredClone(s);await app.loadGraphData(s);return true;});
await page.waitForTimeout(500);
const after=await page.evaluate(async()=>{const app=window.testApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');const prompt=await app.graphToPrompt();const seed=n.inputs.findIndex(s=>s.name==='seed');return {source:prompt.output[n.id].inputs.character1,seed:prompt.output[n.id].inputs.seed,seedHidden:n.widgets.find(w=>w.name==='seed').type==='rs-tipo-hidden',seedPos:n.getConnectionPos(true,seed),ports:n.inputs.map((s,i)=>({name:s.name,y:n.getConnectionPos(true,i)[1]})).filter(s=>s.name==='seed'||s.name==='building_pipe'||/^(common_prompt|interaction_prompt|character\d+)$/.test(s.name))};});
assert.equal(after.seedHidden,false);assert.equal(after.seed,42);assert(after.source.includes('blonde hair'));assert.equal(new Set(after.ports.map(s=>s.y)).size,after.ports.length);
const exclusion=await page.evaluate(async()=>{const app=window.testApp,n=app.graph._nodes.find(n=>n.type==='RSTIPOPrompt');return (await app.graphToPrompt()).output[n.id].inputs.excluded_tags;});assert.equal(exclusion,'long_hair, white shirt');
await page.evaluate(async g=>{await window.testApp.loadGraphData(g);},JSON.parse(fs.readFileSync('examples/shared-interaction.json','utf8')));
await page.waitForTimeout(700);
await page.setViewportSize({width:1280,height:1060});
await page.evaluate(()=>{const app=window.testApp;app.canvas.ds.scale=.95;app.canvas.ds.offset=[70,135];app.canvas.setDirty(true,true);});
await page.waitForTimeout(400);await page.screenshot({path:`validation/compatibility/socket-layout-${(process.argv[3]||'local')}.png`});
fs.writeFileSync(`validation/compatibility/socket-layout-${process.argv[3]}.json`,JSON.stringify({states,after,errors},null,2));console.log('PASS',process.argv[3],JSON.stringify({counts:states.map(s=>s.count),after,errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
