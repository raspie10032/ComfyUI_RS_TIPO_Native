import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
let extension;
const app = {configuringGraph: false, registerExtension(e) { extension = e; }};
globalThis.comfyAPI = {widgetInputs: {setWidgetConfig(input) {input.configured = true;}}};
const sockets = await readFile(new URL('../web/sockets.js', import.meta.url), 'utf8');
const socketURL = 'data:text/javascript;base64,' + Buffer.from(sockets).toString('base64');
const source = (await readFile(new URL('../web/interface.js', import.meta.url), 'utf8'))
    .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.tipoTestApp;')
    .replace('import { api } from "../../scripts/api.js";', 'const api = {addEventListener(){}};')
    .replace('"./download-status.js"', JSON.stringify('data:text/javascript;base64,' + Buffer.from(await readFile(new URL('../web/download-status.js', import.meta.url), 'utf8')).toString('base64')))
    .replace('"./sockets.js"', JSON.stringify(socketURL));
globalThis.tipoTestApp = app;
await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
class Node {
    constructor() {
        this.widgets = [{name: 'character_count', value: 2}, {name: 'advanced', value: false}];
        this.inputs = [{name:'common_prompt',type:'STRING'}, {name:'character1',type:'STRING',link:1}];
        this.outputs = [{name:'building_pipe'}, {name:'common_prompt'}];
        this.size=[390,400]; this.properties={};
        this.primitive={type:'PrimitiveNode', widgets_values:['restored source'],widgets:[{value:''}], applyToGraph(){}};
        this.graph={links:{1:{origin_id:9}},getNodeById:()=>this.primitive};
    }
    addInput(name,type,extra){this.inputs.push({name,type,...extra});}
    removeInput(i){this.inputs.splice(i,1);}
    addOutput(name,type){this.outputs.push({name,type});}
    removeOutput(i){this.outputs.splice(i,1);}
    addWidget(type,name,value){const w={type,name,value};this.widgets.push(w);return w;}
    computeSize(){return this.size;}
    setSize(size){this.size=size;}
    setDirtyCanvas(){}
}
await extension.beforeRegisterNodeDef(Node,{name:'RSTIPOPrompt'});
const node=new Node();
app.configuringGraph=true;
node.onConfigure({properties:{rs_tipo_widget_values:{character_count:1, advanced:false, tipo_results:{bad:true}}}});
await new Promise(r=>setTimeout(r,15));
assert.equal(node.inputs.length,2,'must not change sockets while core configures forceInput');
app.configuringGraph=false;
await new Promise(r=>setTimeout(r,30));
assert.equal(node.widgets.find(w=>w.name==='character_count').value,1);
assert.equal(node.inputs.filter(i=>i.name==='character1').length,1);
assert(node.inputs.find(i=>i.name==='character1').configured);
assert.equal(node.widgets.find(w=>w.name==='character1').type,'rs-tipo-string-socket');
assert.equal(node.widgets.find(w=>w.name==='character1').computeSize()[1],20);
assert.equal(node.widgets[0].name,'common_prompt');
assert.equal(node.widgets[1].name,'character1');
assert.equal(node.primitive.widgets[0].value,'restored source');
node.primitive.widgets[0].value='edited after loading';
node.onGraphConfigured();
await new Promise(r=>setTimeout(r,15));
assert.equal(node.primitive.widgets[0].value,'edited after loading','must never replay old saved text during an interactive refresh');
const saved={}; node.onSerialize(saved);
assert.equal(saved.properties.rs_tipo_widget_values.character1,'edited after loading');
assert.equal(saved.properties.rs_tipo_widget_values.character_count,1);
assert.doesNotThrow(()=>structuredClone(saved));
node.addInput('interaction_prompt', 'STRING', {link:2});
node.addWidget('converted-widget','interaction_prompt','older description');
node.onGraphConfigured();
await new Promise(r=>setTimeout(r,15));
assert(!node.inputs.some(s=>s.name==='interaction_prompt'));
assert(!node.widgets.some(s=>s.name==='interaction_prompt'));
assert.equal(node.properties.rs_tipo_legacy_interaction,'older description');
assert.equal(node.primitive.widgets[0].value,'edited after loading');
console.log('Deferred core conversion, legacy STRING backing, named restore and preservation of interactive edits PASS');
