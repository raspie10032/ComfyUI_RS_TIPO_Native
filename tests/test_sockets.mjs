import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../web/sockets.js', import.meta.url), 'utf8');
const { updateSockets } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const node = {
    inputs: [{name:'common_prompt'}, ...Array.from({length:10}, (_,i)=>({name:`character${i+1}`,link:i===9?10:null}))],
    outputs: [{name:'building_pipe'}, {name:'common_prompt'}, ...Array.from({length:10},(_,i)=>({name:`character${i+1}`, links:i===9?[20]:null}))],
    removeInput(i){this.inputs.splice(i,1);}, addInput(name,type){this.inputs.push({name,type});},
    removeOutput(i){this.outputs.splice(i,1);}, addOutput(name,type){this.outputs.push({name,type});},
};
updateSockets(node,2);
assert.equal(node.outputs[11].name,'character10');
assert.equal(node.outputs[11].label,'character10 (inactive)');
assert.equal(node.inputs.find(i=>i.name==='character10').link,10);
assert.equal(node.inputs.some(i=>i.name==='character3'),false);
updateSockets(node,10);
assert.equal(node.outputs[11].links[0],20);
assert.equal(new Set(node.inputs.map(i=>i.name)).size,node.inputs.length);
assert.equal(node.inputs.filter(i=>/^character\d+$/.test(i.name)).length,10);
node.outputs[11].links=[];node.inputs.find(i=>i.name==='character10').link=null;
updateSockets(node,0);
assert.equal(node.outputs.length,2);
updateSockets(node,3);
assert.deepEqual(node.outputs.map(i=>i.name),['building_pipe','common_prompt','character1','character2','character3']);
assert.doesNotThrow(()=>structuredClone(JSON.parse(JSON.stringify(node))));
console.log('Socket count, inactive links, fixed output indexes and serialization PASS');
