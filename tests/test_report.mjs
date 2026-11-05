import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../web/report.js', import.meta.url), 'utf8');
const {resultText, modelSummary} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const report = {
    model_family: 'Gemma TIPO', adapter: 'local:scale-20pct-001', lora_targets: 190,
    adapter_details: {rank: 8, weights_sha256: 'a'.repeat(64)},
    raw_output: 'RAW MODEL TEXT',
    slots: [{slot: 'common', original: '카페', output: 'cafe'},
            {slot: 1, original: '컵을 건넴', output: '<script>plain model text</script>'},
            {slot: 2, original: '컵을 받음', output: 'receiving cup'}],
};
assert.equal(resultText(report, '1'), 'Character 1\n<script>plain model text</script>');
assert.equal(resultText(report, '2', 'source'), 'Character 2\n컵을 받음');
assert(resultText(report).includes('Common prompt\ncafe'));
assert(!resultText(report).includes('RAW MODEL TEXT'));
assert(resultText(report, 'all', 'details').includes('RAW MODEL TEXT'));
assert.equal(resultText({...report, slots: report.slots.slice(0, 2)}, '2'), '');
assert.equal(resultText(null), '');
assert(modelSummary(report).includes('local:scale-20pct-001 · rank 8 · 190 targets'));
const expanded = {...report, slots: [
    {slot: 'common', added_tags: [], added_tag_count: 0, expansion_status: 'unchanged'},
    {slot: 1, added_tags: ['head tilt'], added_tag_count: 1, expansion_status: 'expanded'},
    {slot: 2, expansion_status: 'translated'},
]};
assert(modelSummary(expanded).includes('common: unchanged / 1: +1 tags / 2: translated'));
assert(resultText(expanded, 'all', 'details').includes('added_tag_count'));
const recovered = {...report, warnings: [{stage: 'joint_prose', fields: ['character1'],
    action: 'source_restored_after_retry', reason: 'Malformed JSON'}]};
assert(modelSummary(recovered).includes('⚠ 1 recovery warning'));
assert(resultText(recovered, 'all', 'details').includes('source_restored_after_retry'));
console.log('Separate source/output/model details, slot routing and trained-weight identity PASS');
