#!/usr/bin/env node
// Headless smoke test for src/web/index.html.
//
//   node tools/web-smoke.js
//
// Stubs just enough DOM + fetch to run the page script and drive its buttons, so a
// runtime error in the UI (like the scope bug that silently froze every step) fails
// here instead of in the browser. Exits non-zero on any error.
const fs = require('fs');

const html = fs.readFileSync('src/web/index.html', 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) {
  console.error('no <script> block found in src/web/index.html');
  process.exit(1);
}
const code = m[1];

const noop = () => {};
const ctx = new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : noop),
  set: (t, k, v) => { t[k] = v; return true; },
});

const els = {};
function mkEl(id) {
  return {
    id, textContent: '', className: '', innerHTML: '', value: '',
    width: 400, height: 400, disabled: false, onclick: null, onchange: null, oninput: null,
    classList: { add: noop, remove: noop },
    getContext: () => ctx,
    click() { return this.onclick ? this.onclick() : undefined; },
  };
}
const $ = id => (els[id] = els[id] || mkEl(id));

let calls = 0;
function state(step, alive) {
  return {
    size: 8, score: step > 3 ? 1 : 0, step, alive, illegal: 0,
    death: alive ? '' : 'wall', dir: 'right',
    food: [2, 5], snake: [[4, 4], [4, 3], [4, 2]],
    fatal: [false, false, true, false],
    last: {
      dir: 'left', probs: [0.001, 0.31, 0.48, 0.209], conf: 0.244,
      act: 1, tokens: 175, ms: 650,
    },
    prompt: '8x8 snake board. H=head o=body *=food .=empty.\n0 ........\n',
  };
}

global.document = { getElementById: $, addEventListener: noop };
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = noop;
global.fetch = async (url) => {
  calls++;
  if (url.indexOf('/api/new') === 0) return { ok: true, json: async () => state(0, true) };
  if (url.indexOf('/api/step') === 0) {
    const alive = calls % 6 !== 0;               // die every 5th move
    return { ok: true, json: async () => state(calls, alive) };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

const errors = [];
process.on('uncaughtException', e => errors.push('uncaught: ' + e.message));
process.on('unhandledRejection', e => errors.push('unhandled: ' + (e && e.message)));

const check = (label, ok) => {
  console.log((ok ? '  ok   ' : '  FAIL ') + label);
  if (!ok) errors.push(label);
};

(async () => {
  try {
    eval(code);
  } catch (e) {
    errors.push('eval: ' + e.message);
  }
  await new Promise(r => setTimeout(r, 60));
  check('page loads and calls /api/new', $('c-status').textContent === '就绪');

  for (let i = 0; i < 5; i++) {
    try {
      await $('b-step').onclick();
    } catch (e) {
      errors.push('step ' + i + ': ' + e.message);
    }
    await new Promise(r => setTimeout(r, 5));
  }
  check('decision panel rendered', $('dec').innerHTML.indexOf('opt') >= 0);
  check('stats panel rendered', $('st').innerHTML.length > 0);
  check('prompt panel filled', $('prompt').textContent.length > 10);
  check('five steps issued five requests', calls === 6);
  check('death is reported in the status chip', $('c-status').textContent.indexOf('墙') >= 0);
  // the stub repeats the same board every step, so a cycle must be flagged
  check('repeating board is reported as a cycle',
        $('dec').innerHTML.indexOf('陷入循环') >= 0);
  // the stub's chosen direction is always the fatal one
  check('observation panel counts fatal picks',
        $('st').innerHTML.indexOf('会立刻死掉') >= 0 &&
        $('st').innerHTML.indexOf('循环') >= 0);

  try { await $('b-reset').onclick(); } catch (e) { errors.push('reset: ' + e.message); }
  await new Promise(r => setTimeout(r, 40));
  check('reset works', $('c-status').textContent === '就绪');

  try { await $('policy').onchange(); } catch (e) { errors.push('policy: ' + e.message); }
  await new Promise(r => setTimeout(r, 40));
  check('policy switch works', $('c-status').textContent === '就绪');
  check('switching policy clears the observations', $('st').innerHTML.indexOf('积累观察数据') >= 0);

  try { await $('b-bench').onclick(); } catch (e) { errors.push('bench: ' + e.message); }
  await new Promise(r => setTimeout(r, 20));
  check('three-way bench renders a verdict',
        $('st').innerHTML.indexOf('random') >= 0 && $('st').innerHTML.indexOf('model') >= 0);
  // the stub's board never changes, so games abort on the repeat instead of
  // grinding to the 400-step guard; at least one must be counted that way
  const bh = $('st').innerHTML;
  check('bench aborts looping games and counts them',
        bh.indexOf('循环中止') >= 0 && /<td class="num bad">[1-9]\d*\/\d+<\/td>/.test(bh));

  if (errors.length) {
    console.log('\n' + errors.length + ' failure(s)\n' + errors.join('\n'));
    process.exit(1);
  }
  console.log('\nweb ui smoke test passed');
})();
