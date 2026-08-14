"use strict";
/* Pure sampling + tiering logic. Loaded by index.html; `node sampler.js` runs the self-test. */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* cuts = { weak:[lo,hi], medium:[lo,hi], strong:[lo,hi] }, inclusive; gaps between
   ranges are buffer zones — models there get no tier and never sample */
function tierOf(m, cuts, overrides) {
  if (overrides && overrides[m.name]) return overrides[m.name];
  if (m.eci == null) return null;
  for (const t of ["Weak", "Medium", "Strong"]) {
    const r = cuts[t.toLowerCase()];
    if (r && m.eci >= r[0] && m.eci <= r[1]) return t;
  }
  return null;
}

function isEligible(m, cap) {
  return m.on_openrouter || (m.open_weights && m.params != null && m.params <= cap);
}
function isLocal(m, cap) { return isEligible(m, cap) && !m.on_openrouter; }

function buildCells(pool, cuts, overrides, cap) {
  const cells = new Map();
  for (const m of pool) {
    if (!isEligible(m, cap) || !m.half) continue;
    const t = tierOf(m, cuts, overrides);
    if (!t) continue;
    const k = m.half + "|" + t;
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(m);
  }
  return cells;
}

function pickN(arr, n, rng) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, Math.min(n, a.length));
}

function sampleSet(pool, opts) {
  const rng = mulberry32(opts.seed);
  const cells = buildCells(pool, opts.cuts, opts.overrides, opts.cap);
  let best = null;
  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    const picks = [];
    cells.forEach(ms => picks.push(...pickN(ms, opts.perCell, rng)));
    let cost = 0;
    for (const m of picks) if (m.price != null) cost += m.price;
    const draw = { names: picks.map(m => m.name).sort(), cost, fit: opts.budget == null || cost <= opts.budget, attempts: attempt };
    if (draw.fit) return draw;
    if (!best || draw.cost < best.cost) best = draw;
  }
  return best || { names: [], cost: 0, fit: true, attempts: 0 };
}

const Sampler = { mulberry32, tierOf, isEligible, isLocal, buildCells, pickN, sampleSet };
if (typeof window !== "undefined") window.Sampler = Sampler;
else module.exports = Sampler;

if (typeof window === "undefined") {
  const assert = require("assert");
  const S = module.exports;
  const P = (over) => Object.assign({ name:"m", dev:"X", eci:120, eci_source:"real",
    release:"2025-01-01", half:"2025-H1", desc:"", params:null, open_weights:false,
    on_openrouter:true, or_id:"x/m", price_in:1, price_out:3, price:4, tier_manual:null }, over);
  const cuts = { weak:[0,119], medium:[120,139], strong:[140,999] };

  // tierOf: ranges, override precedence, unscored, buffer gap
  assert.equal(S.tierOf(P({eci:150}), cuts, {}), "Strong");
  assert.equal(S.tierOf(P({eci:125}), cuts, {}), "Medium");
  assert.equal(S.tierOf(P({eci:100}), cuts, {}), "Weak");
  assert.equal(S.tierOf(P({eci:150, name:"a"}), cuts, {a:"Weak"}), "Weak");
  assert.equal(S.tierOf(P({eci:null, eci_source:null}), cuts, {}), null);
  assert.equal(S.tierOf(P({eci:null, name:"b"}), cuts, {b:"Medium"}), "Medium");
  const gap = { weak:[0,110], medium:[120,139], strong:[140,999] };
  assert.equal(S.tierOf(P({eci:115}), gap, {}), null);          // in the buffer
  assert.equal(S.tierOf(P({eci:115, name:"g"}), gap, {g:"Weak"}), "Weak");

  // eligibility: OR, or open-weights under cap
  assert.ok(S.isEligible(P({}), 80e9));
  assert.ok(!S.isEligible(P({on_openrouter:false}), 80e9));
  assert.ok(S.isEligible(P({on_openrouter:false, open_weights:true, params:70e9}), 80e9));
  assert.ok(!S.isEligible(P({on_openrouter:false, open_weights:true, params:90e9}), 80e9));
  assert.ok(!S.isEligible(P({on_openrouter:false, open_weights:true, params:null}), 80e9));
  assert.ok(S.isLocal(P({on_openrouter:false, open_weights:true, params:1e9}), 80e9));
  assert.ok(!S.isLocal(P({}), 80e9));

  // determinism + distinctness
  const pool = [];
  for (let i = 0; i < 12; i++) pool.push(P({name:"m"+i, eci:100+i*5, price:1}));
  const opts = { seed:42, perCell:2, budget:null, cap:80e9, cuts, overrides:{}, maxAttempts:500 };
  const a = S.sampleSet(pool, opts), b = S.sampleSet(pool, opts);
  assert.deepEqual(a.names, b.names);
  assert.equal(new Set(a.names).size, a.names.length);
  const c = S.sampleSet(pool, Object.assign({}, opts, {seed:43}));
  assert.notDeepEqual(a.names, c.names);

  // perCell honored; small cells give everything they have
  const cells = S.buildCells(pool, cuts, {}, 80e9);
  let expected = 0;
  cells.forEach(ms => expected += Math.min(2, ms.length));
  assert.equal(a.names.length, expected);

  // budget retry finds a fitting draw
  const pool2 = [ P({name:"cheapA", price:1, eci:100}), P({name:"cheapB", price:1, eci:100}),
                  P({name:"exp", price:100, eci:100}) ];
  const r = S.sampleSet(pool2, { seed:1, perCell:1, budget:5, cap:80e9, cuts, overrides:{}, maxAttempts:500 });
  assert.ok(r.fit && r.cost <= 5, "budget respected: " + JSON.stringify(r));

  // impossible budget: fit=false, cheapest draw returned
  const r2 = S.sampleSet(pool2, { seed:1, perCell:3, budget:5, cap:80e9, cuts, overrides:{}, maxAttempts:50 });
  assert.ok(!r2.fit);
  console.log("sampler.js self-test OK");
}
