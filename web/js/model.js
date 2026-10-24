// Runs the exported LightGBM models in the browser: prediction and exact TreeSHAP.
// Tree format is documented in src/olculu/export_web.py.
(function (root) {
  "use strict";

  function goesLeft(tree, node, value) {
    const missing = tree.m[node];
    let v = value;
    if (v === null || v === undefined) v = NaN;
    if (Number.isNaN(v) && missing !== 2) v = 0;
    if ((missing === 1 && Math.abs(v) <= 1e-35) || (missing === 2 && Number.isNaN(v))) {
      return tree.d[node] === 1;
    }
    return v <= tree.t[node];
  }

  function predictTree(tree, x) {
    let node = 0;
    while (tree.f[node] !== -1) {
      node = goesLeft(tree, node, x[tree.f[node]]) ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  }

  function predict(model, x) {
    let sum = 0;
    for (const tree of model.trees) sum += predictTree(tree, x);
    return sum;
  }

  // All quantiles of a quantile-model bundle, sorted per row so they never cross.
  function predictQuantiles(bundle, x) {
    const values = bundle.models.map((m) => predict(m, x));
    return values.sort((a, b) => a - b);
  }

  // ---- TreeSHAP (Lundberg et al. 2018, Algorithm 2) ----

  function extend(path, pz, po, pi) {
    const l = path.length;
    const m = path.map((e) => ({ d: e.d, z: e.z, o: e.o, w: e.w }));
    m.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
    for (let i = l - 1; i >= 0; i--) {
      m[i + 1].w += (po * m[i].w * (i + 1)) / (l + 1);
      m[i].w = (pz * m[i].w * (l - i)) / (l + 1);
    }
    return m;
  }

  function unwind(path, i) {
    const l = path.length - 1;
    const m = path.map((e) => ({ d: e.d, z: e.z, o: e.o, w: e.w }));
    let n = m[l].w;
    const o = m[i].o;
    const z = m[i].z;
    for (let j = l - 1; j >= 0; j--) {
      if (o !== 0) {
        const t = m[j].w;
        m[j].w = (n * (l + 1)) / ((j + 1) * o);
        n = t - (m[j].w * z * (l - j)) / (l + 1);
      } else {
        m[j].w = (m[j].w * (l + 1)) / (z * (l - j));
      }
    }
    for (let j = i; j < l; j++) {
      m[j].d = m[j + 1].d;
      m[j].z = m[j + 1].z;
      m[j].o = m[j + 1].o;
    }
    m.pop();
    return m;
  }

  function unwoundSum(path, i) {
    return unwind(path, i).reduce((s, e) => s + e.w, 0);
  }

  function treeShap(tree, x, phi) {
    function recurse(j, path, pz, po, pi) {
      let m = extend(path, pz, po, pi);
      if (tree.f[j] === -1) {
        for (let i = 1; i < m.length; i++) {
          phi[m[i].d] += unwoundSum(m, i) * (m[i].o - m[i].z) * tree.v[j];
        }
        return;
      }
      const feature = tree.f[j];
      const left = goesLeft(tree, j, x[feature]);
      const hot = left ? tree.l[j] : tree.r[j];
      const cold = left ? tree.r[j] : tree.l[j];
      let iz = 1;
      let io = 1;
      for (let k = 1; k < m.length; k++) {
        if (m[k].d === feature) {
          iz = m[k].z;
          io = m[k].o;
          m = unwind(m, k);
          break;
        }
      }
      const cover = tree.c[j];
      recurse(hot, m, (iz * tree.c[hot]) / cover, io, feature);
      recurse(cold, m, (iz * tree.c[cold]) / cover, 0, feature);
    }
    recurse(0, [], 1, 1, -1);
  }

  function expectedValue(model) {
    let sum = 0;
    for (const tree of model.trees) {
      let num = 0;
      let den = 0;
      for (let i = 0; i < tree.f.length; i++) {
        if (tree.f[i] === -1) {
          num += tree.v[i] * tree.c[i];
          den += tree.c[i];
        }
      }
      sum += num / den;
    }
    return sum;
  }

  // SHAP values per feature for one model; base + sum(phi) equals the prediction.
  function shap(model, x, nFeatures) {
    const phi = new Array(nFeatures).fill(0);
    for (const tree of model.trees) treeShap(tree, x, phi);
    return { phi, base: expectedValue(model) };
  }

  const api = { predict, predictQuantiles, shap, expectedValue };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OlculuModel = api;
})(typeof window !== "undefined" ? window : globalThis);
