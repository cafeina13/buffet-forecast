// The chef's screen. Runs both models in the browser (js/model.js) on demo services
// exported by src/olculu/train.py. Chef actions are kept in this browser only.
(function () {
  "use strict";

  const M = window.OlculuModel;
  const RISKS = [0.2, 0.1, 0.05];
  const MEALS = [
    { id: "breakfast", label: "Kahvaltı" },
    { id: "lunch", label: "Öğle" },
    { id: "dinner", label: "Akşam" },
  ];
  const GROUPS = {
    soup: "Çorba", red_meat: "Kırmızı et", chicken: "Tavuk", fish: "Balık", vegetables: "Sebze",
    rice_pasta: "Pilav ve makarna", salad: "Salata", dessert: "Tatlı", breakfast: "Kahvaltılık",
  };
  const GROUP_ORDER = Object.keys(GROUPS);
  const DAYS = ["Paz", "Pzt", "Sal", "Çar", "Per", "Cum", "Cmt"];
  const MONTHS = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"];
  const LEAD_HOURS = 0.25;

  const nf = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 1 });
  const nf0 = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 0 });
  const pct = (x) => "%" + nf.format(x * 100);
  const pct0 = (x) => "%" + nf0.format(x * 100);

  // Front-office counts the chef can change, per meal. They feed the presence rule
  // (src/olculu/features.py: present_guests). Breakfast first-meal share is not offered:
  // the dish model barely reacts to it on this synthetic data.
  const PRESENCE_CONTROLS = {
    breakfast: [
      { key: "departures_very_early", label: "Kahvaltıdan önce ayrılan" },
    ],
    lunch: [
      { key: "excursions", label: "Tura çıkan" },
      { key: "departures_early", label: "Öğleden önce ayrılan" },
    ],
    dinner: [
      { key: "alacarte", label: "A la carte rezervasyonu" },
      { key: "arrivals_night", label: "Bu gece geç inecek (akşama yetişmez)" },
    ],
  };
  const PRESENCE_KEYS = ["in_house", "arrivals_night", "departures_early", "departures_very_early", "excursions", "alacarte"];
  // Reasons are counterfactual (the forecast with the input changed), not SHAP shares: on this
  // synthetic data the models learned weak inputs (temperature, child share, nationality) unreliably,
  // and SHAP faithfully attributed that noise (site audit 1, findings 2.3-2.6).
  const MIN_TEMP_EFFECT = 0.05; // a temperature reason needs at least a 5% effect on the median
  const MILD_DAY_C = 27;        // reference "mild day" for the temperature comparison

  let demo, G, D, gIdx, dIdx;
  const state = {
    day: 0, meal: "dinner", risk: 0.1, occ: 0, temp: 0, tab: "plan", clock: 0,
    // Per-service overrides; null means "as booked".
    match: null, themed: null, child: null, presence: {},
  };
  let builtFor = null;

  // ---------- storage (per-viewer convenience only) ----------
  function storeKey(service, dishId) {
    return `olculu-demo:${service.date}:${service.meal}:${dishId}`;
  }
  function loadAction(service, dishId) {
    try {
      const raw = localStorage.getItem(storeKey(service, dishId));
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return memory[storeKey(service, dishId)] || null;
    }
  }
  const memory = {};
  function saveAction(service, dishId, value) {
    const k = storeKey(service, dishId);
    memory[k] = value;
    try {
      if (value) localStorage.setItem(k, JSON.stringify(value));
      else localStorage.removeItem(k);
    } catch (e) { /* storage unavailable: kept in memory */ }
  }

  // ---------- helpers ----------
  function fmtTime(h) {
    const total = Math.round(h * 60);
    const hh = Math.floor(total / 60);
    const mm = total % 60;
    return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  }
  function parseDate(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d);
  }
  function roundHalfUp(units) {
    return Math.max(0.5, Math.ceil(units * 2 - 1e-9) / 2);
  }
  function roundHalf(units) {
    return Math.max(0, Math.round(units * 2) / 2);
  }
  function interp(levels, values, level) {
    if (level <= levels[0]) return values[0];
    for (let i = 1; i < levels.length; i++) {
      if (level <= levels[i]) {
        const t = (level - levels[i - 1]) / (levels[i] - levels[i - 1]);
        return values[i - 1] + t * (values[i] - values[i - 1]);
      }
    }
    return values[values.length - 1];
  }
  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of [].concat(children || [])) {
      if (c === null || c === undefined) continue;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
  }
  function unitText(n, unit) {
    return `${nf.format(n)} ${unit}`;
  }

  function currentService() {
    const date = demo.dates[state.day];
    return demo.services.find((s) => s.date === date && s.meal === state.meal);
  }

  // ---------- model ----------
  // Occupancy slider moves occupancy by points; counts scale with it, in-house guests capped at capacity.
  function occupancyFactor(service) {
    const occ = service.guest_x[gIdx.occupancy];
    const target = Math.min(1, Math.max(0.05, occ + state.occ / 100));
    return target / occ;
  }

  function basePresence(service) {
    const f = occupancyFactor(service);
    const p = Object.fromEntries(PRESENCE_KEYS.map((k) => [k, service.presence[k] * f]));
    p.in_house = Math.min(p.in_house, demo.support.capacity_guests);
    return p;
  }

  function clamp(v, [lo, hi]) {
    return Math.min(hi, Math.max(lo, v));
  }

  function scenario(service) {
    const dinner = state.meal === "dinner";
    const match = dinner ? (state.match === null ? service.event === "match" : state.match) : false;
    let themed = dinner ? (state.themed === null ? service.event === "themed" : state.themed) : false;
    if (match) themed = false;
    const base = basePresence(service);
    const presence = { ...base };
    for (const [k, v] of Object.entries(state.presence)) presence[k] = v;
    const baseChild = service.guest_x[gIdx.child_share];
    const child = state.child === null ? baseChild : state.child;
    return { match, themed, presence, base, child, occ: state.occ, temp: state.temp };
  }

  function presentGuests(meal, p) {
    const v = meal === "breakfast" ? p.in_house - p.departures_very_early
      : meal === "lunch" ? p.in_house - p.departures_early - p.excursions
        : p.in_house - p.arrivals_night - p.alacarte;
    return Math.max(1, v);
  }

  function setEvents(x, idx, sc) {
    if (state.meal !== "dinner") return;
    x[idx.event_match] = sc.match ? 1 : 0;
    x[idx.event_themed] = sc.themed ? 1 : 0;
  }

  function guestVector(service, sc) {
    const x = service.guest_x.slice();
    x[gIdx.occupancy] = Math.min(1, x[gIdx.occupancy] * occupancyFactor(service));
    x[gIdx.temperature] = clamp(x[gIdx.temperature] + sc.temp, demo.support.temperature);
    x[gIdx.child_share] = sc.child;
    setEvents(x, gIdx, sc);
    return x;
  }

  function dishVector(dish, sc) {
    const x = dish.x.slice();
    x[dIdx.temperature] = clamp(x[dIdx.temperature] + sc.temp, demo.support.temperature);
    x[dIdx.child_share] = sc.child;
    setEvents(x, dIdx, sc);
    return x;
  }

  function hasHistory(x) {
    return x[dIdx.hist_per100] !== null && x[dIdx.hist_per100] !== undefined;
  }

  // Event effect on guests: the forecast with the event switched off, minus with it on.
  function guestDrivers(gx, present, sc) {
    const items = [];
    for (const [flag, label] of [["event_match", "maç akşamı"], ["event_themed", "tema gecesi"]]) {
      if (gx[gIdx[flag]] !== 1) continue;
      const off = gx.slice();
      off[gIdx[flag]] = 0;
      const withEvent = M.predict(G.models[1], gx) * present;
      const without = M.predict(G.models[1], off) * present;
      items.push({ label, people: withEvent - without, without });
    }
    return items;
  }

  // One sentence per dish: guests, the dish's own recent consumption, and temperature only when the
  // model's effect is large enough and points the way a kitchen would expect.
  function explain(x, info, guests) {
    const parts = [`${nf0.format(guests)} misafir bekleniyor`];
    if (hasHistory(x)) {
      parts.push(`bu yemeğin bu öğündeki talebi şimdiye kadar 100 misafir başına ortalama ${nf.format(x[dIdx.hist_per100])} kg (tükenen servisler düzeltilerek)`);
    } else {
      parts.push("bu yemeğin bu öğünde geçmişi yok, tahmin daha belirsiz");
    }
    const t = x[dIdx.temperature];
    if (info.heat !== 0 && Math.abs(t - MILD_DAY_C) >= 3) {
      const mild = x.slice();
      mild[dIdx.temperature] = MILD_DAY_C;
      const now = M.predict(D.models[1], x);
      const ref = M.predict(D.models[1], mild);
      const effect = ref > 0 ? (now - ref) / ref : 0;
      const expected = info.heat * Math.sign(t - MILD_DAY_C);
      if (Math.abs(effect) >= MIN_TEMP_EFFECT && Math.sign(effect) === expected) {
        const word = t > MILD_DAY_C ? "sıcak" : "serin";
        parts.push(`${word} hava (${nf0.format(t)} °C) talebi ${effect > 0 ? "artırıyor" : "azaltıyor"} (ılık bir güne göre ${effect > 0 ? "+" : "−"}%${nf0.format(Math.abs(effect) * 100)})`);
      }
    }
    return parts.join("; ") + ".";
  }

  function refillPlan(totalUnits, medianKg, unitKg, profile) {
    const slots = profile.slots;
    const shares = profile.shares;
    const cum = [];
    shares.reduce((acc, s, i) => (cum[i] = acc + s), 0);
    const initial = Math.min(totalUnits, Math.max(1, Math.ceil(totalUnits * 0.4)));
    // Small batches, often: about a fifth of the total each, never less than one unit.
    const batchSize = Math.max(1, Math.ceil(totalUnits * 0.2 * 2) / 2);
    let out = initial * unitKg;
    let left = totalUnits - initial;
    const refills = [];
    const start = slots[0];
    while (left > 1e-9) {
      const batch = Math.min(batchSize, left);
      // Refill when the tray on the buffet is expected to be half empty, minus cooking lead time.
      const idx = cum.findIndex((c) => medianKg * c >= out - 0.5 * unitKg);
      if (idx < 0) break;
      const t = Math.round(Math.max(start, slots[idx] - LEAD_HOURS) * 4) / 4;
      const last = refills[refills.length - 1];
      if (last && Math.abs(last.time - t) < 1e-9) last.units += batch;
      else refills.push({ time: t, units: batch });
      out += batch * unitKg;
      left -= batch;
    }
    return { initial, refills, reserve: Math.max(0, left) };
  }

  function compute(service) {
    const sc = scenario(service);
    const present = presentGuests(service.meal, sc.presence);
    const gx = guestVector(service, sc);
    const rates = M.predictQuantiles(G, gx);
    const gq = rates.map((r) => Math.max(0, r * present));
    const guests = gq[1];
    const drivers = guestDrivers(gx, present, sc);
    const prof = demo.profiles[service.meal];
    const profile = { slots: prof.slots, shares: sc.match ? prof.match : prof.normal };
    const level = 1 - state.risk;

    const dishes = service.dishes.map((d) => {
      const info = demo.dishes[d.id];
      const x = dishVector(d, sc);
      const q = M.predictQuantiles(D, x).map((r) => Math.max(0, (r * guests) / 100));
      const recKg = interp(D.quantiles, q, level);
      const recUnits = roundHalfUp(recKg / info.kg_per_unit);
      const action = loadAction(service, d.id);
      const finalUnits = action && action.units ? action.units : recUnits;
      return {
        id: d.id, info, q, recKg, recUnits, finalUnits, action, isNew: !hasHistory(x),
        lowUnits: roundHalf(q[0] / info.kg_per_unit), highUnits: roundHalf(interp(D.quantiles, q, 0.9) / info.kg_per_unit),
        chefUnits: d.chef_plan_kg / info.kg_per_unit,
        reason: explain(x, info, guests),
        refill: refillPlan(finalUnits, q[1], info.kg_per_unit, profile),
      };
    });
    dishes.sort((a, b) => GROUP_ORDER.indexOf(a.info.group) - GROUP_ORDER.indexOf(b.info.group)
      || a.info.name.localeCompare(b.info.name, "tr"));
    return { sc, gq, guests, present, rate: rates[1], drivers, profile, dishes };
  }

  // ---------- rendering ----------
  function renderControls() {
    const chips = document.getElementById("day-chips");
    chips.innerHTML = "";
    demo.dates.forEach((iso, i) => {
      const d = parseDate(iso);
      const hasMatch = demo.services.some((s) => s.date === iso && s.event === "match");
      chips.appendChild(el("button", {
        type: "button", role: "radio", class: "chip", "aria-checked": String(i === state.day),
        onclick: () => { state.day = i; resetService(); render(); },
      }, [el("span", { class: "chip-day", text: DAYS[d.getDay()] }),
        el("span", { text: `${d.getDate()} ${MONTHS[d.getMonth()]}` }),
        hasMatch ? el("span", { class: "chip-flag", text: "maç" }) : null]));
    });

    const meals = document.getElementById("meal-seg");
    meals.innerHTML = "";
    MEALS.forEach((m) => meals.appendChild(el("button", {
      type: "button", role: "radio", class: "seg", "aria-checked": String(state.meal === m.id), text: m.label,
      onclick: () => { state.meal = m.id; state.clock = 0; resetService(); render(); },
    })));

    const risks = document.getElementById("risk-seg");
    risks.innerHTML = "";
    RISKS.forEach((r) => risks.appendChild(el("button", {
      type: "button", role: "radio", class: "seg", "aria-checked": String(state.risk === r), text: pct(r),
      onclick: () => { state.risk = r; render(); },
    })));

    document.getElementById("match-wrap").hidden = state.meal !== "dinner";
    document.getElementById("themed-wrap").hidden = state.meal !== "dinner";
    document.getElementById("occ-val").textContent = (state.occ > 0 ? "+" : "") + state.occ + " puan";
    document.getElementById("temp-val").textContent = (state.temp > 0 ? "+" : "") + state.temp + " °C";
  }

  function resetService() {
    state.match = null;
    state.themed = null;
    state.child = null;
    state.presence = {};
  }

  // Sliders are rebuilt only when the service changes, so dragging is never interrupted.
  function buildScenarioControls(service, sc) {
    const key = `${service.date}:${service.meal}:${state.occ}`;
    if (builtFor === key) return;
    builtFor = key;
    const box = document.getElementById("presence-controls");
    box.innerHTML = "";
    PRESENCE_CONTROLS[service.meal].forEach((c) => {
      const base = Math.round(sc.base[c.key]);
      const max = Math.max(150, Math.ceil((base * 3) / 10) * 10);
      const input = el("input", { type: "range", min: "0", max: String(max), step: "5",
        value: String(Math.round(sc.presence[c.key])), "data-key": c.key });
      input.addEventListener("input", (e) => { state.presence[c.key] = Number(e.target.value); render(); });
      const recorded = Math.round(service.presence[c.key]);
      box.appendChild(el("label", { class: "slider" }, [
        el("span", {}, [`${c.label} `, el("b", { "data-val": c.key }), el("span", { class: "base", text: ` (kayıtlı: ${recorded})` })]),
        input,
      ]));
    });
    // Child share only within the range seen in training; outside it the models are flat.
    const child = document.getElementById("child");
    const [lo, hi] = demo.support.child_share;
    child.min = String(Math.ceil(lo * 100));
    child.max = String(Math.floor(hi * 100));
    child.value = String(Math.round(sc.child * 100));
  }

  function updateScenarioLabels(sc) {
    document.querySelectorAll("#presence-controls [data-val]").forEach((b) => {
      b.textContent = `${nf0.format(sc.presence[b.dataset.val])} kişi`;
    });
    const [lo, hi] = demo.support.child_share;
    const outside = sc.child > hi + 1e-9 || sc.child < lo - 1e-9;
    document.getElementById("child-val").textContent = pct0(sc.child);
    document.getElementById("child-note").textContent = outside
      ? `(eğitim aralığı %${nf0.format(lo * 100)}–${nf0.format(hi * 100)} dışında; model bu aralığın dışında değişmez)` : "";
    // Effective occupancy change after the 100% cap, in points.
    const service = currentService();
    const occ = service.guest_x[gIdx.occupancy];
    const eff = Math.min(1, Math.max(0.05, occ + state.occ / 100)) - occ;
    document.getElementById("occ-val").textContent = (state.occ > 0 ? "+" : "") + state.occ + " puan" +
      (Math.abs(eff * 100 - state.occ) >= 0.5 ? ` (etkin: ${eff >= 0 ? "+" : "−"}${nf.format(Math.abs(eff * 100))} puan, en fazla %100)` : "");
  }

  function renderGuests(service, r) {
    document.getElementById("guest-number").textContent = nf0.format(r.guests);
    document.getElementById("guest-range").textContent =
      `Muhtemel aralık ${nf0.format(r.gq[0])}–${nf0.format(r.gq[2])} kişi`;
    // The presence rule, step by step, then the model's share of present guests who come.
    const p = r.sc.presence;
    const minus = {
      breakfast: [["departures_very_early", "kahvaltıdan önce ayrılıyor"]],
      lunch: [["excursions", "turda"], ["departures_early", "öğleden önce ayrılıyor"]],
      dinner: [["alacarte", "a la carte’ta"], ["arrivals_night", "gece geç geliyor"]],
    }[service.meal];
    // Shown with rounded counts that add up; the model uses the unrounded values.
    const shown = { in_house: Math.round(p.in_house) };
    const lines = [el("li", { text: `Otelde ${nf0.format(shown.in_house)} misafir` })];
    let presentShown = shown.in_house;
    minus.forEach(([k, label]) => {
      const v = Math.round(p[k]);
      if (v >= 1) lines.push(el("li", { text: `− ${nf0.format(v)} ${label}` }));
      presentShown -= v;
    });
    lines.push(el("li", { text: `= ${nf0.format(presentShown)} kişi öğünde otelde` }));
    lines.push(el("li", { class: "total", text: `× ${pct(r.rate)} büfeye gelir (model) ≈ ${nf0.format(r.guests)}` }));
    const list = document.getElementById("guest-breakdown");
    list.innerHTML = "";
    lines.forEach((li) => list.appendChild(li));
    document.getElementById("guest-drivers").textContent = r.drivers.length
      ? r.drivers.map((d) => `${d.label[0].toLocaleUpperCase("tr-TR") + d.label.slice(1)}: ${d.people > 0 ? "+" : "−"}${nf0.format(Math.abs(d.people))} kişi (olmasaydı ≈ ${nf0.format(d.without)})`).join(" · ")
      : "";

    const tag = document.getElementById("event-tag");
    let text = null;
    if (service.meal === "dinner" && r.sc.match) text = "Akşam maç var: yemek erkene kayar, maçtan sonra geç bir dalga gelir.";
    else if (service.meal === "dinner" && r.sc.themed) text = "Bu akşam tema gecesi.";
    else if (service.meal === "lunch" && service.event === "conference") text = "Kongre grubu öğle yemeğine katılıyor.";
    tag.hidden = !text;
    tag.textContent = text || "";

    const box = document.getElementById("flow-chart");
    const w = 560, h = 120, pad = 22;
    const n = r.profile.slots.length;
    const max = Math.max(...r.profile.shares);
    const bw = (w - pad) / n;
    let bars = "";
    r.profile.shares.forEach((s, i) => {
      const bh = (s / max) * (h - pad - 6);
      const x = pad / 2 + i * bw;
      bars += `<rect x="${x + 1}" y="${h - pad - bh}" width="${bw - 2}" height="${bh}" rx="2" fill="var(--clay)"><title>${fmtTime(r.profile.slots[i])}: ${nf0.format(s * r.guests)} kişi</title></rect>`;
      if (Math.abs(r.profile.slots[i] % 1) < 1e-6 || Math.abs((r.profile.slots[i] % 1) - 0.5) < 1e-6) {
        bars += `<text x="${x + bw / 2}" y="${h - 5}" text-anchor="middle" font-size="12" fill="var(--muted)">${fmtTime(r.profile.slots[i])}</text>`;
      }
    });
    box.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Servis boyunca 15 dakikalık dilimlerde varsayılan misafir gelişi">${bars}</svg>`;
  }

  function refillText(d) {
    const u = d.info.unit;
    const parts = [`İlk: ${unitText(d.refill.initial, u)}`];
    d.refill.refills.forEach((f) => parts.push(`+${nf.format(f.units)} ${fmtTime(f.time)}`));
    if (d.refill.reserve > 0) parts.push(`yedek ${nf.format(d.refill.reserve)} (gerekirse)`);
    return parts.join(" · ");
  }

  function renderDishes(service, r) {
    const grid = document.getElementById("dish-grid");
    grid.innerHTML = "";
    r.dishes.forEach((d) => {
      const u = d.info.unit;
      const status = d.action ? d.action.status : null;
      // An approved or adjusted amount stays fixed; say so when the live recommendation has moved since.
      const stale = d.action && d.action.status !== "soldout" && d.action.units !== d.recUnits;
      const statusText = {
        approved: stale ? `Onaylandı: ${unitText(d.finalUnits, u)}. Girdiler değişti, güncel öneri ${unitText(d.recUnits, u)}.` : "Onaylandı",
        adjusted: `Düzeltildi: ${unitText(d.finalUnits, u)} (güncel öneri ${unitText(d.recUnits, u)})`,
        soldout: `Bitti olarak kaydedildi${d.action && d.action.time ? " · " + d.action.time : ""}. Pilotta bu kayıt, o servisin talebini tükenme saatine göre yukarı düzeltmek için kullanılacak (bu demoda yalnızca tarayıcınızda tutulur).`,
      }[status] || "";

      const step = (delta) => () => {
        const units = Math.max(0.5, d.finalUnits + delta);
        saveAction(service, d.id, { status: units === d.recUnits ? "approved" : "adjusted", units });
        render();
      };
      const card = el("article", { class: `dish ${status ? "is-" + status : ""}` }, [
        el("p", { class: "dish-group" }, [GROUPS[d.info.group], d.isNew ? el("span", { class: "badge", text: "YENİ" }) : null]),
        el("h3", { class: "dish-name", text: d.info.name }),
        el("p", { class: "dish-qty" }, [el("b", { text: nf.format(d.finalUnits) }), ` ${u}`]),
        el("p", { class: "dish-kg muted", text: `≈ ${nf.format(d.finalUnits * d.info.kg_per_unit)} kg · beklenen talep genelde ${nf.format(d.lowUnits)}–${nf.format(d.highUnits)} ${u}` }),
        el("p", { class: "dish-refill", text: refillText(d) }),
        el("p", { class: "dish-reason", text: d.reason }),
        el("p", { class: "dish-chef note", text: `Klasik plan (simüle şef, kayıtlı gün; senaryodan etkilenmez): ${nf.format(d.chefUnits)} ${u}` }),
        el("div", { class: "dish-actions" }, [
          el("button", { type: "button", class: "act act-ok", "aria-pressed": String(status === "approved"),
            onclick: () => { saveAction(service, d.id, status === "approved" ? null : { status: "approved", units: d.recUnits }); render(); } }, "Onayla"),
          el("div", { class: "stepper", role: "group", "aria-label": "Miktarı düzelt" }, [
            el("button", { type: "button", class: "act", "aria-label": "Yarım birim azalt", onclick: step(-0.5) }, "−"),
            el("span", { class: "stepper-val", text: nf.format(d.finalUnits) }),
            el("button", { type: "button", class: "act", "aria-label": "Yarım birim artır", onclick: step(0.5) }, "+"),
          ]),
          el("button", { type: "button", class: "act act-out", "aria-pressed": String(status === "soldout"),
            onclick: () => {
              const time = state.tab === "live" ? fmtTime(r.profile.slots[state.clock]) : null;
              saveAction(service, d.id, status === "soldout" ? null : { status: "soldout", units: d.finalUnits, time });
              render();
            } }, "Bitti"),
        ]),
        el("p", { class: "dish-status", text: statusText }),
      ]);
      grid.appendChild(card);
    });
  }

  function renderLive(service, r) {
    const slider = document.getElementById("clock");
    slider.max = String(r.profile.slots.length - 1);
    if (state.clock > r.profile.slots.length - 1) state.clock = 0;
    slider.value = String(state.clock);
    const now = r.profile.slots[state.clock];
    // Guests who arrived before this time (slots that have already ended).
    const arrived = r.profile.shares.slice(0, state.clock).reduce((a, b) => a + b, 0);
    document.getElementById("clock-val").textContent =
      `${fmtTime(now)} · bu saate kadar gelen ≈ ${nf0.format(arrived * r.guests)} kişi (${pct0(arrived)}, varsayılan profile göre)`;

    const nowList = document.getElementById("refill-now");
    const nextList = document.getElementById("refill-next");
    nowList.innerHTML = "";
    nextList.innerHTML = "";
    const due = [];
    const next = [];
    r.dishes.forEach((d) => {
      if (d.action && d.action.status === "soldout") return;
      d.refill.refills.forEach((f) => {
        if (f.time <= now + 1e-9 && f.time > now - 0.25 - 1e-9) due.push({ d, f });
        else if (f.time > now) next.push({ d, f });
      });
    });
    next.sort((a, b) => a.f.time - b.f.time);
    if (!due.length) nowList.appendChild(el("li", { class: "live-empty", text: "Şu an tazelenecek tepsi yok." }));
    due.forEach(({ d, f }) => nowList.appendChild(el("li", { class: "live-item" }, [
      el("span", { class: "live-qty", text: `+${nf.format(f.units)} ${d.info.unit}` }),
      el("span", { text: d.info.name }),
      el("button", { type: "button", class: "act act-out", onclick: () => {
        saveAction(service, d.id, { status: "soldout", units: d.finalUnits, time: fmtTime(now) });
        render();
      } }, "Bitti"),
    ])));
    next.slice(0, 8).forEach(({ d, f }) => nextList.appendChild(el("li", { class: "live-item" }, [
      el("span", { class: "live-time", text: fmtTime(f.time) }),
      el("span", { text: `${d.info.name} +${nf.format(f.units)} ${d.info.unit}` }),
    ])));
    if (!next.length) nextList.appendChild(el("li", { class: "live-empty", text: "Bu servis için başka tazeleme planlanmadı." }));
    const sold = r.dishes.filter((d) => d.action && d.action.status === "soldout");
    if (sold.length) {
      nowList.appendChild(el("li", { class: "live-sold", text:
        `Bitti olarak kaydedilen: ${sold.map((d) => d.info.name + (d.action.time ? " (" + d.action.time + ")" : "")).join(", ")}` }));
    }
  }

  function renderReport() {
    const rep = demo.metrics.monthly_report;
    const lvl = rep.levels.find((l) => Math.abs(l.risk - state.risk) < 1e-9);
    const box = document.getElementById("report");
    box.innerHTML = "";
    const avoided = lvl.avoided_kg;
    const positive = avoided > 0;
    box.appendChild(el("p", { class: "report-head" }, [
      el("b", { text: rep.month }), ` · risk ayarı ${pct0(state.risk)} · simüle şef planıyla karşılaştırma`]));
    const tiles = el("div", { class: "report-tiles" });
    const tile = (label, value, note) => el("div", { class: "report-tile" }, [
      el("p", { class: "control-label", text: label }), el("p", { class: "report-value", text: value }),
      note ? el("p", { class: "note", text: note }) : null]);
    if (positive) {
      tiles.appendChild(tile("Önlenen atık", `${nf0.format(avoided)} kg`));
      tiles.appendChild(tile("Önlenen atığın maliyeti", `${nf0.format(lvl.avoided_try)} TL`,
        "Yemek grubuna göre kg maliyeti varsayımı; tükenmenin maliyeti dahil değil"));
      tiles.appendChild(tile("Önlenen CO₂ eşdeğeri", `${nf0.format(lvl.avoided_co2_kg)} kg`, "Önlenen atık × 1,9"));
    } else {
      tiles.appendChild(tile("Ek atık", `${nf0.format(-avoided)} kg`, "Bu kadar güvenli bir ayarda model şeften fazla pişirir ve fazla atar."));
    }
    tiles.appendChild(tile("Talebin planı aştığı yemekler", `${pct(lvl.plan_short_rate)}`,
      `Hedeflenen: %${nf0.format(state.risk * 100)}. Simüle şef: ${pct(rep.chef.plan_short_rate)}`));
    tiles.appendChild(tile("Tükenen yemek oranı", `${pct(lvl.stockout_rate)}`,
      `Servis edilen yemeklerin tükenen payı. Simüle şef: ${pct(rep.chef.stockout_rate)}`));
    tiles.appendChild(tile("En az bir yemeği biten öğün", `${pct0(lvl.services_with_stockout)}`,
      `Simüle şef: ${pct0(rep.chef.services_with_stockout)}`));
    box.appendChild(tiles);
    const fewerOut = lvl.services_with_stockout < rep.chef.services_with_stockout;
    const lessWaste = positive;
    let tradeoff;
    if (lessWaste && fewerOut) tradeoff = "Bu ayarda model hem daha az atık üretiyor hem de şeften daha az öğünde yemek bitiyor.";
    else if (lessWaste) tradeoff = "Bu ayarda atık azalıyor ama şeften daha fazla öğünde en az bir yemek bitiyor: riski düşürmek bu dengeyi değiştirir.";
    else tradeoff = "Bu ayarda yemek daha az bitiyor ama atık şeften fazla: riski yükseltmek bu dengeyi değiştirir.";
    box.appendChild(el("p", { class: "report-tradeoff", text: tradeoff }));
    box.appendChild(el("p", { class: "note", text:
      `Risk ayarı, talebin önerilen toplam miktarı aşması için hedeflenen olasılıktır; modelin aralıkları henüz kalibre olmadığı için gerçekleşen oran biraz daha yüksek (yukarıdaki ilk kutu). Rapor şu varsayımla hesaplandı: önerilen miktarın tamamı servis boyunca hazırlanır; talep bunu aşarsa mutfak, planın en fazla %${nf0.format(demo.max_refill_share * 100)}'si kadar ek pişirme yapar (ek pişirme, gerektiği anda ve gerektiği kadar). Aynı kural simüle şefe de uygulandı. Ekrandaki parti zamanlaması ve “yedek (gerekirse)” kısmı bu hesaba katılmadı; yedeğin yalnızca gerektiğinde pişirilmesi atığı burada gösterilenden düşük tutabilir, zamanında hazırlanamazsa tükenmeyi artırabilir. Sentetik test sezonunda, modelin görmediği günlerde hesaplandı. Ölçüm sonucu değildir; gerçek rakam ancak pilot otelde ölçülebilir.` }));
  }

  function renderTabs() {
    for (const t of ["plan", "live", "report"]) {
      document.getElementById(`tab-${t}`).setAttribute("aria-selected", String(state.tab === t));
      document.getElementById(`panel-${t}`).hidden = state.tab !== t;
    }
  }

  function render() {
    const service = currentService();
    renderControls();
    const r = compute(service);
    buildScenarioControls(service, r.sc);
    updateScenarioLabels(r.sc);
    document.getElementById("match").checked = r.sc.match;
    document.getElementById("themed").checked = r.sc.themed;
    renderGuests(service, r);
    renderDishes(service, r);
    renderLive(service, r);
    renderReport();
    renderTabs();
  }

  // ---------- boot ----------
  async function boot() {
    try {
      const [d, g, m] = await Promise.all(["data/demo.json", "data/model-guests.json", "data/model-dishes.json"]
        .map((u) => fetch(u).then((res) => { if (!res.ok) throw new Error(u); return res.json(); })));
      demo = d; G = g; D = m;
    } catch (e) {
      document.getElementById("loading").hidden = true;
      document.getElementById("load-error").hidden = false;
      return;
    }
    gIdx = Object.fromEntries(G.features.map((f, i) => [f, i]));
    dIdx = Object.fromEntries(D.features.map((f, i) => [f, i]));
    demo.dates = [...new Set(demo.services.map((s) => s.date))].sort();
    const matchDay = demo.dates.findIndex((iso) => demo.services.some((s) => s.date === iso && s.event === "match"));
    state.day = matchDay >= 0 ? matchDay : 0;
    document.getElementById("record").textContent = demo.record;
    // Optional deep link, e.g. demo.html?tab=live&clock=4
    const params = new URLSearchParams(location.search);
    if (["plan", "live", "report"].includes(params.get("tab"))) state.tab = params.get("tab");
    if (params.get("clock")) state.clock = Number(params.get("clock")) || 0;
    if (params.get("whatif")) document.getElementById("whatif").open = true;
    if (["breakfast", "lunch", "dinner"].includes(params.get("meal"))) state.meal = params.get("meal");

    document.getElementById("occ").addEventListener("input", (e) => { state.occ = Number(e.target.value); render(); });
    document.getElementById("temp").addEventListener("input", (e) => { state.temp = Number(e.target.value); render(); });
    document.getElementById("match").addEventListener("change", (e) => {
      state.match = e.target.checked;
      if (state.match) state.themed = false;
      render();
    });
    document.getElementById("themed").addEventListener("change", (e) => {
      state.themed = e.target.checked;
      if (state.themed) state.match = false;
      render();
    });
    document.getElementById("child").addEventListener("input", (e) => { state.child = Number(e.target.value) / 100; render(); });
    document.getElementById("clock").addEventListener("input", (e) => { state.clock = Number(e.target.value); render(); });
    document.getElementById("reset").addEventListener("click", () => {
      state.occ = 0; state.temp = 0;
      resetService();
      builtFor = null;
      document.getElementById("occ").value = "0";
      document.getElementById("temp").value = "0";
      render();
    });
    for (const t of ["plan", "live", "report"]) {
      document.getElementById(`tab-${t}`).addEventListener("click", () => { state.tab = t; renderTabs(); });
    }

    document.getElementById("loading").hidden = true;
    document.getElementById("app").hidden = false;
    render();
  }

  boot();
})();
