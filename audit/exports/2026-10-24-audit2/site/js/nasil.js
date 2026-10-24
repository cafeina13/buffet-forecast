// Charts for the "how it works" page, drawn from data/demo.json (written by olculu.train).
(function () {
  "use strict";
  const nf = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 1 });
  const nf0 = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 0 });
  const pct = (x) => "%" + nf.format(x * 100);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function riskChart(m) {
    const pts = m.risk_curve.map((p) => ({ x: p.stockout_rate * 100, y: p.waste_kg / 1000, r: p.risk }));
    const chef = { x: m.chef.stockout_rate * 100, y: m.chef.waste_kg / 1000 };
    const W = 760, H = 420, L = 70, R = 48, T = 20, B = 60;
    const xmax = Math.ceil(Math.max(...pts.map((p) => p.x), chef.x) / 5) * 5;
    const ymax = Math.ceil(Math.max(...pts.map((p) => p.y), chef.y) / 20) * 20;
    const sx = (x) => L + (x / xmax) * (W - L - R);
    const sy = (y) => H - B - (y / ymax) * (H - T - B);
    let g = "";
    for (let x = 0; x <= xmax; x += 5) {
      g += `<line x1="${sx(x)}" x2="${sx(x)}" y1="${T}" y2="${H - B}" stroke="#e2dacb"/>`;
      g += `<text x="${sx(x)}" y="${H - B + 22}" text-anchor="middle" font-size="14" fill="#4a5752">%${x}</text>`;
    }
    for (let y = 0; y <= ymax; y += 20) {
      g += `<line x1="${L}" x2="${W - R}" y1="${sy(y)}" y2="${sy(y)}" stroke="#e2dacb"/>`;
      g += `<text x="${L - 10}" y="${sy(y) + 5}" text-anchor="end" font-size="14" fill="#4a5752">${y} t</text>`;
    }
    g += `<text x="${(L + W - R) / 2}" y="${H - 12}" text-anchor="middle" font-size="14" fill="#1e3a33">Servis edilen yemeklerin tükenen payı (tazeleme sonrası)</text>`;
    g += `<text x="18" y="${(T + H - B) / 2}" text-anchor="middle" font-size="14" fill="#1e3a33" transform="rotate(-90 18 ${(T + H - B) / 2})">Toplam atık (ton)</text>`;
    g += `<polyline fill="none" stroke="#1e3a33" stroke-width="3" points="${pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}"/>`;
    pts.forEach((p) => {
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="7" fill="#1e3a33"><title>Risk ayarı %${nf0.format(p.r * 100)}: ${nf.format(p.y)} t atık, tükenen yemek payı %${nf.format(p.x)}</title></circle>`;
      g += `<text x="${sx(p.x) + 10}" y="${sy(p.y) - 10}" font-size="13" fill="#1e3a33">%${nf0.format(p.r * 100)}</text>`;
    });
    g += `<circle cx="${sx(chef.x)}" cy="${sy(chef.y)}" r="10" fill="#a94e27"><title>Simüle şef: ${nf.format(chef.y)} t atık, tükenen yemek payı %${nf.format(chef.x)}</title></circle>`;
    g += `<text x="${sx(chef.x) + 14}" y="${sy(chef.y) + 5}" font-size="14" font-weight="700" fill="#a94e27">Klasik plan</text>`;
    document.getElementById("risk-chart").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}">${g}</svg><div class="legend"><span class="lg-model">Model; etiket: risk ayarı (talebin ilk pişirilen miktarı aşma olasılığı)</span><span class="lg-chef">Simüle şefin klasik planı</span></div>`;

    // The same evaluation in a table, with the meal-level view: a meal counts if any dish ran out.
    const rows = [["Simüle şef", m.chef]].concat(
      m.risk_curve.filter((c) => [0.2, 0.1, 0.05].includes(c.risk)).map((c) => [`Model, %${nf0.format(c.risk * 100)} risk`, c]));
    document.getElementById("risk-table").innerHTML = `<table class="data-table">
      <thead><tr><th></th><th>Toplam atık</th><th>Talebin ilk pişirimi aştığı yemekler</th><th>Tazelemeye rağmen tükenen yemekler</th><th>En az bir yemeği biten öğünler</th></tr></thead>
      <tbody>${rows.map(([name, c]) => `<tr><th>${name}</th><td>${nf.format(c.waste_kg / 1000)} t</td><td>${pct(c.plan_short_rate)}</td><td>${pct(c.stockout_rate)}</td><td>${pct(c.services_with_stockout)}</td></tr>`).join("")}</tbody>
    </table>`;

    const c20 = m.risk_curve.find((c) => c.risk === 0.2);
    const c10 = m.risk_curve.find((c) => c.risk === 0.1);
    document.getElementById("risk-note").textContent =
      `Denge gerçek: %20 ayarında model şeften %${nf0.format((1 - c20.waste_kg / m.chef.waste_kg) * 100)} daha az atık üretiyor ama en az bir yemeği biten öğün payı daha yüksek (${pct(c20.services_with_stockout)} / şef ${pct(m.chef.services_with_stockout)}). %10 ayarında atık şefe yakın (%${nf0.format((1 - c10.waste_kg / m.chef.waste_kg) * 100)} daha az) ve yemek bitmesi belirgin biçimde daha az (en az bir yemeği biten öğün payı ${pct(c10.services_with_stockout)}). Bu rakamlar sentetik verinin varsayımlarından gelir; ölçüm sonucu değildir.`;
  }

  function guestCards(m) {
    const g = m.guest;
    const card = (step, value, text) =>
      `<div class="card"><p class="step">${step}</p><p class="stat">${nf.format(value)}</p><p>${text}</p></div>`;
    document.getElementById("guest-cards").innerHTML =
      card("EVDEKİ MİSAFİR × ÖĞÜNÜN ORTALAMA ORANI", g.mae_in_house_times_meal_rate, "kişi ortalama hata; en basit kural") +
      card("ÖN BÜRO KURALI × ÖĞÜNÜN ORTALAMA ORANI", g.mae_presence_rule_times_meal_rate, "kişi ortalama hata; model yok") +
      card("ÖN BÜRO KURALI + MODEL (MAÇ, TEMA GECESİ)", g.mae_full, "kişi ortalama hata; şefin ekranındaki tahmin");
    document.getElementById("occ-only").textContent = nf.format(g.mae_occupancy_only);
  }

  function neighbourList(d) {
    const items = Object.values(d.dishes).filter((dish) => dish.new).map((dish) =>
      `<li><b>${esc(dish.name)}</b> <span class="muted">(${esc(dish.ingredients.join(", "))})</span><br>→ ${dish.neighbours.map(esc).join(" · ")}</li>`);
    document.getElementById("map-chart").innerHTML = `<ul class="neighbour-list">${items.join("")}</ul>`;
  }

  fetch("data/demo.json").then((r) => r.json()).then((d) => {
    document.getElementById("record").textContent = d.record;
    riskChart(d.metrics);
    guestCards(d.metrics);
    neighbourList(d);
  }).catch(() => {
    document.getElementById("risk-chart").textContent = "Grafik yüklenemedi.";
  });
})();
