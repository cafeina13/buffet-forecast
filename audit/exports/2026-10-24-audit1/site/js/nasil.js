// Charts for the "how it works" page, drawn from data/demo.json (written by olculu.train).
(function () {
  "use strict";
  const nf = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 1 });
  const nf0 = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 0 });
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function riskChart(m) {
    const pts = m.risk_curve.map((p) => ({ x: p.stockout_rate * 100, y: p.waste_kg / 1000, r: p.risk }));
    const chef = { x: m.chef.stockout_rate * 100, y: m.chef.waste_kg / 1000 };
    const W = 760, H = 420, L = 70, R = 24, T = 20, B = 60;
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
    g += `<text x="${(L + W - R) / 2}" y="${H - 12}" text-anchor="middle" font-size="14" fill="#1e3a33">Tükenen servis oranı</text>`;
    g += `<text x="18" y="${(T + H - B) / 2}" text-anchor="middle" font-size="14" fill="#1e3a33" transform="rotate(-90 18 ${(T + H - B) / 2})">Toplam atık (ton)</text>`;
    g += `<polyline fill="none" stroke="#1e3a33" stroke-width="3" points="${pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}"/>`;
    pts.forEach((p) => {
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="7" fill="#1e3a33"><title>Risk %${nf0.format(p.r * 100)}: ${nf.format(p.y)} t atık, %${nf.format(p.x)} tükenme</title></circle>`;
      g += `<text x="${sx(p.x) + 10}" y="${sy(p.y) - 10}" font-size="13" fill="#1e3a33">%${nf0.format(p.r * 100)}</text>`;
    });
    g += `<circle cx="${sx(chef.x)}" cy="${sy(chef.y)}" r="10" fill="#a94e27"><title>Simüle şef: ${nf.format(chef.y)} t atık, %${nf.format(chef.x)} tükenme</title></circle>`;
    g += `<text x="${sx(chef.x) + 14}" y="${sy(chef.y) + 5}" font-size="14" font-weight="700" fill="#a94e27">Klasik plan</text>`;
    document.getElementById("risk-chart").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}">${g}</svg><div class="legend"><span class="lg-model">Model, bitme riski ayarına göre (etiket)</span><span class="lg-chef">Simüle şefin klasik planı</span></div>`;

    const atChef = pts.filter((p) => p.x <= chef.x).sort((a, b) => a.y - b.y)[0];
    if (atChef) {
      const saving = 1 - atChef.y / chef.y;
      document.getElementById("risk-note").textContent =
        `Sentetik test sezonunda, şefin tükenme oranını aşmayan ilk ayarda (%${nf0.format(atChef.r * 100)} risk) model %${nf0.format(saving * 100)} daha az atık üretti. Bu fark sentetik verinin varsayımlarından gelir; ölçüm sonucu değildir.`;
    }
  }

  function guestCards(m) {
    const g = m.guest;
    const box = document.getElementById("guest-cards");
    box.innerHTML = `
      <div class="card"><p class="step">YALNIZ DOLULUK</p><p class="stat">±${nf0.format(g.mae_occupancy_only)}</p><p>kişi ortalama hata (öğün başına)</p></div>
      <div class="card"><p class="step">ÖN BÜRO KURALI + MODEL (MAÇ, TEMA GECESİ, HAVA)</p><p class="stat">±${nf0.format(g.mae_full)}</p><p>kişi ortalama hata (öğün başına)</p></div>`;
  }

  function mapChart(d) {
    const pts = d.map;
    const W = 760, H = 520, P = 40;
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sx = (x) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P);
    const sy = (y) => H - P - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * P);
    const labelled = new Set();
    Object.values(d.dishes).forEach((dish) => {
      if (dish.new) {
        labelled.add(dish.name);
        dish.neighbours.slice(0, 2).forEach((n) => labelled.add(n));
      }
    });
    let g = "";
    pts.forEach((p) => {
      const dish = d.dishes[p.id];
      const fill = dish.new ? "#a94e27" : "#7a9a8c";
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="${dish.new ? 8 : 6}" fill="${fill}"><title>${esc(dish.name)}: ${esc(dish.ingredients.join(", "))}</title></circle>`;
      if (labelled.has(dish.name)) {
        g += `<text x="${sx(p.x) + 10}" y="${sy(p.y) + 4}" font-size="13" fill="${dish.new ? "#a94e27" : "#1e3a33"}" font-weight="${dish.new ? 700 : 400}">${esc(dish.name)}</text>`;
      }
    });
    document.getElementById("map-chart").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}">${g}</svg><div class="legend"><span class="lg-new">Menüye sezon ortasında giren yemek</span><span class="lg-dish">Katalogdaki diğer yemekler (üzerine gelince adı görünür)</span></div>`;
  }

  fetch("data/demo.json").then((r) => r.json()).then((d) => {
    document.getElementById("record").textContent = d.record;
    riskChart(d.metrics);
    guestCards(d.metrics);
    mapChart(d);
  }).catch(() => {
    document.getElementById("risk-chart").textContent = "Grafik yüklenemedi.";
  });
})();
