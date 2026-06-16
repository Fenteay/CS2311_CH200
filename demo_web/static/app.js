/* ── app.js — TT-SFUDA Web Demo ── */

let adaptLossChart  = null;
let adaptationChart = null;
let flHistoryChart  = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupFilePicker("predictFile",  "predictFileName");
  setupFilePicker("adaptFile",    "adaptFileName");

  await Promise.all([
    loadModels(),
    loadMetrics(),
    loadGalleryTargets(),
  ]);
});

// ── File picker label ─────────────────────────────────────────────────────────
function setupFilePicker(inputId, labelId) {
  const inp = document.getElementById(inputId);
  const lbl = document.getElementById(labelId);
  inp.addEventListener("change", () => {
    lbl.textContent = inp.files[0]?.name || "Choose image…";
    if (inputId === "adaptFile" && inp.files[0]) {
      const url = URL.createObjectURL(inp.files[0]);
      ["adaptOrigImg", "adaptOrigImg2"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.src = url;
      });
    }
    if (inputId === "predictFile" && inp.files[0]) {
      const url = URL.createObjectURL(inp.files[0]);
      const el = document.getElementById("origImg");
      if (el) el.src = url;
    }
  });
}

// ── Gallery targets ──────────────────────────────────────────────────────────
async function loadGalleryTargets() {
  try {
    const res     = await fetch("/api/gallery_targets");
    const targets = await res.json();
    const sel     = document.getElementById("galleryTarget");
    if (!sel) return;
    sel.innerHTML = "";
    targets.forEach(t => {
      const opt = document.createElement("option");
      opt.value       = t.value;
      opt.textContent = t.label;
      sel.appendChild(opt);
    });
    // Auto-load first (best) dataset
    await loadGallery();
  } catch(e) { console.error("loadGalleryTargets:", e); }
}

// ── Gallery ────────────────────────────────────────────────────────────────────
async function loadGallery() {
  const target  = document.getElementById("galleryTarget")?.value || "leafandmask_full";
  const loader  = document.getElementById("galleryLoader");
  const grid    = document.getElementById("galleryGrid");
  const stats   = document.getElementById("galleryStats");
  const metricsBox = document.getElementById("galleryMetrics");
  if (!grid) return;

  loader.classList.add("active");
  grid.innerHTML = "";
  stats.style.display = "none";
  if (metricsBox) { metricsBox.style.display = "none"; metricsBox.textContent = ""; }

  try {
    const res  = await fetch(`/api/test_gallery?target=${target}`);
    const data = await res.json();
    if (data.error) { grid.innerHTML = `<pre style="color:var(--red)">${data.error}</pre>`; return; }

    // Metrics file content
    if (data.metrics_txt && metricsBox) {
      metricsBox.textContent = data.metrics_txt;
      metricsBox.style.display = "block";
    }

    const items = data.items || [];
    const dices = items.map(i => i.dice).filter(d => d !== null && d !== undefined);
    const avgDice   = dices.length ? (dices.reduce((a,b) => a+b, 0) / dices.length) : null;
    const nearZero  = items.filter(i => i.dice !== null && i.dice < 0.05).length;
    const goodCount = items.filter(i => i.dice !== null && i.dice >= 0.5).length;

    stats.innerHTML = `
      <div class="gallery-stat">
        <span class="s-label">Total images</span>
        <span class="s-value">${items.length}</span>
      </div>
      <div class="gallery-stat">
        <span class="s-label">Avg Dice</span>
        <span class="s-value" style="color:var(--yellow)">${avgDice !== null ? avgDice.toFixed(4) : "—"}</span>
      </div>
      <div class="gallery-stat">
        <span class="s-label">Dice ≥ 0.5 (good)</span>
        <span class="s-value" style="color:var(--green)">${goodCount}</span>
      </div>
      <div class="gallery-stat">
        <span class="s-label">Dice &lt; 0.05 (near-zero)</span>
        <span class="s-value" style="color:var(--red)">${nearZero}</span>
      </div>
    `;
    stats.style.display = "flex";

    items.forEach(item => {
      const dice      = item.dice;
      const diceClass = dice === null ? "" : dice >= 0.5 ? "good" : dice >= 0.2 ? "mid" : "bad";
      const diceTxt   = dice !== null ? `Dice: ${dice.toFixed(3)}` : "no pred";

      const card = document.createElement("div");
      card.className = "gallery-card";

      // Compare image row (full-width) if available
      const compareHtml = item.compare_b64 ? `
        <div class="compare-row">
          <span class="compare-label">Input | GT (blue=TP, white=FP) | Overlay</span>
          <img src="data:image/png;base64,${item.compare_b64}" alt="compare ${item.id}">
        </div>` : "";

      card.innerHTML = `
        <div class="gallery-card-header">
          <span class="gallery-id">${item.id}</span>
          <span class="dice-badge ${diceClass}">${diceTxt}</span>
        </div>
        <div class="gallery-imgs">
          <div>
            <span>Input</span>
            <img src="data:image/png;base64,${item.img_b64}" alt="${item.id}">
          </div>
          <div>
            <span>GT Mask</span>
            <img src="${item.gt_b64 ? 'data:image/png;base64,' + item.gt_b64 : ''}" alt="gt"
                 style="${!item.gt_b64 ? 'opacity:.3' : ''}">
          </div>
          <div>
            <span>Predicted</span>
            <img src="${item.pred_b64 ? 'data:image/png;base64,' + item.pred_b64 : ''}" alt="pred"
                 style="${!item.pred_b64 ? 'opacity:.3' : ''}">
          </div>
        </div>
        ${compareHtml}
      `;
      grid.appendChild(card);
    });

  } catch(e) {
    grid.innerHTML = `<pre style="color:var(--red)">${e}</pre>`;
  } finally {
    loader.classList.remove("active");
  }
}

// ── Load model list ───────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const models = await res.json();

    // Only show best models
    const PINNED = [
      { name: "leafandmask_full_unet", label: "Supervised  (Dice=0.5879)" },
      { name: "fl_warm_r5_e2",         label: "FL warm 5r  (Dice=0.5476)" },
      { name: "fl_hot_r10_e2",         label: "FL hot 10r  (Adapted=0.5539)" },
    ];

    ["predictSource", "adaptSource"].forEach(selId => {
      const sel = document.getElementById(selId);
      sel.innerHTML = "";
      PINNED.forEach(p => {
        const exists = models.find(m => m.name === p.name);
        if (!exists) return;
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.label;
        sel.appendChild(opt);
      });
    });
    // Separate defaults: Predict → Supervised, Adaptation → best FL
    document.getElementById("predictSource").value = "leafandmask_full_unet";
    document.getElementById("adaptSource").value   = "fl_hot_r10_e2";

    // Build model table
    const wrap = document.getElementById("modelTable");
    if (!wrap) return;
    let html = `<table><thead><tr>
      <th>Model</th><th>Type</th><th>Target configs</th>
    </tr></thead><tbody>`;
    models.forEach(m => {
      const cfgTags = (m.configs || []).map(c => `<span class="tag">${c}</span>`).join(" ");
      const isSup = m.type === "Supervised";
      const typeColor = isSup ? "var(--green)" : "var(--yellow)";
      html += `<tr>
        <td style="${isSup ? 'color:var(--green)' : ''}">${m.name}</td>
        <td><span style="color:${typeColor};font-weight:600">${m.type}</span></td>
        <td>${cfgTags || "—"}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    wrap.innerHTML = html;
  } catch(e) {
    console.error("loadModels:", e);
  }
}

// ── Load FL + adaptation metrics ─────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch("/api/fl_metrics");
    const data = await res.json();

    document.getElementById("deviceBadge").textContent = "CPU / CUDA (auto)";

    const exp  = data.experiment_results || {};
    const best = exp.best_adaptation    || {};
    const sup  = exp.supervised         || {};

    setCard("cv-rounds",    best.model ? "10" : "—");
    setCard("cv-src-dice",  best.source_only_dice?.toFixed(4) ?? "—");
    setCard("cv-ada-dice",  best.adapted_dice?.toFixed(4)     ?? "—");
    setCard("cv-train-iou", best.train_iou?.toFixed(4)        ?? "—");
    setCard("cv-refine-iou",best.refine_iou?.toFixed(4)       ?? "—");

    // Adaptation bar chart — pass best + supervised baseline
    drawAdaptationChart(best, sup);

    // FL history line chart — pass rounds array directly
    const rounds = exp.fl_warm_r5?.rounds;
    if (rounds?.length) drawFlHistoryChart(rounds);

  } catch(e) {
    console.error("loadMetrics:", e);
  }
}

function setCard(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function drawFlHistoryChart(rounds) {
  const ctx = document.getElementById("flHistoryChart");
  if (!ctx) return;
  if (flHistoryChart) flHistoryChart.destroy();

  const labels   = rounds.map(r => `Round ${r.round}`);
  const diceData = rounds.map(r => parseFloat(r.val_dice.toFixed(4)));
  const iouData  = rounds.map(r => parseFloat(r.val_iou.toFixed(4)));

  flHistoryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Val Dice",
          data: diceData,
          borderColor: "rgba(52,211,153,0.9)",
          backgroundColor: "rgba(52,211,153,0.12)",
          borderWidth: 2, pointRadius: 5, fill: true, tension: 0.3,
          yAxisID: "y",
        },
        {
          label: "Val IoU",
          data: iouData,
          borderColor: "rgba(79,156,249,0.85)",
          backgroundColor: "rgba(79,156,249,0.08)",
          borderWidth: 2, pointRadius: 5, fill: true, tension: 0.3,
          yAxisID: "y",
        },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" } },
        y: { min: 0.3, max: 0.7, ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" },
             title: { display: true, text: "Score", color: "#94a3b8" } },
      }
    }
  });
}

function drawAdaptationChart(best, sup) {
  const ctx = document.getElementById("adaptationChart");
  if (!ctx) return;
  if (adaptationChart) adaptationChart.destroy();

  const supDice = sup?.source_only_dice ?? 0;
  const labels = ["Supervised Baseline", "FL Source-Only", "FL Adapted"];
  const values = [
    supDice,
    best.source_only_dice ?? 0,
    best.adapted_dice     ?? 0,
  ];
  const colors = [
    "rgba(251,191,36,0.8)",
    "rgba(79,156,249,0.8)",
    "rgba(52,211,153,0.8)",
  ];

  adaptationChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Dice Score",
        data: values,
        backgroundColor: colors,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.raw.toFixed(4)}` }
        }
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" } },
        y: { min: 0.4, max: 0.7, ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" } }
      }
    }
  });
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function runPredict() {
  const fileInp  = document.getElementById("predictFile");
  const source   = document.getElementById("predictSource").value;
  const btn      = document.getElementById("btnPredict");
  const loader   = document.getElementById("predictLoader");
  const errBox   = document.getElementById("predictError");
  const stats    = document.getElementById("predictStats");
  const panels   = document.getElementById("predictResults");

  if (!fileInp.files[0]) { showError(errBox, "Please select an image file."); return; }

  clearError(errBox);
  btn.disabled = true;
  loader.classList.add("active");
  panels.style.display = "none";

  const fd = new FormData();
  fd.append("image",  fileInp.files[0]);
  fd.append("source", source);

  try {
    const t0  = Date.now();
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    if (data.error) { showError(errBox, data.error); return; }

    setImgB64("maskImg",    data.mask_b64);
    setImgB64("overlayImg", data.overlay_b64);
    setImgB64("heatmapImg", data.heatmap_b64);

    stats.innerHTML = `
      <b>Model:</b> ${data.source}&nbsp;&nbsp;
      <b>Inference time:</b> ${elapsed}s
    `;
    panels.style.display = "flex";

    // Disease analysis card
    const pct = parseFloat(data.fg_ratio);
    const card     = document.getElementById("diseaseCard");
    const verdict  = document.getElementById("diseaseVerdict");
    const barFill  = document.getElementById("diseaseBarFill");
    const diseasePct = document.getElementById("diseasePct");
    const note     = document.getElementById("diseaseNote");
    card.style.display = "block";

    let level, color, emoji, noteText;
    if (pct < 1) {
      level = "Khỏe mạnh"; color = "#22c55e"; emoji = "🟢";
      noteText = "Không phát hiện vùng bệnh trên lá.";
    } else if (pct < 5) {
      level = "Bệnh nhẹ"; color = "#a3e635"; emoji = "🟡";
      noteText = "Phát hiện vùng bệnh nhỏ, có thể theo dõi thêm.";
    } else if (pct < 15) {
      level = "Bệnh trung bình"; color = "#f59e0b"; emoji = "🟠";
      noteText = "Vùng bệnh đáng kể, nên xử lý sớm.";
    } else if (pct < 40) {
      level = "Bệnh nặng"; color = "#f97316"; emoji = "🟠";
      noteText = "Lá bị bệnh nhiều, cần can thiệp kịp thời.";
    } else {
      level = "Rất nặng"; color = "#ef4444"; emoji = "🔴";
      noteText = "Hơn 40% diện tích lá bị ảnh hưởng.";
    }

    verdict.innerHTML = `<span style="font-size:1.5rem">${emoji}</span> <span style="color:${color};font-weight:700;font-size:1.2rem">${level}</span>`;
    barFill.style.width = Math.min(pct, 100) + "%";
    barFill.style.background = color;
    diseasePct.textContent = pct.toFixed(1) + "%";
    diseasePct.style.color = color;
    note.innerHTML = noteText + `<br><span style="color:#64748b;font-size:0.78rem">⚠ Lưu ý: model chỉ được train trên <b>lá bệnh</b> — không thể phân biệt lá khỏe hoàn toàn. Kết quả chỉ mang tính tham khảo.</span>`;

  } catch(e) {
    showError(errBox, e.toString());
  } finally {
    btn.disabled = false;
    loader.classList.remove("active");
  }
}

// ── Adaptation demo ───────────────────────────────────────────────────────────
async function runAdapt() {
  const fileInp   = document.getElementById("adaptFile");
  const source    = document.getElementById("adaptSource").value;
  const steps     = document.getElementById("adaptSteps").value;
  const btn       = document.getElementById("btnAdapt");
  const loader    = document.getElementById("adaptLoader");
  const errBox    = document.getElementById("adaptError");
  const results   = document.getElementById("adaptResults");
  const lossBox   = document.getElementById("adaptLossBox");

  if (!fileInp.files[0]) { showError(errBox, "Please select an image file."); return; }

  clearError(errBox);
  btn.disabled = true;
  loader.classList.add("active");
  results.style.display = "none";
  lossBox.style.display  = "none";

  const fd = new FormData();
  fd.append("image",        fileInp.files[0]);
  fd.append("source",       source);
  fd.append("adapt_steps",  steps);

  try {
    const res  = await fetch("/api/adapt_and_predict", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) { showError(errBox, data.error); return; }

    // Fill images
    const origUrl = URL.createObjectURL(fileInp.files[0]);
    ["adaptOrigImg", "adaptOrigImg2"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.src = origUrl;
    });

    setImgB64("srcMaskImg",    data.source_mask_b64);
    setImgB64("srcOverlayImg", data.source_overlay_b64);
    setImgB64("adaMaskImg",    data.adapted_mask_b64);
    setImgB64("adaOverlayImg", data.adapted_overlay_b64);

    document.getElementById("srcFgPill").textContent = `FG: ${data.source_fg_ratio}%`;
    document.getElementById("adaFgPill").textContent = `FG: ${data.adapted_fg_ratio}%`;

    results.style.display = "flex";

    // Loss chart
    if (data.adapt_losses?.length) {
      lossBox.style.display = "block";
      drawLossChart(data.adapt_losses);
    }

  } catch(e) {
    showError(errBox, e.toString());
  } finally {
    btn.disabled = false;
    loader.classList.remove("active");
  }
}

function drawLossChart(losses) {
  const ctx = document.getElementById("adaptLossChart");
  if (!ctx) return;
  if (adaptLossChart) adaptLossChart.destroy();
  adaptLossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: losses.map((_, i) => `Step ${i + 1}`),
      datasets: [{
        label: "Adaptation Loss",
        data: losses,
        borderColor: "rgba(79,156,249,0.9)",
        backgroundColor: "rgba(79,156,249,0.15)",
        borderWidth: 2,
        pointRadius: 4,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#2e3250" } },
      }
    }
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setImgB64(id, b64) {
  const el = document.getElementById(id);
  if (el && b64) el.src = "data:image/png;base64," + b64;
}

function showError(el, msg) {
  el.textContent = msg;
  el.classList.add("active");
}

function clearError(el) {
  el.textContent = "";
  el.classList.remove("active");
}
