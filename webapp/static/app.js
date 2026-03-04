(function () {
  "use strict";

  const poemInput    = document.getElementById("poem-input");
  const analyzeBtn   = document.getElementById("analyze-btn");
  const clearBtn     = document.getElementById("clear-btn");
  const resultSection= document.getElementById("result-section");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const loadingEl    = document.getElementById("loading");

  // result sub-elements
  const eraHeader    = document.getElementById("era-header");
  const eraBadge     = document.getElementById("era-badge");
  const eraName      = document.getElementById("era-name");
  const eraYears     = document.getElementById("era-years");
  const eraDesc      = document.getElementById("era-desc");
  const probBars     = document.getElementById("prob-bars");
  const reasonsList  = document.getElementById("reasons-list");
  const featureTable = document.getElementById("feature-table").querySelector("tbody");
  const correctEra   = document.getElementById("correct-era");
  const feedbackThanks = document.getElementById("feedback-thanks");

  function esc(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  function hideAll() {
    resultSection.hidden = true;
    errorSection.hidden  = true;
    loadingEl.hidden     = true;
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorSection.hidden = false;
    resultSection.hidden = true;
    loadingEl.hidden = true;
  }

  function chipClass(ctx) {
    const map = { high: "chip-high", low: "chip-low", moderate: "chip-typical",
                  none: "chip-none", present: "chip-present" };
    return map[ctx] || "chip-typical";
  }

  function renderResult(data) {
    errorSection.hidden = true;

    // Era header
    eraBadge.style.background = data.era_meta.color;
    eraName.textContent  = data.era_meta.label;
    eraYears.textContent = data.era_meta.years;
    eraDesc.textContent  = data.era_meta.description;

    // Probability bars
    probBars.innerHTML = "";
    const sorted = [...data.probabilities].sort((a, b) => b.prob - a.prob);
    sorted.forEach(function (item) {
      const pct  = Math.round(item.prob * 100);
      const isActive = item.era === data.era;
      const row  = document.createElement("div");
      row.className = "prob-row" + (isActive ? " active" : "");

      // Choose bar colour from era meta (sent in probabilities array's colour)
      const colorMap = {
        "Pre-1800":  "#5c4033",
        "1800-1900": "#2d4a3e",
        "Post-1900": "#1a3a5c",
      };
      const barColor = isActive ? (colorMap[item.era] || "#2d4a3e") : "#c8c0b4";

      row.innerHTML =
        '<div class="prob-era-name">' + esc(item.label) + "</div>" +
        '<div class="prob-bar-bg"><div class="prob-bar-fill" style="width:' +
        pct + '%;background:' + barColor + '"></div></div>' +
        '<div class="prob-pct">' + pct + "%</div>";
      probBars.appendChild(row);
    });

    // Reasons
    reasonsList.innerHTML = "";
    (data.top_reasons || []).forEach(function (reason) {
      const li = document.createElement("li");
      li.innerHTML = '<span class="reason-bullet"></span><span>' + esc(reason) + "</span>";
      reasonsList.appendChild(li);
    });

    // Reset feedback dropdown
    correctEra.value = "";
    feedbackThanks.hidden = true;

    // Feature table
    featureTable.innerHTML = "";
    (data.features || []).forEach(function (f) {
      const valStr = f.unit ? f.value + "\u202f" + f.unit : String(f.value);
      const ctxLabel = { high: "High", low: "Low", moderate: "Moderate",
                         none: "None", present: "Present" }[f.context] || f.context;
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td class="feat-name">'  + esc(f.label)  + "</td>" +
        '<td class="feat-value">' + esc(valStr)    + "</td>" +
        '<td class="feat-ctx"><span class="feat-chip ' + chipClass(f.context) + '">' +
        esc(ctxLabel) + "</span></td>";
      featureTable.appendChild(tr);
    });

    resultSection.hidden = false;
    loadingEl.hidden = true;
  }

  analyzeBtn.addEventListener("click", function () {
    const poem = poemInput.value.trim();
    hideAll();
    if (!poem) { showError("Please paste or type a poem first."); return; }

    loadingEl.hidden = false;
    analyzeBtn.disabled = true;

    fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ poem: poem }),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.error || "Analysis failed.");
          return body;
        });
      })
      .then(renderResult)
      .catch(function (err) { showError(err.message || "Something went wrong. Try again."); })
      .finally(function () { loadingEl.hidden = true; analyzeBtn.disabled = false; });
  });

  clearBtn.addEventListener("click", function () {
    poemInput.value = "";
    poemInput.focus();
    hideAll();
  });

  poemInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); analyzeBtn.click(); }
  });

  correctEra.addEventListener("change", function () {
    if (this.value) { feedbackThanks.hidden = false; }
  });
})();
