(function () {
  const poemInput = document.getElementById("poem-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-btn");
  const resultSection = document.getElementById("result-section");
  const resultContent = document.getElementById("result-content");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const loadingEl = document.getElementById("loading");

  function hideAllFeedback() {
    resultSection.hidden = true;
    errorSection.hidden = true;
    loadingEl.hidden = true;
  }

  function showLoading(show) {
    loadingEl.hidden = !show;
    if (show) {
      resultSection.hidden = true;
      errorSection.hidden = true;
    }
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorSection.hidden = false;
    resultSection.hidden = true;
    loadingEl.hidden = true;
  }

  function renderResult(data) {
    errorSection.hidden = true;

    if (data.message && !data.era) {
      resultContent.innerHTML =
        '<p class="result-message">' + escapeHtml(data.message) + "</p>";
    } else {
      let html = "";
      if (data.era) {
        html += '<p class="result-era">' + escapeHtml(data.era) + "</p>";
        if (data.confidence != null) {
          const pct = Math.round(data.confidence * 100);
          html +=
            '<p class="result-confidence">Confidence: ' + pct + "%</p>";
        }
      }
      if (data.alternatives && data.alternatives.length > 0) {
        html += '<div class="result-alternatives">';
        html += "<h3>Other close matches</h3><ul>";
        data.alternatives.forEach(function (alt) {
          const pct = Math.round((alt.confidence || 0) * 100);
          html +=
            "<li><span>" +
            escapeHtml(alt.era) +
            "</span> <span>" +
            pct +
            "%</span></li>";
        });
        html += "</ul></div>";
      }
      if (data.message && data.era) {
        html += '<p class="result-message">' + escapeHtml(data.message) + "</p>";
      }
      resultContent.innerHTML = html || "<p>No result.</p>";
    }

    resultSection.hidden = false;
    loadingEl.hidden = true;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  analyzeBtn.addEventListener("click", function () {
    const poem = poemInput.value.trim();
    hideAllFeedback();

    if (!poem) {
      showError("Please paste or type a poem first.");
      return;
    }

    showLoading(true);
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
      .catch(function (err) {
        showError(err.message || "Something went wrong. Try again.");
      })
      .finally(function () {
        showLoading(false);
        analyzeBtn.disabled = false;
      });
  });

  clearBtn.addEventListener("click", function () {
    poemInput.value = "";
    poemInput.focus();
    hideAllFeedback();
  });

  poemInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      analyzeBtn.click();
    }
  });
})();
