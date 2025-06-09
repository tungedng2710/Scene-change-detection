// /static/app.js
const form      = document.getElementById("uploadForm");
const results   = document.getElementById("results");
const percentEl = document.getElementById("percent");
const logsEl    = document.getElementById("logs");
const maskEl    = document.getElementById("maskImg");
const btn       = form.querySelector("button");

/* ——— tiny helper ——— */
function setBusy(isBusy) {
  if (isBusy) {
    results.classList.add("opacity-50", "pointer-events-none");
    btn.disabled   = true;
    btn.textContent = "⏳ Working…";
  } else {
    results.classList.remove("opacity-50", "pointer-events-none");
    btn.disabled   = false;
    btn.textContent = "🔍 Detect Changes";
  }
}

/* ——— main handler ——— */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  /* ensure the panel is visible no matter what */
  results.classList.remove("hidden");

  setBusy(true);

  try {
    const data = new FormData(form);
    const res  = await fetch("/detect", { method: "POST", body: data });
    if (!res.ok) throw new Error(await res.text());

    const out = await res.json();

    // populate UI
    percentEl.textContent = out.percent ?? "";
    logsEl.value          = out.logs    ?? "";

    if (out.mask_url) {
      maskEl.src = `${out.mask_url}?${Date.now()}`; // cache-bust
      maskEl.classList.remove("hidden");
    } else {
      maskEl.classList.add("hidden");
    }
  } catch (err) {
    alert("Lỗi: " + err.message);
  } finally {
    setBusy(false);
  }
});
