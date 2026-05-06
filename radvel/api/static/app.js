// RadVel operator UI — vanilla ES module, no build step.
//
// Route table (hash-based):
//   #/runs                       — list of runs
//   #/runs/new                   — create a run from JSON
//   #/runs/{id}                  — overview + step buttons
//   #/runs/{id}/jobs/{jid}       — live MCMC/NS progress
//   #/runs/{id}/files            — output files
//
// Keyboard shortcuts: g r (runs), g n (new run), ? (help), j/k (table nav).

const $app = document.getElementById("app");
const $version = document.getElementById("version-tag");

const api = {
  async get(path) { return fetch(path).then(r => r.json()); },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, body: data };
  },
  async del(path) {
    const r = await fetch(path, { method: "DELETE" });
    const data = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, body: data };
  },
};

// ---- helpers --------------------------------------------------------------

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function fmtTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KiB";
  return (n / 1024 / 1024).toFixed(1) + " MiB";
}

function badge(state) {
  const colors = {
    queued: "bg-slate-200 text-slate-700",
    running: "bg-amber-100 text-amber-800",
    succeeded: "bg-emerald-100 text-emerald-800",
    failed: "bg-rose-100 text-rose-800",
    cancelled: "bg-slate-200 text-slate-600",
  };
  return el("span", { class: `text-xs px-2 py-0.5 rounded ${colors[state] || "bg-slate-100"}` }, state || "—");
}

function clear() {
  $app.replaceChildren();
}

// ---- example payload -----------------------------------------------------

const EXAMPLE_PAYLOAD = {
  starname: "epic203771098",
  nplanets: 2,
  instnames: ["j"],
  fitting_basis: "per tc secosw sesinw k",
  bjd0: 2454833.0,
  planet_letters: { "1": "b", "2": "c" },
  params: {
    per1: { value: 20.885258, vary: false },
    tc1: { value: 2072.79438, vary: false },
    secosw1: { value: 0.0, vary: false },
    sesinw1: { value: 0.1, vary: false },
    k1: { value: 10.0 },
    per2: { value: 42.363011, vary: false },
    tc2: { value: 2082.62516, vary: false },
    secosw2: { value: 0.0, vary: false },
    sesinw2: { value: 0.1, vary: false },
    k2: { value: 10.0 },
    dvdt: { value: 0.0 },
    curv: { value: 0.0 },
    gamma_j: { value: 1.0, vary: false, linear: true },
    jit_j: { value: 2.6 },
  },
  data: { kind: "dataset_ref", dataset: "epic203771098.csv" },
  priors: [
    { type: "eccentricity", num_planets: 2 },
    { type: "positivek", num_planets: 2 },
    { type: "jeffreys", param: "k1", minval: 1e-2, maxval: 1e3 },
    { type: "jeffreys", param: "k2", minval: 1e-2, maxval: 1e3 },
    { type: "hardbounds", param: "jit_j", minval: 0.0, maxval: 15.0 },
    { type: "gaussian", param: "dvdt", mu: 0.0, sigma: 1.0 },
    { type: "gaussian", param: "curv", mu: 0.0, sigma: 0.1 },
  ],
  stellar: { mstar: 1.12, mstar_err: 0.05 },
};

// ---- views ----------------------------------------------------------------

async function viewRunsList() {
  clear();
  $app.appendChild(el("div", { class: "text-slate-400" }, el("span", { class: "spinner" }), " loading runs…"));
  const runs = await api.get("/runs");
  clear();
  $app.appendChild(el("div", { class: "flex justify-between items-center mb-4" },
    el("h1", { class: "text-2xl font-semibold" }, "Runs"),
    el("a", { href: "#/runs/new", class: "bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm" }, "+ New run")
  ));
  if (!runs.length) {
    $app.appendChild(el("p", { class: "text-slate-500" }, "No runs yet — start with ", el("a", { href: "#/runs/new", class: "text-blue-600 underline" }, "+ New run"), "."));
    return;
  }
  const table = el("table", { class: "w-full text-sm bg-white shadow-sm rounded overflow-hidden", "data-table": "runs" });
  table.appendChild(el("thead", { class: "bg-slate-100 text-left" },
    el("tr", {},
      el("th", { class: "px-4 py-2" }, "run_id"),
      el("th", { class: "px-4 py-2" }, "starname"),
      el("th", { class: "px-4 py-2" }, "nplanets"),
      el("th", { class: "px-4 py-2" }, "created"),
    )
  ));
  const tbody = el("tbody", {});
  for (const r of runs) {
    tbody.appendChild(el("tr", {
      class: "border-t border-slate-100 hover:bg-slate-50 cursor-pointer",
      onclick: () => { location.hash = `#/runs/${r.run_id}`; },
    },
      el("td", { class: "px-4 py-2 mono" }, r.run_id),
      el("td", { class: "px-4 py-2" }, r.starname || "—"),
      el("td", { class: "px-4 py-2" }, String(r.nplanets ?? "—")),
      el("td", { class: "px-4 py-2 text-slate-500" }, fmtTime(r.created_at)),
    ));
  }
  table.appendChild(tbody);
  $app.appendChild(table);
}

async function viewNewRun() {
  clear();
  $app.appendChild(el("h1", { class: "text-2xl font-semibold mb-4" }, "Create a run"));

  // Health check tells us whether .py upload is permitted.
  const health = await api.get("/healthz").catch(() => ({}));
  const allowPyUpload = !!health.allow_py_upload;

  // Tab strip
  const tabs = ["Form", "JSON", ...(allowPyUpload ? ["Upload .py"] : [])];
  const panels = {};
  let active = "Form";

  const tabBar = el("div", { class: "flex border-b border-slate-300 mb-4" });
  const setActive = (name) => {
    active = name;
    for (const t of tabs) {
      const btn = panels[t].btn;
      btn.className = "px-4 py-2 text-sm border-b-2 -mb-px " + (
        t === name
          ? "border-blue-600 text-blue-700 font-medium"
          : "border-transparent text-slate-500 hover:text-slate-800"
      );
      panels[t].body.style.display = t === name ? "" : "none";
    }
  };

  const status = el("div", { class: "mt-3 text-sm" });
  panels["Form"] = buildFormPanel(status);
  panels["JSON"] = buildJsonPanel(status);
  if (allowPyUpload) panels["Upload .py"] = buildUploadPyPanel(status);

  for (const t of tabs) {
    const btn = panels[t].btn;
    btn.onclick = () => setActive(t);
    tabBar.appendChild(btn);
  }
  $app.appendChild(tabBar);
  for (const t of tabs) $app.appendChild(panels[t].body);
  $app.appendChild(status);

  setActive("Form");
}

// ---- "JSON" tab — paste/edit a full payload --------------------------

function buildJsonPanel(status) {
  const btn = el("button", { class: "px-4 py-2 text-sm" }, "JSON");
  const ta = el("textarea", {
    class: "w-full h-96 font-mono text-xs p-3 border border-slate-300 rounded focus:outline-none focus:border-blue-500",
    placeholder: "Paste a setup JSON payload here…",
  });
  const submit = el("button", {
    class: "bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm disabled:opacity-50",
    onclick: async () => submitPayload(ta.value, status, submit),
  }, "Create run");
  const example = el("button", {
    class: "ml-2 px-3 py-2 text-sm border border-slate-300 rounded hover:bg-slate-100",
    onclick: () => { ta.value = JSON.stringify(EXAMPLE_PAYLOAD, null, 2); },
  }, "Load example");
  const body = el("div", {}, ta, el("div", { class: "mt-3" }, submit, example));
  return { btn, body };
}

async function submitPayload(jsonText, status, submitBtn) {
  submitBtn.disabled = true;
  status.replaceChildren(el("span", { class: "spinner" }), " creating run…");
  let payload;
  try { payload = JSON.parse(jsonText); }
  catch (e) {
    status.textContent = "Invalid JSON: " + e.message;
    submitBtn.disabled = false;
    return;
  }
  const resp = await api.post("/runs", payload);
  submitBtn.disabled = false;
  if (resp.ok) {
    location.hash = `#/runs/${resp.body.run_id}`;
  } else {
    status.replaceChildren(el("pre", { class: "text-rose-700 whitespace-pre-wrap" },
      JSON.stringify(resp.body, null, 2)));
  }
}

// ---- "Upload .py" tab — multipart upload of a setup file -------------

function buildUploadPyPanel(status) {
  const btn = el("button", { class: "px-4 py-2 text-sm" }, "Upload .py");
  const fileInput = el("input", { type: "file", accept: ".py", class: "block text-sm" });
  const submit = el("button", {
    class: "mt-3 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm disabled:opacity-50",
    onclick: async () => {
      const f = fileInput.files[0];
      if (!f) { status.textContent = "Pick a .py file first."; return; }
      submit.disabled = true;
      status.replaceChildren(el("span", { class: "spinner" }), ` uploading ${f.name}…`);
      const fd = new FormData();
      fd.append("setup_file", f);
      const r = await fetch("/runs/upload-py", { method: "POST", body: fd });
      const data = await r.json().catch(() => ({}));
      submit.disabled = false;
      if (r.ok) {
        location.hash = `#/runs/${data.run_id}`;
      } else {
        status.replaceChildren(el("pre", { class: "text-rose-700 whitespace-pre-wrap" },
          `${r.status} ${JSON.stringify(data, null, 2)}`));
      }
    },
  }, "Upload");
  const body = el("div", {},
    el("p", { class: "text-sm text-slate-600 mb-3" },
      "Upload an existing radvel setup file (the same Python module the CLI uses). " +
      "Server will import it to extract metadata, then run the pipeline against it directly."),
    fileInput, el("div", {}, submit),
    el("p", { class: "mt-3 text-xs text-slate-500" },
      "Disabled by default. Enabled here because RADVEL_API_ALLOW_PY_UPLOAD=true. " +
      "Only do this with setup files you trust — they execute on the server."),
  );
  return { btn, body };
}

// ---- "Form" tab — guided builder for the most common shape -----------

function buildFormPanel(status) {
  const btn = el("button", { class: "px-4 py-2 text-sm" }, "Form");

  const F = (lbl, input, hint) => el("label", { class: "block mb-3" },
    el("div", { class: "text-xs text-slate-600 mb-1" }, lbl,
      hint ? el("span", { class: "text-slate-400" }, " — " + hint) : null),
    input
  );
  const TXT = (val = "") => el("input", { type: "text", value: val, class: "border border-slate-300 rounded px-2 py-1 w-full text-sm" });
  const NUM = (val = "", step = "any") => el("input", { type: "number", value: String(val), step, class: "border border-slate-300 rounded px-2 py-1 w-full text-sm font-mono" });
  const CHK = (val = false) => { const i = el("input", { type: "checkbox" }); i.checked = val; return i; };

  const starname = TXT("epic203771098");
  const nplanets = NUM(2, "1");
  const instnames = TXT("j");
  const fittingBasis = el("select", { class: "border border-slate-300 rounded px-2 py-1 w-full text-sm" },
    ...["per tc secosw sesinw k", "per tc e w k", "per tp e w k", "per tc se w k",
        "per tc secosw sesinw logk", "logper tc secosw sesinw k",
        "logper tc secosw sesinw logk", "per tc se omega k", "per tc ecosw esinw k",
        "logper tc ecosw esinw k"].map(b => el("option", { value: b }, b))
  );
  const bjd0 = NUM(2454833.0);

  // Data source — either reference a built-in CSV under RADVEL_DATADIR
  // or upload a CSV from the operator's machine. Upload reads the file
  // in the browser, base64-encodes it, and submits as kind="csv_base64"
  // so no second HTTP round-trip is needed.
  const dataSource = el("select", { class: "border border-slate-300 rounded px-2 py-1 w-full text-sm" },
    el("option", { value: "dataset_ref" }, "Built-in dataset"),
    el("option", { value: "csv_upload" }, "Upload CSV"),
  );
  const dataset = TXT("epic203771098.csv");
  const csvFile = el("input", { type: "file", accept: ".csv,text/csv", class: "block text-sm" });
  const csvSep = TXT(",");
  csvSep.style.maxWidth = "4rem";
  const datasetGroup = el("div", {}, F("dataset (built-in CSV)", dataset, "path under DATADIR"));
  const csvGroup = el("div", { style: "display:none" },
    F("CSV file", csvFile, "columns: time, mnvel, errvel, tel — or t/vel/errvel"),
    el("div", { class: "flex items-center gap-2 -mt-2" },
      el("span", { class: "text-xs text-slate-600" }, "separator:"),
      csvSep,
    ),
  );
  dataSource.onchange = () => {
    const upload = dataSource.value === "csv_upload";
    datasetGroup.style.display = upload ? "none" : "";
    csvGroup.style.display = upload ? "" : "none";
  };

  const mstar = NUM(1.12);
  const mstarErr = NUM(0.05);
  const dvdtVary = CHK(true);
  const curvVary = CHK(true);
  const eccPrior = CHK(true);
  const positiveK = CHK(true);

  // Per-planet params shown when nplanets changes
  const planetsContainer = el("div", { class: "space-y-3" });

  const renderPlanets = () => {
    const n = Math.max(1, Math.min(20, parseInt(nplanets.value || "1", 10)));
    planetsContainer.replaceChildren();
    for (let i = 1; i <= n; i++) {
      const per = NUM(i === 1 ? 20.885258 : 42.363011);
      const tc = NUM(i === 1 ? 2072.79438 : 2082.62516);
      const k = NUM(10);
      const perVary = CHK(false);
      const tcVary = CHK(false);
      const kVary = CHK(true);
      const card = el("div", { class: "border border-slate-200 rounded p-3 bg-slate-50" },
        el("div", { class: "font-medium mb-2 text-sm" }, "Planet " + i),
        el("div", { class: "grid grid-cols-3 gap-3" },
          el("div", {}, F("per (days)", per), el("label", { class: "text-xs flex gap-1 -mt-2" }, perVary, " vary")),
          el("div", {}, F("tc", tc), el("label", { class: "text-xs flex gap-1 -mt-2" }, tcVary, " vary")),
          el("div", {}, F("K (m/s)", k), el("label", { class: "text-xs flex gap-1 -mt-2" }, kVary, " vary")),
        ),
      );
      planetsContainer.appendChild(card);
      planetsContainer._inputs ||= [];
      planetsContainer._inputs[i] = { per, tc, k, perVary, tcVary, kVary };
    }
  };
  nplanets.oninput = renderPlanets;
  renderPlanets();

  const collect = () => buildPayloadFromForm({
    starname, nplanets, instnames, fittingBasis, bjd0,
    dataSource, dataset, csvFile, csvSep,
    mstar, mstarErr, dvdtVary, curvVary, eccPrior, positiveK,
    planets: planetsContainer._inputs,
  });

  const submit = el("button", {
    class: "bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm disabled:opacity-50",
    onclick: async () => {
      try {
        const payload = await collect();
        submitPayload(JSON.stringify(payload), status, submit);
      } catch (e) {
        status.textContent = e.message || String(e);
      }
    },
  }, "Create run");

  const previewBtn = el("button", {
    class: "ml-2 px-3 py-2 text-sm border border-slate-300 rounded hover:bg-slate-100",
    onclick: async () => {
      try {
        const payload = await collect();
        // Truncate the base64 blob for display so the panel stays useful.
        const display = JSON.parse(JSON.stringify(payload));
        if (display?.data?.csv_base64) {
          const n = display.data.csv_base64.length;
          display.data.csv_base64 = `<${n} chars of base64-encoded CSV>`;
        }
        status.replaceChildren(el("pre", { class: "text-xs bg-slate-100 p-3 rounded whitespace-pre-wrap" },
          JSON.stringify(display, null, 2)));
      } catch (e) {
        status.textContent = e.message || String(e);
      }
    },
  }, "Preview JSON");

  const body = el("div", { class: "space-y-4" },
    el("div", { class: "grid grid-cols-2 gap-4" },
      el("div", {}, F("starname", starname)),
      el("div", {}, F("nplanets", nplanets, "1–20")),
      el("div", {}, F("instnames", instnames, "comma-separated, e.g. \"j,k\"")),
      el("div", {}, F("fitting_basis", fittingBasis)),
      el("div", {}, F("bjd0", bjd0, "reference epoch")),
      el("div", {}, F("data source", dataSource), datasetGroup, csvGroup),
      el("div", {}, F("mstar (M☉)", mstar)),
      el("div", {}, F("mstar_err", mstarErr)),
    ),
    planetsContainer,
    el("div", { class: "border-t pt-3" },
      el("div", { class: "text-sm font-medium mb-2" }, "Trend / priors"),
      el("div", { class: "grid grid-cols-4 gap-3 text-xs" },
        el("label", { class: "flex items-center gap-1" }, dvdtVary, " vary dvdt"),
        el("label", { class: "flex items-center gap-1" }, curvVary, " vary curv"),
        el("label", { class: "flex items-center gap-1" }, eccPrior, " eccentricity prior"),
        el("label", { class: "flex items-center gap-1" }, positiveK, " positive-K prior"),
      )
    ),
    el("div", {}, submit, previewBtn),
  );

  return { btn, body };
}

async function buildPayloadFromForm(f) {
  const insts = f.instnames.value.split(",").map(s => s.trim()).filter(Boolean);
  const params = {};
  const priors = [];
  const n = parseInt(f.nplanets.value, 10);
  for (let i = 1; i <= n; i++) {
    const p = f.planets[i];
    params[`per${i}`] = { value: Number(p.per.value), vary: p.perVary.checked };
    params[`tc${i}`] = { value: Number(p.tc.value), vary: p.tcVary.checked };
    params[`secosw${i}`] = { value: 0.0, vary: false };
    params[`sesinw${i}`] = { value: 0.1, vary: false };
    params[`k${i}`] = { value: Number(p.k.value), vary: p.kVary.checked };
    priors.push({ type: "jeffreys", param: `k${i}`, minval: 0.01, maxval: 1000 });
  }
  params.dvdt = { value: 0.0, vary: f.dvdtVary.checked };
  params.curv = { value: 0.0, vary: f.curvVary.checked };
  for (const i of insts) {
    params[`gamma_${i}`] = { value: 1.0, vary: false, linear: true };
    params[`jit_${i}`] = { value: 2.6, vary: true };
    priors.push({ type: "hardbounds", param: `jit_${i}`, minval: 0.0, maxval: 15.0 });
  }
  if (f.dvdtVary.checked) priors.push({ type: "gaussian", param: "dvdt", mu: 0.0, sigma: 1.0 });
  if (f.curvVary.checked) priors.push({ type: "gaussian", param: "curv", mu: 0.0, sigma: 0.1 });
  if (f.eccPrior.checked) priors.push({ type: "eccentricity", num_planets: n });
  if (f.positiveK.checked) priors.push({ type: "positivek", num_planets: n });

  let data;
  if (f.dataSource.value === "csv_upload") {
    const file = f.csvFile.files[0];
    if (!file) throw new Error("Pick a CSV file or switch the data source to a built-in dataset.");
    const csv_base64 = await readFileAsBase64(file);
    data = { kind: "csv_base64", csv_base64, separator: f.csvSep.value || "," };
  } else {
    data = { kind: "dataset_ref", dataset: f.dataset.value };
  }

  const payload = {
    starname: f.starname.value,
    nplanets: n,
    instnames: insts,
    fitting_basis: f.fittingBasis.value,
    bjd0: Number(f.bjd0.value),
    params,
    data,
    priors,
  };
  if (f.mstar.value) {
    payload.stellar = { mstar: Number(f.mstar.value), mstar_err: Number(f.mstarErr.value || 0) };
  }
  return payload;
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      // result is "data:<mime>;base64,<payload>" — strip the prefix.
      const s = String(r.result);
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    r.onerror = () => reject(r.error || new Error("file read failed"));
    r.readAsDataURL(file);
  });
}

async function viewRunDetail(runId) {
  clear();
  const header = el("div", { class: "flex justify-between items-baseline mb-4" },
    el("h1", { class: "text-2xl font-semibold mono" }, runId),
    el("div", { class: "flex gap-3 text-sm" },
      el("a", { href: `#/runs/${runId}/files`, class: "text-blue-600 underline" }, "Files →"),
    ),
  );
  const summary = el("div", { class: "bg-white rounded p-4 mb-4 shadow-sm" });
  const stepsContainer = el("div", { class: "space-y-2 mb-6" });
  const resultsContainer = el("div", { class: "mb-4" });
  const log = el("pre", { class: "bg-slate-900 text-slate-100 p-3 rounded text-xs overflow-x-auto whitespace-pre-wrap min-h-[6rem]" }, "ready.\n");
  $app.append(header, summary, stepsContainer, resultsContainer, log);

  window.__rv_log = (line) => {
    log.textContent += line + "\n";
    log.scrollTop = log.scrollHeight;
  };

  // Step buttons call this after a sync action so the panel refreshes
  // immediately rather than waiting up to 3 s for the next interval tick.
  window.__rv_refresh = () => render().catch(() => {});

  let lastSig = "";
  const render = async () => {
    const run = await api.get(`/runs/${runId}`);
    if (run.detail) {
      stepsContainer.replaceChildren(
        el("p", { class: "text-rose-700" }, `Error: ${JSON.stringify(run.detail)}`)
      );
      return;
    }
    // Cheap fingerprint to skip re-render when nothing changed.
    const sig = JSON.stringify([run.stat, run.active_job]);
    if (sig === lastSig) return;
    lastSig = sig;

    summary.replaceChildren(
      el("div", { class: "grid grid-cols-2 gap-2 text-sm" },
        el("div", { class: "text-slate-500" }, "starname"),
        el("div", {}, run.starname || "—"),
        el("div", { class: "text-slate-500" }, "nplanets"),
        el("div", {}, String(run.nplanets ?? "—")),
        el("div", { class: "text-slate-500" }, "outputdir"),
        el("div", { class: "mono text-xs truncate" }, run.outputdir || ""),
        el("div", { class: "text-slate-500" }, "active job"),
        el("div", {}, run.active_job
          ? el("a", { href: `#/runs/${runId}/jobs/${run.active_job.job_id}`,
                      class: "text-blue-600 underline mono" }, run.active_job.job_id)
          : "—"),
      )
    );

    stepsContainer.replaceChildren(...renderSteps(runId, run));
    resultsContainer.replaceChildren(await renderResults(runId, run));
  };

  await render();
  // Re-poll every 3 s while a job is active (or just after one finished)
  // so the step list flips to "done" without a manual refresh.
  const interval = setInterval(async () => {
    if (!document.body.contains(stepsContainer)) {
      clearInterval(interval);
      return;
    }
    await render().catch(() => {});
  }, 3000);
}

function renderSteps(runId, run) {
  const stat = run.stat || {};
  const has = (k) => stat[k] && stat[k].run === "True";
  const active = run.active_job;
  const activeKind = active && ["queued", "running"].includes(active.state) ? active.kind : null;

  // Reason text that explains why a step is gated.
  const reasons = {
    mcmc: !has("fit") && "Run Fit first.",
    ns: !has("fit") && "Run Fit first.",
    derive: !(has("mcmc") || has("ns")) && "Run MCMC or NS first.",
    ic: !has("fit") && "Run Fit first.",
    tables: !has("fit") && "Run Fit first.",
    "plots-rv": !has("fit") && "Run Fit first.",
    "plots-corner": !(has("mcmc") || has("ns")) && "Run MCMC or NS first.",
    report: !(has("mcmc") || has("ns")) && "Run MCMC or NS first.",
  };

  const steps = [
    {
      id: "fit", label: "Fit (MAP)", enabled: true,
      done: has("fit"),
      summary: () => has("fit") && "Maximum-likelihood fit completed",
      action: () => runStep(runId, "fit", {}),
    },
    {
      id: "mcmc", label: "MCMC",
      enabled: has("fit") && !activeKind,
      running: activeKind === "mcmc",
      done: has("mcmc"),
      reason: reasons.mcmc,
      summary: () => has("mcmc") && [
        stat.mcmc.nsteps ? `${stat.mcmc.nsteps} steps` : null,
        stat.mcmc.minafactor ? `A-factor ${Number(stat.mcmc.minafactor).toFixed(1)}` : null,
        stat.mcmc.maxgr ? `GR ${Number(stat.mcmc.maxgr).toFixed(3)}` : null,
      ].filter(Boolean).join(" · "),
      action: () => mcmcModal(runId),
      live: () => active && active.kind === "mcmc"
        ? `#/runs/${runId}/jobs/${active.job_id}` : null,
    },
    {
      id: "ns", label: "Nested sampling",
      enabled: has("fit") && !activeKind,
      running: activeKind === "ns",
      done: has("ns"),
      reason: reasons.ns,
      summary: () => has("ns") && `chains: ${(stat.ns.chainfile || "").split("/").pop() || "—"}`,
      action: () => nsModal(runId),
      live: () => active && active.kind === "ns"
        ? `#/runs/${runId}/jobs/${active.job_id}` : null,
    },
    {
      id: "derive", label: "Derive physical params",
      enabled: (has("mcmc") || has("ns")) && !activeKind,
      done: has("derive"),
      reason: reasons.derive,
      summary: () => has("derive") && "physical-parameter chains written",
      action: () => runStep(runId, "derive", {}),
    },
    {
      id: "ic", label: "IC compare",
      enabled: has("fit") && !activeKind,
      done: has("ic_compare"),
      reason: reasons.ic,
      summary: () => has("ic_compare") && "AIC/BIC table written",
      action: () => runStep(runId, "ic", { types: ["e"], simple: true }),
    },
    {
      id: "tables", label: "Tables",
      enabled: has("fit") && !activeKind,
      done: !!(stat.tables),
      reason: reasons.tables,
      summary: () => stat.tables && Object.keys(stat.tables).join(", "),
      action: () => runStep(runId, "tables", { types: ["params", "priors", "rv"] }),
    },
    {
      id: "plots-rv", label: "Plot: RV multipanel",
      enabled: has("fit") && !activeKind,
      done: !!(stat.plot && stat.plot.rv_plot),
      reason: reasons["plots-rv"],
      summary: () => stat.plot && stat.plot.rv_plot && "PDF written",
      action: () => runStep(runId, "plots", { types: ["rv"] }),
    },
    {
      id: "plots-corner", label: "Plot: corner",
      enabled: (has("mcmc") || has("ns")) && !activeKind,
      done: !!(stat.plot && stat.plot.corner_plot),
      reason: reasons["plots-corner"],
      summary: () => stat.plot && stat.plot.corner_plot && "PDF written",
      action: () => runStep(runId, "plots", { types: ["corner"] }),
    },
    {
      id: "report", label: "Report (PDF)",
      enabled: (has("mcmc") || has("ns")) && !activeKind,
      done: !!(stat.report),
      reason: reasons.report,
      summary: () => stat.report && "PDF written",
      action: () => runStep(runId, "report", {}),
    },
  ];

  return steps.map((s) => {
    let badge, dotClass, btn;
    if (s.running) {
      badge = el("span", { class: "text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800" },
        el("span", { class: "spinner mr-1", style: "vertical-align:-2px" }), "running");
      dotClass = "bg-amber-400";
      const live = s.live && s.live();
      btn = el("a", {
        href: live || "#",
        class: "px-3 py-1 text-sm border border-blue-300 rounded text-blue-700 hover:bg-blue-50",
      }, "View progress →");
    } else if (s.done) {
      badge = el("span", { class: "text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-800" }, "✓ done");
      dotClass = "bg-emerald-500";
      btn = _actionButton(s, "Re-run",
        "px-3 py-1 text-sm border border-slate-300 rounded text-slate-700 hover:bg-slate-100");
    } else if (s.enabled) {
      badge = el("span", { class: "text-xs px-2 py-0.5 rounded bg-slate-200 text-slate-700" }, "ready");
      dotClass = "bg-slate-400";
      btn = _actionButton(s, "Run",
        "px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700");
    } else {
      badge = el("span", { class: "text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-400" }, "blocked");
      dotClass = "bg-slate-200";
      btn = el("button", {
        class: "px-3 py-1 text-sm bg-slate-100 text-slate-400 rounded cursor-not-allowed",
        disabled: true,
        title: s.reason || "",
      }, "Run");
    }
    const summary = (s.summary && s.summary()) || (!s.enabled && s.reason) || "";
    return el("div", { class: "bg-white rounded p-3 shadow-sm flex items-center gap-3" },
      el("span", { class: `inline-block w-2 h-2 rounded-full ${dotClass}` }),
      el("div", { class: "flex-1" },
        el("div", { class: "flex items-center gap-2" },
          el("span", { class: "font-medium text-sm" }, s.label),
          badge,
        ),
        summary ? el("div", { class: "text-xs text-slate-500 mt-0.5" }, summary) : null,
      ),
      btn,
    );
  });
}

async function renderResults(runId, run) {
  const stat = run.stat || {};
  // Pull the file list once and slot files into the right result panel.
  const files = await api.get(`/runs/${runId}/files`).catch(() => []);
  const byName = (suffix) => files.find(f => f.name.endsWith(suffix));
  const url = (name) => `/runs/${runId}/files/${name}`;

  const sections = [];

  // ---- Plots ----------------------------------------------------------
  const plotEntries = Object.entries(stat.plot || {})
    .filter(([k]) => k.endsWith("_plot"));
  if (plotEntries.length) {
    sections.push(_section("Plots", el("div", { class: "grid grid-cols-1 md:grid-cols-2 gap-4" },
      ...plotEntries.map(([key, relPath]) => {
        const name = String(relPath).split("/").pop();
        const ptype = key.replace(/_plot$/, "");
        return el("div", { class: "border border-slate-200 rounded overflow-hidden" },
          el("div", { class: "px-3 py-2 bg-slate-50 flex justify-between items-center" },
            el("span", { class: "text-sm font-medium" }, ptype),
            el("a", { href: url(name), download: name, class: "text-xs text-blue-600 underline" }, "Download"),
          ),
          el("embed", { src: url(name), type: "application/pdf",
            class: "w-full", style: "height:480px" }),
        );
      })
    )));
  }

  // ---- Report ---------------------------------------------------------
  const reportFile = byName("_results.pdf");
  if (reportFile) {
    sections.push(_section("Report",
      el("div", { class: "border border-slate-200 rounded overflow-hidden" },
        el("div", { class: "px-3 py-2 bg-slate-50 flex justify-between items-center" },
          el("span", { class: "text-sm font-medium" }, reportFile.name),
          el("a", { href: url(reportFile.name), download: reportFile.name,
            class: "text-xs text-blue-600 underline" }, "Download PDF"),
        ),
        el("embed", { src: url(reportFile.name), type: "application/pdf",
          class: "w-full", style: "height:600px" }),
      )
    ));
  }

  // ---- Tables (inline LaTeX) -----------------------------------------
  const tableEntries = Object.entries(stat.table || {}).filter(([k]) => k.endsWith("_tex"));
  if (tableEntries.length) {
    sections.push(_section("Tables (LaTeX)",
      el("div", { class: "space-y-2" },
        ...tableEntries.map(([key, relPath]) => {
          const name = String(relPath).split("/").pop();
          const tabtype = key.replace(/_tex$/, "");
          const block = el("pre", {
            class: "bg-slate-100 p-3 rounded text-xs whitespace-pre-wrap font-mono max-h-72 overflow-y-auto",
          }, "Loading…");
          fetch(url(name)).then(r => r.text()).then(t => { block.textContent = t; });
          return el("details", { class: "border border-slate-200 rounded" },
            el("summary", { class: "cursor-pointer px-3 py-2 bg-slate-50 text-sm flex justify-between items-center" },
              el("span", { class: "font-medium" }, tabtype),
              el("a", { href: url(name), download: name,
                class: "text-xs text-blue-600 underline",
                onclick: (e) => e.stopPropagation() }, "Download"),
            ),
            block,
          );
        })
      )
    ));
  }

  // ---- IC compare summary --------------------------------------------
  if (stat.ic_compare && stat.ic_compare.ic) {
    sections.push(_section("Information criteria",
      el("pre", { class: "bg-slate-100 p-3 rounded text-xs whitespace-pre-wrap font-mono overflow-x-auto" },
        String(stat.ic_compare.ic))));
  }

  // ---- All output files -----------------------------------------------
  if (files.length) {
    sections.push(_section("All output files",
      buildFilesTable(runId, files)
    ));
  }

  if (sections.length === 0) {
    return el("p", { class: "text-sm text-slate-400" },
      "Results will appear here once you run a step.");
  }
  return el("div", { class: "space-y-4" }, ...sections);
}

function _section(title, body) {
  return el("section", { class: "bg-white rounded p-4 shadow-sm" },
    el("h2", { class: "text-lg font-semibold mb-3" }, title),
    body,
  );
}

function _actionButton(step, label, className) {
  // Wraps the step action in a button that shows immediate feedback —
  // disables itself, swaps the label for a spinner, then re-renders the
  // step list so a successful run flips to "✓ done" without waiting for
  // the 3 s polling tick.
  const btn = el("button", { class: className }, label);
  btn.onclick = async () => {
    btn.disabled = true;
    btn.classList.add("opacity-50", "cursor-not-allowed");
    const orig = btn.textContent;
    btn.replaceChildren(el("span", { class: "spinner mr-1", style: "vertical-align:-2px" }), "running…");
    try {
      const result = await Promise.resolve(step.action());
      // mcmc/ns return early — they navigate to a job page.
      // Sync steps return a {ok, status, body} shape from runStep.
      if (result && result.ok === false) {
        _showStepError(result);
      }
    } finally {
      btn.disabled = false;
      btn.classList.remove("opacity-50", "cursor-not-allowed");
      btn.textContent = orig;
      window.__rv_refresh?.();
    }
  };
  return btn;
}

function _showStepError(resp) {
  // Surface errors above the fold so the user can't miss them.
  const banner = el("div", {
    class: "fixed top-4 right-4 max-w-md bg-rose-50 border border-rose-300 text-rose-800 rounded shadow-lg p-3 text-sm z-50",
  },
    el("div", { class: "flex justify-between items-start gap-3" },
      el("div", {},
        el("div", { class: "font-medium mb-1" }, `Step failed (${resp.status})`),
        el("pre", { class: "text-xs whitespace-pre-wrap font-mono" },
          JSON.stringify(resp.body?.detail || resp.body, null, 2)),
      ),
      el("button", { class: "text-rose-600", onclick: () => banner.remove() }, "✕"),
    ),
  );
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 12000);
}

async function runStep(runId, step, body) {
  window.__rv_log?.(`POST /runs/${runId}/${step}…`);
  const resp = await api.post(`/runs/${runId}/${step}`, body);
  if (resp.ok) {
    window.__rv_log?.(`OK ${resp.status}: ${JSON.stringify(resp.body, null, 2)}`);
  } else {
    window.__rv_log?.(`FAIL ${resp.status}: ${JSON.stringify(resp.body)}`);
  }
  return resp;
}

function mcmcModal(runId) {
  const overlay = el("div", { class: "fixed inset-0 bg-black/40 flex items-center justify-center z-50" });
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  const fields = [
    ["nsteps", 200],
    ["nwalkers", 30],
    ["ensembles", 2],
    ["thin", 1],
  ];
  const inputs = {};
  const body = el("div", { class: "grid grid-cols-2 gap-3" },
    ...fields.flatMap(([k, def]) => {
      const i = el("input", { type: "number", value: String(def), class: "border rounded px-2 py-1 text-sm w-full" });
      inputs[k] = i;
      return [el("label", { class: "self-center text-sm text-slate-700" }, k), i];
    }),
    el("label", { class: "self-center text-sm text-slate-700" }, "serial"),
    (() => { const cb = el("input", { type: "checkbox" }); cb.checked = true; inputs.serial = cb; return cb; })(),
  );
  const card = el("div", { class: "bg-white rounded p-6 w-96 shadow-lg" },
    el("h2", { class: "text-lg font-semibold mb-3" }, "Start MCMC"),
    body,
    el("div", { class: "flex justify-end gap-2 mt-4" },
      el("button", { class: "px-3 py-1 text-sm border rounded", onclick: close }, "Cancel"),
      el("button", {
        class: "px-3 py-1 text-sm bg-blue-600 text-white rounded",
        onclick: async () => {
          const payload = {};
          for (const [k, i] of Object.entries(inputs)) {
            payload[k] = i.type === "checkbox" ? i.checked : Number(i.value);
          }
          const resp = await api.post(`/runs/${runId}/mcmc`, payload);
          close();
          if (resp.ok) location.hash = `#/runs/${runId}/jobs/${resp.body.job_id}`;
          else window.__rv_log?.(`FAIL ${resp.status}: ${JSON.stringify(resp.body)}`);
        },
      }, "Start"),
    )
  );
  overlay.appendChild(card);
  document.body.appendChild(overlay);
}

function nsModal(runId) {
  const overlay = el("div", { class: "fixed inset-0 bg-black/40 flex items-center justify-center z-50" });
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  const sampler = el("select", { class: "border rounded px-2 py-1 text-sm w-full" },
    el("option", { value: "ultranest" }, "ultranest"),
    el("option", { value: "dynesty" }, "dynesty"),
    el("option", { value: "multinest" }, "multinest"),
  );
  const card = el("div", { class: "bg-white rounded p-6 w-96 shadow-lg" },
    el("h2", { class: "text-lg font-semibold mb-3" }, "Start nested sampling"),
    el("label", { class: "block text-sm text-slate-700 mb-1" }, "sampler"),
    sampler,
    el("div", { class: "flex justify-end gap-2 mt-4" },
      el("button", { class: "px-3 py-1 text-sm border rounded", onclick: close }, "Cancel"),
      el("button", {
        class: "px-3 py-1 text-sm bg-blue-600 text-white rounded",
        onclick: async () => {
          const resp = await api.post(`/runs/${runId}/ns`, { sampler: sampler.value });
          close();
          if (resp.ok) location.hash = `#/runs/${runId}/jobs/${resp.body.job_id}`;
          else window.__rv_log?.(`FAIL ${resp.status}: ${JSON.stringify(resp.body)}`);
        },
      }, "Start"),
    )
  );
  overlay.appendChild(card);
  document.body.appendChild(overlay);
}

async function viewJob(runId, jobId) {
  clear();
  const titleRow = el("div", { class: "flex items-center justify-between mb-4" },
    el("h1", { class: "text-2xl font-semibold mono" }, jobId),
    el("a", { href: `#/runs/${runId}`, class: "text-sm text-blue-600 underline" }, "← run"),
  );
  $app.appendChild(titleRow);
  const stateBadge = el("div", { class: "mb-3" });
  const grid = el("div", { class: "grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm bg-white rounded p-4 shadow-sm" });
  const bar = el("div", { class: "h-3 rounded progress-bar mt-3" });
  const cancelBtn = el("button", {
    class: "mt-4 bg-rose-600 text-white px-3 py-1 rounded text-sm hover:bg-rose-700 disabled:opacity-50",
    onclick: async () => {
      cancelBtn.disabled = true;
      await api.del(`/jobs/${jobId}`);
    },
  }, "Cancel");
  $app.append(stateBadge, grid, bar, cancelBtn);

  let stop = false;
  async function tick() {
    if (stop) return;
    const j = await api.get(`/jobs/${jobId}`);
    stateBadge.replaceChildren(badge(j.state));
    const p = j.progress || {};
    grid.replaceChildren(
      ...Object.entries({
        kind: j.kind,
        submitted: fmtTime(j.submitted_at),
        started: fmtTime(j.started_at),
        finished: fmtTime(j.finished_at),
        nsteps: p.nsteps_complete ?? "—",
        totsteps: p.totsteps ?? "—",
        rate: p.rate ?? "—",
        ar: p.ar ?? "—",
        minafactor: p.minafactor ?? "—",
        maxarchange: p.maxarchange ?? "—",
        mintz: p.mintz ?? "—",
        maxgr: p.maxgr ?? "—",
      }).flatMap(([k, v]) => [
        el("div", { class: "text-slate-500" }, k),
        el("div", { class: "mono" }, String(v)),
      ])
    );
    bar.style.setProperty("--pct", `${Math.min(100, Number(p.pcomplete) || 0)}%`);
    if (["succeeded", "failed", "cancelled"].includes(j.state)) {
      stop = true;
      cancelBtn.disabled = true;
      // pcomplete may stop reporting before the last batch of steps because
      // convergence is reached early. Snap to 100% on success so the bar
      // matches the badge.
      if (j.state === "succeeded") bar.style.setProperty("--pct", "100%");
      if (j.error) {
        $app.appendChild(el("pre", { class: "mt-3 bg-rose-50 text-rose-800 p-3 rounded text-xs whitespace-pre-wrap" }, j.error));
      }
      // Offer a clear way back to the run, where the new step status will
      // reflect the just-completed work.
      $app.appendChild(el("div", { class: "mt-4" },
        el("a", { href: `#/runs/${runId}`, class: "px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700" },
          "← back to run")
      ));
      return;
    }
    setTimeout(tick, 2000);
  }
  tick();
}

async function viewFiles(runId) {
  clear();
  $app.appendChild(el("h1", { class: "text-2xl font-semibold mb-4" }, "Files"));
  const list = await api.get(`/runs/${runId}/files`);
  if (!Array.isArray(list) || list.length === 0) {
    $app.appendChild(el("p", { class: "text-slate-500" }, "No files."));
    return;
  }
  $app.appendChild(buildFilesTable(runId, list, { compact: false }));
}

// ---- Files table + viewer (shared between run page + Files page) ------

const _TEXT_EXTS = new Set([
  "json", "py", "stat", "tex", "csv", "log", "txt", "ini", "yaml", "yml", "md",
]);
const _PDF_EXTS = new Set(["pdf"]);

function _ext(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

function buildFilesTable(runId, files, opts = {}) {
  const compact = opts.compact !== false;
  const cellPad = compact ? "py-1" : "px-4 py-2";
  const fileUrl = (name) => `/runs/${runId}/files/${name}`;

  const header = el("thead", { class: compact ? "text-left text-xs text-slate-500" : "bg-slate-100 text-left" },
    el("tr", {},
      el("th", { class: cellPad }, "name"),
      el("th", { class: cellPad }, "size"),
      el("th", { class: cellPad }, "modified"),
      el("th", { class: cellPad + " text-right" }, ""),
    ),
  );
  const tbody = el("tbody", {});
  for (const f of files) {
    const ext = _ext(f.name);
    const actions = [];
    if (_TEXT_EXTS.has(ext) || _PDF_EXTS.has(ext)) {
      actions.push(el("a", {
        class: "text-xs text-blue-600 underline",
        href: "#",
        onclick: (e) => {
          e.preventDefault();
          openFileViewer(fileUrl(f.name), f.name, ext);
        },
      }, "view"));
    }
    actions.push(el("a", {
      class: "text-xs text-blue-600 underline",
      href: fileUrl(f.name),
      download: f.name,
    }, "download"));

    tbody.appendChild(el("tr", { class: "border-t border-slate-100" },
      el("td", { class: cellPad + " mono text-xs" }, f.name),
      el("td", { class: cellPad + " text-xs" }, fmtBytes(f.size)),
      el("td", { class: cellPad + " text-xs text-slate-500" }, fmtTime(f.mtime)),
      el("td", { class: cellPad + " text-right" },
        el("div", { class: "flex justify-end gap-3" }, ...actions)),
    ));
  }
  return el("table", {
    class: compact
      ? "w-full text-sm"
      : "w-full text-sm bg-white rounded overflow-hidden shadow-sm",
  }, header, tbody);
}

async function openFileViewer(url, name, ext) {
  const overlay = el("div", { class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-6" });
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  const card = el("div", { class: "bg-white rounded shadow-xl w-full max-w-4xl max-h-[85vh] flex flex-col" });
  const header = el("div", { class: "px-4 py-2 border-b flex justify-between items-center" },
    el("span", { class: "font-medium text-sm mono" }, name),
    el("div", { class: "flex gap-3 text-sm" },
      el("a", { href: url, download: name, class: "text-blue-600 underline" }, "Download"),
      el("button", { class: "text-slate-500 hover:text-slate-800", onclick: close }, "Close ✕"),
    ),
  );
  const body = el("div", { class: "flex-1 overflow-auto" }, el("div", { class: "p-4 text-slate-400" },
    el("span", { class: "spinner" }), " loading…"));
  card.append(header, body);
  overlay.appendChild(card);
  document.body.appendChild(overlay);

  if (_PDF_EXTS.has(ext)) {
    body.replaceChildren(el("embed", {
      src: url, type: "application/pdf", class: "w-full h-full", style: "min-height:75vh",
    }));
    return;
  }

  try {
    const r = await fetch(url);
    const text = await r.text();
    if (ext === "json") {
      try {
        const parsed = JSON.parse(text);
        body.replaceChildren(el("pre", {
          class: "p-4 text-xs whitespace-pre-wrap font-mono",
        }, JSON.stringify(parsed, null, 2)));
        return;
      } catch { /* fall through to plain text */ }
    }
    if (ext === "csv") {
      body.replaceChildren(_renderCsvTable(text));
      return;
    }
    body.replaceChildren(el("pre", {
      class: "p-4 text-xs whitespace-pre-wrap font-mono",
    }, text));
  } catch (e) {
    body.replaceChildren(el("pre", { class: "p-4 text-xs text-rose-700" },
      "Failed to load: " + (e.message || String(e))));
  }
}

function _renderCsvTable(text) {
  // Render the first ~500 rows in a real table. Anything bigger gets the
  // raw-text fallback to keep the modal responsive.
  const lines = text.split(/\r?\n/);
  const MAX = 500;
  if (lines.length > MAX) {
    return el("pre", { class: "p-4 text-xs whitespace-pre-wrap font-mono" }, text);
  }
  const rows = lines.filter(Boolean).map(l => l.split(","));
  if (rows.length === 0) return el("pre", { class: "p-4 text-xs text-slate-400" }, "(empty)");
  const [head, ...body] = rows;
  return el("table", { class: "w-full text-xs font-mono" },
    el("thead", { class: "bg-slate-100 text-left" },
      el("tr", {}, ...head.map(h => el("th", { class: "px-2 py-1" }, h))),
    ),
    el("tbody", {},
      ...body.map(r => el("tr", { class: "border-t border-slate-100" },
        ...r.map(c => el("td", { class: "px-2 py-1" }, c)),
      )),
    ),
  );
}

// ---- router --------------------------------------------------------------

function route() {
  const hash = location.hash || "#/runs";
  const parts = hash.replace(/^#\//, "").split("/");
  // [runs]
  // [runs, new]
  // [runs, {id}]
  // [runs, {id}, files]
  // [runs, {id}, jobs, {jid}]
  if (parts[0] === "runs" && parts.length === 1) return viewRunsList();
  if (parts[0] === "runs" && parts[1] === "new") return viewNewRun();
  if (parts[0] === "runs" && parts.length === 2) return viewRunDetail(parts[1]);
  if (parts[0] === "runs" && parts[2] === "files") return viewFiles(parts[1]);
  if (parts[0] === "runs" && parts[2] === "jobs") return viewJob(parts[1], parts[3]);
  $app.replaceChildren(el("p", {}, "Unknown route ", el("code", {}, hash)));
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const v = await api.get("/version");
    if (v.api && $version) $version.textContent = `v${v.api}`;
  } catch {}
  route();
});

// ---- keyboard shortcuts --------------------------------------------------

let lastKey = null;
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "?") {
    alert("Shortcuts:\n  g r — runs list\n  g n — new run\n  ?    — this help");
    return;
  }
  if (lastKey === "g" && e.key === "r") { location.hash = "#/runs"; lastKey = null; return; }
  if (lastKey === "g" && e.key === "n") { location.hash = "#/runs/new"; lastKey = null; return; }
  lastKey = e.key === "g" ? "g" : null;
});
