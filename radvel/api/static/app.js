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

function viewNewRun() {
  clear();
  $app.appendChild(el("h1", { class: "text-2xl font-semibold mb-4" }, "Create a run"));
  const ta = el("textarea", {
    class: "w-full h-96 font-mono text-xs p-3 border border-slate-300 rounded focus:outline-none focus:border-blue-500",
    placeholder: "Paste a setup JSON payload here…",
  });
  const status = el("div", { class: "mt-3 text-sm text-slate-600" });
  const submit = el("button", {
    class: "bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm disabled:opacity-50",
    onclick: async () => {
      submit.disabled = true;
      status.replaceChildren(el("span", { class: "spinner" }), " creating run…");
      let payload;
      try { payload = JSON.parse(ta.value); }
      catch (e) {
        status.textContent = `Invalid JSON: ${e.message}`;
        submit.disabled = false;
        return;
      }
      const resp = await api.post("/runs", payload);
      submit.disabled = false;
      if (resp.ok) {
        location.hash = `#/runs/${resp.body.run_id}`;
      } else {
        status.replaceChildren(el("pre", { class: "text-rose-700 whitespace-pre-wrap" }, JSON.stringify(resp.body, null, 2)));
      }
    },
  }, "Create run");
  const example = el("button", {
    class: "ml-2 px-3 py-2 text-sm border border-slate-300 rounded hover:bg-slate-100",
    onclick: () => { ta.value = JSON.stringify(EXAMPLE_PAYLOAD, null, 2); },
  }, "Load example");
  $app.appendChild(ta);
  $app.appendChild(el("div", { class: "mt-3" }, submit, example));
  $app.appendChild(status);
}

async function viewRunDetail(runId) {
  clear();
  $app.appendChild(el("div", { class: "text-slate-400" }, el("span", { class: "spinner" }), " loading…"));
  const run = await api.get(`/runs/${runId}`);
  clear();
  if (run.detail) {
    $app.appendChild(el("p", { class: "text-rose-700" }, `Error: ${JSON.stringify(run.detail)}`));
    return;
  }
  const stat = run.stat || {};
  const has = (k) => stat[k] && stat[k].run === "True";

  $app.appendChild(el("div", { class: "flex justify-between items-baseline mb-4" },
    el("h1", { class: "text-2xl font-semibold mono" }, runId),
    el("a", { href: `#/runs/${runId}/files`, class: "text-sm text-blue-600 underline" }, "Files →")
  ));
  $app.appendChild(el("div", { class: "bg-white rounded p-4 mb-4 shadow-sm" },
    el("div", { class: "grid grid-cols-2 gap-2 text-sm" },
      el("div", { class: "text-slate-500" }, "starname"),
      el("div", {}, run.starname || "—"),
      el("div", { class: "text-slate-500" }, "nplanets"),
      el("div", {}, String(run.nplanets ?? "—")),
      el("div", { class: "text-slate-500" }, "outputdir"),
      el("div", { class: "mono text-xs truncate" }, run.outputdir || ""),
      el("div", { class: "text-slate-500" }, "active job"),
      el("div", {}, run.active_job
        ? el("a", { href: `#/runs/${runId}/jobs/${run.active_job.job_id}`, class: "text-blue-600 underline mono" }, run.active_job.job_id)
        : "—"),
    )
  ));

  const stepBtn = (label, onclick, enabled = true, hint) => el("button", {
    class: `px-3 py-2 rounded text-sm ${enabled ? "bg-blue-600 text-white hover:bg-blue-700" : "bg-slate-200 text-slate-400 cursor-not-allowed"}`,
    disabled: enabled ? null : true,
    title: hint || "",
    onclick,
  }, label);

  const buttons = el("div", { class: "flex flex-wrap gap-2 mb-4" },
    stepBtn("Fit", async () => runStep(runId, "fit", {})),
    stepBtn("MCMC", () => mcmcModal(runId), has("fit"), "Fit must succeed first"),
    stepBtn("NS", () => nsModal(runId), has("fit"), "Fit must succeed first"),
    stepBtn("Derive", async () => runStep(runId, "derive", {}), has("mcmc") || has("ns"), "Run MCMC or NS first"),
    stepBtn("IC compare", async () => runStep(runId, "ic", { types: ["e"], simple: true }), has("fit"), "Fit first"),
    stepBtn("Tables", async () => runStep(runId, "tables", { types: ["params", "priors", "rv"] }), has("fit")),
    stepBtn("Plots: rv", async () => runStep(runId, "plots", { types: ["rv"] }), has("fit")),
    stepBtn("Plots: corner", async () => runStep(runId, "plots", { types: ["corner"] }), has("mcmc") || has("ns"), "Run MCMC/NS first"),
    stepBtn("Report", async () => runStep(runId, "report", {}), has("mcmc") || has("ns"), "Run MCMC/NS first"),
  );
  $app.appendChild(buttons);
  const log = el("pre", { class: "bg-slate-900 text-slate-100 p-3 rounded text-xs overflow-x-auto whitespace-pre-wrap min-h-[6rem]" }, "ready.\n");
  $app.appendChild(log);

  window.__rv_log = (line) => { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; };
}

async function runStep(runId, step, body) {
  window.__rv_log?.(`POST /runs/${runId}/${step}…`);
  const resp = await api.post(`/runs/${runId}/${step}`, body);
  if (resp.ok) {
    window.__rv_log?.(`OK ${resp.status}: ${JSON.stringify(resp.body, null, 2)}`);
  } else {
    window.__rv_log?.(`FAIL ${resp.status}: ${JSON.stringify(resp.body)}`);
  }
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
      if (j.error) {
        $app.appendChild(el("pre", { class: "mt-3 bg-rose-50 text-rose-800 p-3 rounded text-xs whitespace-pre-wrap" }, j.error));
      }
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
  if (!Array.isArray(list)) {
    $app.appendChild(el("p", { class: "text-slate-500" }, "No files."));
    return;
  }
  const table = el("table", { class: "w-full text-sm bg-white rounded overflow-hidden shadow-sm" });
  table.appendChild(el("thead", { class: "bg-slate-100 text-left" },
    el("tr", {},
      el("th", { class: "px-4 py-2" }, "name"),
      el("th", { class: "px-4 py-2" }, "size"),
      el("th", { class: "px-4 py-2" }, "modified"),
      el("th", { class: "px-4 py-2" }, ""),
    )
  ));
  const tbody = el("tbody", {});
  for (const f of list) {
    tbody.appendChild(el("tr", { class: "border-t border-slate-100" },
      el("td", { class: "px-4 py-2 mono" }, f.name),
      el("td", { class: "px-4 py-2" }, fmtBytes(f.size)),
      el("td", { class: "px-4 py-2 text-slate-500" }, fmtTime(f.mtime)),
      el("td", { class: "px-4 py-2 text-right" },
        el("a", { class: "text-blue-600 underline", href: `/runs/${runId}/files/${f.name}` }, "download")
      ),
    ));
  }
  table.appendChild(tbody);
  $app.appendChild(table);
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
