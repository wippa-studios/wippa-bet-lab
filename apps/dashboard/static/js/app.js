/* ===================================================================
   Wippa Bet Lab — Model Builder Dashboard JS
   =================================================================== */

const API_BASE = '/api';
const state = {
  datasets: [],
  currentDataset: null,
  columns: [],
  preview: null,
  targetSuggestions: [],
  featureConfigs: [],
  modelHistory: [],
  lastTrainResult: null,
};

// ── Helpers ──────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

async function api(path, opts = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      ...opts,
    });
    if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
    return await res.json();
  } catch (e) {
    console.error(`API Error [${path}]:`, e);
    return null;
  }
}

function formatPct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : '—'; }
function formatNum(v, d = 3) { return v != null ? Number(v).toFixed(d) : '—'; }

// ── Navigation ──────────────────────────────────────────────────
$$('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.dataset.panel;
    if (panel === 'builder') {
      show($('#builder-panel'));
      hide($('#history-panel'));
    } else {
      hide($('#builder-panel'));
      show($('#history-panel'));
      loadModelHistory();
    }
  });
});

// ── Dataset Loading ─────────────────────────────────────────────
async function loadDatasets() {
  const data = await api('/datasets/');
  if (!data) return;
  state.datasets = data;
  const sel = $('#dataset-select');
  sel.innerHTML = '<option value="">— Choose dataset —</option>';
  data.forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds.id;
    opt.textContent = `${ds.name} (${ds.sport}) — ${ds.event_count} events`;
    sel.appendChild(opt);
  });
}

$('#dataset-select').addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) {
    state.currentDataset = null;
    hide($('#dataset-info'));
    hide($('#panel-columns'));
    hide($('#grid-config'));
    return;
  }

  const ds = state.datasets.find(d => d.id === id);
  state.currentDataset = ds;

  $('#chip-rows').textContent = `${ds.event_count} rows`;
  $('#chip-cols').textContent = `${ds.market_count || '—'} columns`;
  $('#chip-sport').textContent = ds.sport;
  show($('#dataset-info'));

  await loadColumns(id);
});

async function loadColumns(datasetId) {
  show($('#panel-columns'));
  show($('#grid-config'));
  await loadColumnData(datasetId);
}

async function loadColumnData(datasetId) {
  const tbody = $('#columns-tbody');
  tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">Loading columns...</td></tr>';

  const data = await api(`/models/datasets/${datasetId}/columns`);
  if (data && data.columns) {
    state.columns = data.columns;
    state.preview = data;
    renderColumns(data.columns);
    populateTargetDropdown(data.columns);
    populateFeatures(data.columns);
  } else {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">Connect to the API server to load column data. Start with: <code>uvicorn betlab.api:app --reload</code></td></tr>';
  }
}

function renderColumns(columns) {
  const tbody = $('#columns-tbody');
  const hideIds = $('#col-hide-ids').checked;
  const search = $('#col-search').value.toLowerCase();

  const filtered = columns.filter(col => {
    if (hideIds && col.dtype === 'identifier') return false;
    if (search && !col.name.toLowerCase().includes(search)) return false;
    return true;
  });

  tbody.innerHTML = filtered.map(col => {
    const missingClass = col.missing_pct > 30 ? 'high' : col.missing_pct > 10 ? 'mid' : 'low';
    const samples = (col.sample_values || []).slice(0, 3).map(v =>
      `<span class="sample-val">${String(v).substring(0, 15)}</span>`
    ).join('');

    const stats = col.dtype === 'numeric'
      ? `μ=${formatNum(col.mean, 2)} σ=[${formatNum(col.min, 1)}, ${formatNum(col.max, 1)}]`
      : `${col.unique_count} unique`;

    return `<tr>
      <td><code>${col.name}</code></td>
      <td><span class="type-badge ${col.dtype}">${col.dtype}</span></td>
      <td>
        <div class="missing-bar">
          <div class="missing-bar-track"><div class="missing-bar-fill ${missingClass}" style="width:${Math.min(col.missing_pct, 100)}%"></div></div>
          <span>${col.missing_pct}%</span>
        </div>
      </td>
      <td>${col.unique_count}</td>
      <td style="font-size:11px;color:var(--text-muted)">${stats}</td>
      <td>${samples}</td>
    </tr>`;
  }).join('');
}

// Column search and filter
$('#col-search').addEventListener('input', () => renderColumns(state.columns));
$('#col-hide-ids').addEventListener('change', () => renderColumns(state.columns));

// ── Target Selection ────────────────────────────────────────────
function populateTargetDropdown(columns) {
  const sel = $('#target-select');
  sel.innerHTML = '<option value="">— Select target —</option>';
  const suggestions = columns.filter(c =>
    c.dtype !== 'identifier' && c.unique_count <= 20
  );

  // Sort: suggested first
  suggestions.sort((a, b) => {
    const aName = a.name.toLowerCase();
    const bName = b.name.toLowerCase();
    const keywords = ['result', 'outcome', 'won', 'win', 'target', 'label', 'class'];
    const aMatch = keywords.some(k => aName.includes(k)) ? 0 : 1;
    const bMatch = keywords.some(k => bName.includes(k)) ? 0 : 1;
    return aMatch - bMatch;
  });

  suggestions.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col.name;
    const suggested = ['result', 'outcome', 'won', 'win', 'target', 'label'].some(k => col.name.toLowerCase().includes(k));
    opt.textContent = `${suggested ? '★ ' : ''}${col.name} (${col.dtype}, ${col.unique_count} vals)`;
    sel.appendChild(opt);
  });

  // Show suggestion chips
  const sugDiv = $('#target-suggestions');
  const topSuggestions = suggestions.filter(c =>
    ['result', 'outcome', 'won', 'win', 'target', 'label'].some(k => c.name.toLowerCase().includes(k))
  ).slice(0, 5);

  sugDiv.innerHTML = topSuggestions.map(c =>
    `<span class="target-suggestion" data-col="${c.name}">${c.name}</span>`
  ).join('');

  $$('.target-suggestion').forEach(chip => {
    chip.addEventListener('click', () => {
      $('#target-select').value = chip.dataset.col;
      $('#target-select').dispatchEvent(new Event('change'));
    });
  });
}

function populateFeatures(columns) {
  const target = $('#target-select').value;
  state.featureConfigs = columns
    .filter(c => c.name !== target && c.dtype !== 'identifier')
    .map(c => {
      const suggested = ['numeric', 'categorical', 'boolean'].includes(c.dtype) && c.missing_pct < 30;
      const warning = c.missing_pct > 30 ? `High missing: ${c.missing_pct}%` : null;
      return {
        column: c.name,
        data_type: c.dtype,
        missing_pct: c.missing_pct,
        unique_count: c.unique_count,
        suggested: suggested,
        warning: warning,
        selected: suggested,
      };
    });

  renderFeatureList();
}

function renderFeatureList() {
  const list = $('#feature-list');
  const selectedCount = state.featureConfigs.filter(f => f.selected).length;

  list.innerHTML = state.featureConfigs.map((f, i) => {
    const classes = ['feature-item'];
    if (f.suggested) classes.push('suggested');
    if (f.warning) classes.push('warning');

    const missingClass = f.missing_pct > 30 ? 'missing-val' : 'missing-val ok';
    const typeIcon = { numeric: '#', categorical: 'Aa', boolean: '0/1', datetime: '📅', text: 'T' }[f.data_type] || '?';

    return `<div class="${classes.join(' ')}">
      <input type="checkbox" data-idx="${i}" ${f.selected ? 'checked' : ''}>
      <span class="type-badge ${f.data_type}">${typeIcon}</span>
      <span class="feature-name">${f.column}</span>
      <div class="feature-meta">
        <span class="${missingClass}">∅ ${f.missing_pct}%</span>
        <span>${f.unique_count} unique</span>
        ${f.warning ? `<span style="color:var(--yellow)">⚠ ${f.warning}</span>` : ''}
      </div>
    </div>`;
  }).join('');

  // Bind checkboxes
  list.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      const idx = parseInt(cb.dataset.idx);
      state.featureConfigs[idx].selected = cb.checked;
    });
  });

  // Update train button
  updateTrainButton();
}

$('#target-select').addEventListener('change', (e) => {
  if (e.target.value && state.columns.length) {
    populateFeatures(state.columns);
  }
});

$('#btn-select-suggested').addEventListener('click', () => {
  state.featureConfigs.forEach(f => { f.selected = f.suggested; });
  renderFeatureList();
});

$('#btn-select-all').addEventListener('click', () => {
  state.featureConfigs.forEach(f => { f.selected = true; });
  renderFeatureList();
});

$('#btn-deselect-all').addEventListener('click', () => {
  state.featureConfigs.forEach(f => { f.selected = false; });
  renderFeatureList();
});

// ── Model Parameters ────────────────────────────────────────────
const MODEL_PARAMS = {
  logistic_regression: [
    { key: 'C', label: 'Regularization (C)', type: 'number', default: 1.0, min: 0.001, step: 0.1 },
    { key: 'max_iter', label: 'Max Iterations', type: 'number', default: 1000, min: 100 },
  ],
  random_forest: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 200, min: 10 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', default: 10, min: 1 },
    { key: 'min_samples_split', label: 'Min Samples Split', type: 'number', default: 5, min: 2 },
  ],
  gradient_boosting: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 200, min: 10 },
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.1, min: 0.001, step: 0.01 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', default: 5, min: 1 },
  ],
  extra_trees: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 200, min: 10 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', default: 10, min: 1 },
  ],
  xgboost: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 200, min: 10 },
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.1, min: 0.001, step: 0.01 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', default: 6, min: 1 },
  ],
  lightgbm: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 200, min: 10 },
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.1, min: 0.001, step: 0.01 },
    { key: 'num_leaves', label: 'Num Leaves', type: 'number', default: 31, min: 2 },
  ],
  dummy_baseline: [],
};

$('#model-type').addEventListener('change', renderModelParams);
function renderModelParams() {
  const type = $('#model-type').value;
  const params = MODEL_PARAMS[type] || [];
  const container = $('#model-params');

  if (params.length === 0) {
    container.innerHTML = '<p style="font-size:12px;color:var(--text-muted)">No configurable parameters</p>';
    return;
  }

  container.innerHTML = params.map(p => `
    <div class="form-row">
      <label>${p.label}</label>
      <input type="${p.type}" data-key="${p.key}" value="${p.default}"
        min="${p.min || ''}" step="${p.step || 1}" class="form-control form-control-sm">
    </div>
  `).join('');
}

// Initialize params
renderModelParams();

// ── Train Button State ──────────────────────────────────────────
function updateTrainButton() {
  const target = $('#target-select').value;
  const features = state.featureConfigs.filter(f => f.selected);
  const btn = $('#btn-train');
  btn.disabled = !target || features.length === 0;
}

// ── Training ────────────────────────────────────────────────────
$('#btn-train').addEventListener('click', startTraining);

async function startTraining() {
  const btn = $('#btn-train');
  const progress = $('#train-progress');
  const fill = $('#progress-fill');
  const text = $('#progress-text');
  const spinner = btn.querySelector('.btn-spinner');
  const btnText = btn.querySelector('.btn-text');

  btn.disabled = true;
  show(spinner);
  show(progress);
  btnText.textContent = 'Training...';
  fill.style.width = '10%';
  text.textContent = 'Preparing data...';

  // Collect parameters
  const modelType = $('#model-type').value;
  const params = {};
  $$('#model-params input').forEach(input => {
    const key = input.dataset.key;
    const val = input.type === 'number' ? parseFloat(input.value) : input.value;
    params[key] = val;
  });

  const selectedFeatures = state.featureConfigs
    .filter(f => f.selected)
    .map(f => f.column);

  const targetType = $('#target-type').value;

  // Build payload matching ModelBuilderState schema
  const payload = {
    state: {
      dataset_id: $('#dataset-select').value,
      dataset_version: '',
      target: {
        column: $('#target-select').value,
        type: targetType,
        positive_class: $('#positive-class').value || null,
      },
      features: selectedFeatures.map(col => {
        const cfg = state.featureConfigs.find(f => f.column === col);
        return {
          column: col,
          role: 'feature',
          data_type: cfg ? cfg.data_type : 'numeric',
          missing_pct: cfg ? cfg.missing_pct : 0,
          unique_count: cfg ? cfg.unique_count : 0,
        };
      }),
      model: {
        type: modelType,
        parameters: params,
      },
      preprocessing: {
        numeric_imputation: $('#prep-imputation').value,
        categorical_encoding: $('#prep-encoding').value,
        scaling: $('#prep-scaling').value,
      },
      validation: {
        method: $('#val-method').value,
        training_window: parseInt($('#val-train-window').value),
        test_window: parseInt($('#val-test-window').value),
      },
    },
  };

  // Simulate progress updates
  let progressVal = 10;
  const progressInterval = setInterval(() => {
    if (progressVal < 90) {
      progressVal += Math.random() * 15;
      fill.style.width = `${Math.min(progressVal, 90)}%`;
      const messages = [
        'Preprocessing features...',
        'Fitting model...',
        'Evaluating performance...',
        'Computing calibration...',
        'Calculating feature importance...',
      ];
      text.textContent = messages[Math.floor(Math.random() * messages.length)];
    }
  }, 800);

  try {
    const result = await api('/models/train', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    clearInterval(progressInterval);

    if (result) {
      fill.style.width = '100%';
      text.textContent = 'Complete!';
      state.lastTrainResult = result;
      state.modelHistory.unshift(result);
      renderResults(result);
      show($('#panel-results'));
      show($('#panel-export'));
      $('#btn-export-strategy').disabled = false;
      $('#btn-backtest').disabled = false;
    } else {
      fill.style.width = '0%';
      text.textContent = 'Training failed — check API connection';
      hide(progress);
    }
  } catch (err) {
    clearInterval(progressInterval);
    fill.style.width = '0%';
    text.textContent = `Error: ${err.message}`;
  }

  hide(spinner);
  btnText.textContent = 'Train Model';
  btn.disabled = false;
}

// ── Results Rendering ───────────────────────────────────────────
function renderResults(result) {
  if (!result) return;

  // Model ID and execution time
  $('#result-model-id').textContent = result.model_id ? result.model_id.substring(0, 12) + '...' : '';
  $('#result-exec-time').textContent = result.execution_time_ms ? `${result.execution_time_ms.toFixed(0)}ms` : '';

  // Metrics cards
  const metricsGrid = $('#metrics-grid');
  const metrics = result.metrics || {};
  const metricDefs = [
    { key: 'accuracy', label: 'Accuracy', format: formatPct },
    { key: 'balanced_accuracy', label: 'Balanced Acc', format: formatPct },
    { key: 'roc_auc', label: 'ROC AUC', format: formatPct },
    { key: 'precision', label: 'Precision', format: formatPct },
    { key: 'recall', label: 'Recall', format: formatPct },
    { key: 'f1', label: 'F1 Score', format: formatPct },
    { key: 'brier_score', label: 'Brier Score', format: formatNum },
    { key: 'log_loss', label: 'Log Loss', format: formatNum },
  ];

  metricsGrid.innerHTML = metricDefs
    .filter(m => metrics[m.key] != null)
    .map(m => {
      const val = metrics[m.key];
      let cls = '';
      if (m.key === 'brier_score' || m.key === 'log_loss') {
        cls = val < 0.3 ? 'good' : val < 0.5 ? 'warn' : 'bad';
      } else {
        cls = val >= 0.7 ? 'good' : val >= 0.5 ? 'warn' : 'bad';
      }
      return `<div class="metric-card ${cls}">
        <div class="metric-value">${m.format(val)}</div>
        <div class="metric-label">${m.label}</div>
      </div>`;
    }).join('');

  // Feature importance
  renderFeatureImportance(result.feature_importance || []);

  // Calibration chart
  renderCalibrationChart(result.calibration || {});

  // Confusion matrix (would need to be computed server-side)
  renderConfusionMatrixPlaceholder();

  // Warnings
  if (result.warnings && result.warnings.length > 0) {
    const warnDiv = $('#train-warnings');
    warnDiv.innerHTML = `<strong>Warnings:</strong><ul>${result.warnings.map(w => `<li>${w}</li>`).join('')}</ul>`;
    show(warnDiv);
  } else {
    hide($('#train-warnings'));
  }
}

function renderFeatureImportance(importance) {
  const container = $('#feature-bars');
  if (!importance.length) {
    container.innerHTML = '<p style="color:var(--text-muted);font-size:12px">No feature importance data</p>';
    return;
  }

  const maxImp = Math.max(...importance.map(f => f.importance));
  const top = importance.slice(0, 20);

  container.innerHTML = top.map(f => {
    const pct = maxImp > 0 ? (f.importance / maxImp) * 100 : 0;
    return `<div class="feature-bar-row">
      <span class="feature-bar-label" title="${f.feature}">${f.feature}</span>
      <div class="feature-bar-track">
        <div class="feature-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="feature-bar-value">${formatNum(f.importance, 4)}</span>
    </div>`;
  }).join('');
}

function renderCalibrationChart(calibration) {
  const canvas = $('#calibration-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const padding = 40;

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = '#1c2128';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 10; i++) {
    const x = padding + (i / 10) * (w - 2 * padding);
    const y = padding + (i / 10) * (h - 2 * padding);
    ctx.beginPath();
    ctx.moveTo(x, padding);
    ctx.lineTo(x, h - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(w - padding, y);
    ctx.stroke();
  }

  // Perfect calibration line
  ctx.strokeStyle = '#6e7681';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding, h - padding);
  ctx.lineTo(w - padding, padding);
  ctx.stroke();
  ctx.setLineDash([]);

  // Calibration bins
  const bins = calibration.bins || [];
  if (bins.length > 0) {
    ctx.fillStyle = '#58a6ff';
    bins.forEach(bin => {
      const x = padding + bin.predicted_prob * (w - 2 * padding);
      const y = h - padding - bin.actual_rate * (h - 2 * padding);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Connect dots
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    bins.forEach((bin, i) => {
      const x = padding + bin.predicted_prob * (w - 2 * padding);
      const y = h - padding - bin.actual_rate * (h - 2 * padding);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 10; i++) {
    const val = i / 10;
    const x = padding + val * (w - 2 * padding);
    ctx.fillText(val.toFixed(1), x, h - padding + 15);
  }
  ctx.textAlign = 'right';
  for (let i = 0; i <= 10; i++) {
    const val = i / 10;
    const y = h - padding - val * (h - 2 * padding);
    ctx.fillText(val.toFixed(1), padding - 8, y + 3);
  }

  // Axis labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Mean Predicted Probability', w / 2, h - 5);
  ctx.save();
  ctx.translate(12, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Fraction of Positives', 0, 0);
  ctx.restore();

  // Brier score
  if (calibration.brier_score != null) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Brier: ${calibration.brier_score.toFixed(4)}`, padding + 5, padding + 15);
  }
}

function renderConfusionMatrixPlaceholder() {
  const container = $('#confusion-matrix');
  container.innerHTML = `
    <table>
      <thead>
        <tr><th></th><th>Predicted Neg</th><th>Predicted Pos</th></tr>
      </thead>
      <tbody>
        <tr><th>Actual Neg</th><td class="tn">TN</td><td class="fp">FP</td></tr>
        <tr><th>Actual Pos</th><td class="fn">FN</td><td class="tp">TP</td></tr>
      </tbody>
    </table>
    <p style="font-size:11px;color:var(--text-muted);margin-top:8px;text-align:center">
      Confusion matrix values available after training via API
    </p>`;
}

// ── Strategy Export ─────────────────────────────────────────────
$('#btn-export-strategy').addEventListener('click', exportStrategy);

async function exportStrategy() {
  const modelId = state.lastTrainResult?.model_id;
  if (!modelId) return;

  const bettingConfig = {
    probability_threshold: parseFloat($('#export-threshold').value),
    minimum_edge: parseFloat($('#export-edge').value),
    minimum_odds: parseFloat($('#export-min-odds').value),
    maximum_odds: parseFloat($('#export-max-odds').value),
    market_side: $('#export-side').value,
    staking_method: $('#export-staking').value,
  };

  const strategy = await api(`/models/${modelId}/export-strategy`, {
    method: 'POST',
    body: JSON.stringify(bettingConfig),
  });

  if (strategy) {
    $('#export-json').textContent = JSON.stringify(strategy, null, 2);
    show($('#export-preview'));
  }
}

// ── Backtest Integration ────────────────────────────────────────
$('#btn-backtest').addEventListener('click', runBacktest);

async function runBacktest() {
  const modelId = state.lastTrainResult?.model_id;
  if (!modelId) return;

  const btn = $('#btn-backtest');
  btn.disabled = true;
  btn.textContent = 'Running...';

  const result = await api(`/models/${modelId}/backtest`, {
    method: 'POST',
    body: JSON.stringify({}),
  });

  btn.disabled = false;
  btn.textContent = 'Run Backtest';

  if (result) {
    alert(`Backtest complete! Status: ${result.status || 'unknown'}\n${result.message || ''}`);
  }
}

// ── Model History ───────────────────────────────────────────────
async function loadModelHistory() {
  const data = await api('/models/');
  if (!data) {
    // Use local history
    renderModelHistory(state.modelHistory);
    return;
  }
  state.modelHistory = data;
  renderModelHistory(data);
}

function renderModelHistory(models) {
  const tbody = $('#history-tbody');
  const empty = $('#history-empty');

  if (!models || models.length === 0) {
    tbody.innerHTML = '';
    show(empty);
    return;
  }

  hide(empty);
  tbody.innerHTML = models.map(m => {
    const metrics = m.metrics || {};
    return `<tr>
      <td><code>${(m.model_id || m.id || '').substring(0, 12)}</code></td>
      <td>${m.model_type || m.type || '—'}</td>
      <td>${m.target || (m.target_config && m.target_config.column) || '—'}</td>
      <td>${m.feature_count || (m.feature_columns && m.feature_columns.length) || '—'}</td>
      <td>${metrics.accuracy != null ? formatPct(metrics.accuracy) : '—'}</td>
      <td>${metrics.roc_auc != null ? formatPct(metrics.roc_auc) : '—'}</td>
      <td style="color:var(--text-muted)">${m.created_at || '—'}</td>
    </tr>`;
  }).join('');
}

// ── Refresh Buttons ─────────────────────────────────────────────
$('#btn-refresh-datasets').addEventListener('click', loadDatasets);
$('#btn-refresh-history').addEventListener('click', loadModelHistory);

// ── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadDatasets();
  renderModelHistory(state.modelHistory);
});
