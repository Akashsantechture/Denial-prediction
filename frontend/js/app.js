/**
 * js/app.js
 * ---------
 * Main application controller — wired to predictionapi.py
 *
 * API response shape (predictionapi.py POST /predict):
 * {
 *   claim_id,
 *   claim_summary: {
 *     claim_denial_probability_pct,  claim_risk_level,
 *     highest_activity_risk,         average_activity_risk,
 *     activity_count
 *   },
 *   predictions: [{
 *     activity_code, cpt_category, denial_probability,
 *     predicted_denial, top_drivers, recommendation
 *   }]
 * }
 */

(() => {
  // ------------------------------------------------------------------ //
  // Module state
  // ------------------------------------------------------------------ //
  let _lastPayload = null;
  let _lastResult  = null;

  // ------------------------------------------------------------------ //
  // Tab routing
  // ------------------------------------------------------------------ //
  function initTabs() {
    const buttons = document.querySelectorAll('.tab-btn');
    const panels  = document.querySelectorAll('.tab-panel');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        buttons.forEach(b => b.classList.remove('active'));
        panels.forEach(p  => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${target}`)?.classList.add('active');
        if (_lastResult && _lastPayload) {
          if (target === 'icd')          renderICDTab();
          if (target === 'interactions') renderInteractionsTab();
          if (target === 'shap')         renderSHAPTab(_lastPayload, _lastResult);
        }
      });
    });
  }

  // ------------------------------------------------------------------ //
  // Health check
  // ------------------------------------------------------------------ //
  async function checkHealth() {
    const badge = document.getElementById('health-badge');
    const epEl  = document.getElementById('health-endpoint');
    if (badge) { badge.className = 'health-badge loading'; badge.innerHTML = '<span class="health-dot"></span> Checking…'; }

    const h = await API.health();

    if (badge) {
      if (h.status === 'healthy' && h.model_loaded) {
        badge.className = 'health-badge online';
        badge.innerHTML = '<span class="health-dot"></span> Backend online · Model loaded';
      } else if (h.status === 'healthy') {
        badge.className = 'health-badge loading';
        badge.innerHTML = '<span class="health-dot"></span> Backend online · Model NOT loaded';
      } else {
        badge.className = 'health-badge offline';
        badge.innerHTML = `<span class="health-dot"></span> Backend ${h.status}`;
      }
    }
    if (epEl) epEl.textContent = API.PREDICT_URL;
  }

  // ------------------------------------------------------------------ //
  // Form collection
  // Builds the flat payload including activities[] array from form rows.
  // ------------------------------------------------------------------ //
  function collectPayload() {
    const g = id => document.getElementById(id);
    const v = id => g(id)?.value?.trim() ?? '';

    // Collect activity rows from the dynamic activity table
    const activityRows = document.querySelectorAll('.activity-row');
    const activities = Array.from(activityRows).map(row => ({
      activity_code:    row.querySelector('.act-code')?.value?.trim()   || '',
      activity_quantity: parseFloat(row.querySelector('.act-qty')?.value)  || 1,
      activity_gross:   parseFloat(row.querySelector('.act-gross')?.value) || 0,
      cpt_category:     row.querySelector('.act-cpt-cat')?.value?.trim() || 'Other',
    }));

    return {
      claim_id:             `CLM-${Date.now()}`,
      diagnosis_code:       v('f-diagnosis_code').toUpperCase(),
      icd_category:         API.deriveIcdCategory(v('f-diagnosis_code')),
      patient_age:          parseInt(v('f-patient_age'), 10),
      gender:               v('f-gender'),
      nationality:          v('f-nationality'),
      encounter_type:       v('f-encounter_type'),
      length_of_stay:       parseInt(v('f-length_of_stay'), 10),
      claim_gross:          parseFloat(v('f-claim_gross')),
      claim_net:            parseFloat(v('f-claim_net')),
      billing_lag_days:     parseInt(v('f-billing_lag_days'), 10),
      clinician_profession: v('f-clinician_profession'),
      clinician_category:   v('f-clinician_category'),
      facility_type:        v('f-facility_type'),
      payer_classification: v('f-payer_id') || 'UNKNOWN',
      payer_id:             v('f-payer_id'),
      insurance_plan_tier:  v('f-insurance_plan_tier'),
      activities,
    };
  }

  // ------------------------------------------------------------------ //
  // Spinner
  // ------------------------------------------------------------------ //
  function showSpinner(visible) {
    document.getElementById('spinner-overlay')?.classList.toggle('visible', visible);
  }

  // ------------------------------------------------------------------ //
  // Validation errors
  // ------------------------------------------------------------------ //
  function showValidationErrors(failure) {
    const block = document.getElementById('validation-block');
    if (!block) return;
    document.getElementById('val-title').textContent = 'Submission Could Not Be Processed';
    document.getElementById('val-msg').textContent   = failure.message ?? 'Please review the fields below.';
    const list = document.getElementById('val-error-list');
    list.innerHTML = (failure.errors || []).map(e => `
      <div class="val-error-item">
        <div class="val-error-field">Field: ${e.field ?? 'Unknown'}</div>
        <div class="val-error-issue">${e.issue ?? 'Invalid value'}</div>
        ${e.provided_value != null ? `<div class="val-error-value">Provided: <code>${e.provided_value}</code></div>` : ''}
      </div>`).join('');
    block.classList.add('visible');
    block.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function hideValidationErrors() {
    document.getElementById('validation-block')?.classList.remove('visible');
  }

  // ------------------------------------------------------------------ //
  // Reset result UI (called on error so stale results never persist)
  // ------------------------------------------------------------------ //
  function resetResultUI() {
    document.getElementById('overview-content')?.classList.add('hidden');
    document.getElementById('overview-placeholder')?.classList.remove('hidden');
    document.getElementById('shap-content')?.classList.add('hidden');
    document.getElementById('shap-placeholder')?.classList.remove('hidden');

    ['gauge-chart','shap-waterfall-chart','shap-scatter-chart','shap-dependence-chart']
      .forEach(id => { const el = document.getElementById(id); if (el) Plotly.purge(el); });

    const ab = document.getElementById('action-banner');
    if (ab) { ab.className = 'action-banner low'; ab.innerHTML = '<strong>—</strong>'; }
    const rb = document.getElementById('risk-badge');
    if (rb) { rb.className = 'risk-badge'; rb.textContent = '—'; }
    _setTableRows('patient-table', []);
    ['kpi-prob','kpi-shap','kpi-flags','kpi-revised'].forEach(id => _setKPI(id,'—','neutral','—'));
  }

  // ------------------------------------------------------------------ //
  // Verdict banner
  // ------------------------------------------------------------------ //
  function updateVerdict(result) {
    const banner = document.getElementById('verdict-banner');
    if (!banner) return;
    const pct   = result.claim_summary.claim_denial_probability_pct;
    const level = result.claim_summary.claim_risk_level.toLowerCase();
    const labels = {
      high:   `HIGH LIKELIHOOD — ${pct.toFixed(1)}% Claim Denial Score`,
      medium: `REVIEW REQUIRED — ${pct.toFixed(1)}% Claim Denial Score`,
      low:    `LOW LIKELIHOOD  — ${pct.toFixed(1)}% Claim Denial Score`,
    };
    banner.className    = `verdict-banner visible ${level}`;
    banner.textContent  = labels[level] ?? `${pct.toFixed(1)}% Denial Score`;
  }

  function clearVerdict() {
    const banner = document.getElementById('verdict-banner');
    if (banner) banner.className = 'verdict-banner';
  }

  // ------------------------------------------------------------------ //
  // Overview tab render
  // ------------------------------------------------------------------ //
  function renderOverview(payload, result) {
    const cs    = result.claim_summary;
    const preds = result.predictions ?? [];
    const pct   = cs.claim_denial_probability_pct;
    const level = cs.claim_risk_level.toLowerCase();

    // ── Claim Score Hero ─────────────────────────────────────────────
    const scoreEl = document.getElementById('claim-score-value');
    const levelEl = document.getElementById('claim-score-level');
    const barEl   = document.getElementById('claim-score-bar');
    if (scoreEl) scoreEl.textContent = `${pct.toFixed(1)}%`;
    if (levelEl) {
      levelEl.textContent  = cs.claim_risk_level;
      levelEl.className    = `claim-score-level ${level}`;
    }
    if (barEl) {
      barEl.style.width    = `${Math.min(pct, 100)}%`;
      barEl.className      = `claim-score-fill ${level}`;
    }

    // ── Claim Score KPI strip ─────────────────────────────────────────
    _setKPI('kpi-claim-score',   `${pct.toFixed(1)}%`,                                    level,   'Overall Claim Score');
    _setKPI('kpi-highest-act',   `${(cs.highest_activity_risk * 100).toFixed(1)}%`,        'neutral','Highest Activity');
    _setKPI('kpi-avg-act',       `${(cs.average_activity_risk * 100).toFixed(1)}%`,        'neutral','Average Activity');
    _setKPI('kpi-act-count',     String(cs.activity_count),                                'neutral','Activities Submitted');

    // ── Gauge ─────────────────────────────────────────────────────────
    Charts.gauge('gauge-chart', pct);

    // ── Risk badge ────────────────────────────────────────────────────
    const rb = document.getElementById('risk-badge');
    if (rb) { rb.className = `risk-badge ${level}`; rb.textContent = cs.claim_risk_level; }

    // ── Patient & Claim info table ────────────────────────────────────
    const fmt = v => parseFloat(v).toLocaleString('en', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    _setTableRows('patient-table', [
      ['Age / Gender',   `${payload.patient_age} yrs / ${_titleCase(payload.gender)}`],
      ['Nationality',    _titleCase(payload.nationality)],
      ['ICD Code',       payload.diagnosis_code],
      ['ICD Category',   API.deriveIcdCategory(payload.diagnosis_code)],
      ['Claim Gross',    `AED ${fmt(payload.claim_gross)}`],
      ['Claim Net',      `AED ${fmt(payload.claim_net)}`],
      ['LOS',            `${payload.length_of_stay} day${payload.length_of_stay !== 1 ? 's' : ''}`],
      ['Encounter',      payload.encounter_type],
      ['Payer',          payload.payer_id || payload.payer_classification],
      ['Billing Lag',    `${payload.billing_lag_days} day${payload.billing_lag_days !== 1 ? 's' : ''}`],
    ]);

    // ── Action banner ─────────────────────────────────────────────────
    const ab = document.getElementById('action-banner');
    if (ab) {
      // Use recommendation from highest-risk activity
      const topPred = [...preds].sort((a, b) => b.denial_probability - a.denial_probability)[0];
      const rec = topPred?.recommendation ?? '';
      ab.className  = `action-banner ${level}`;
      ab.innerHTML  = `<strong>${cs.claim_risk_level} Denial Likelihood (${pct.toFixed(1)}%)</strong>${rec}`;
    }

    // ── Activity-level predictions table ─────────────────────────────
    _renderActivityTable(preds, payload.activities ?? []);
  }

  // ── Activity predictions table ───────────────────────────────────────
  function _renderActivityTable(predictions, activities) {
    const tbody = document.getElementById('activity-predictions-body');
    if (!tbody) return;

    tbody.innerHTML = predictions.map((pred, idx) => {
      const pct      = (pred.denial_probability * 100).toFixed(1);
      const lvl      = pred.denial_probability >= 0.70 ? 'high'
                     : pred.denial_probability >= 0.40 ? 'medium' : 'low';
      const denied   = pred.predicted_denial;
      const grossAmt = activities[idx]?.activity_gross ?? '—';
      const qty      = activities[idx]?.activity_quantity ?? '—';

      // Top 3 drivers as inline pills
      const driverPills = (pred.top_drivers || []).slice(0, 3).map(d => {
        const cls = d.impact === 'increase_risk' ? 'pos' : 'neg';
        const sign = d.shap_value >= 0 ? '+' : '';
        return `<span class="shap-pill ${cls}" title="${d.feature}">${sign}${d.shap_value.toFixed(3)}</span>`;
      }).join(' ');

      return `
        <tr class="${denied ? 'row-denied' : ''}">
          <td><code>${pred.activity_code}</code></td>
          <td><span class="cpt-badge">${pred.cpt_category || '—'}</span></td>
          <td>${qty}</td>
          <td>${typeof grossAmt === 'number' ? `AED ${grossAmt.toLocaleString('en',{minimumFractionDigits:2})}` : grossAmt}</td>
          <td>
            <div class="act-prob-row">
              <span class="act-prob-val ${lvl}">${pct}%</span>
              <div class="act-prob-bar-wrap">
                <div class="act-prob-bar ${lvl}" style="width:${Math.min(parseFloat(pct),100)}%"></div>
              </div>
            </div>
          </td>
          <td><span class="denial-verdict ${denied ? 'denied' : 'approved'}">${denied ? '⚠ Likely Denied' : '✓ Likely Approved'}</span></td>
          <td class="driver-pills-cell">${driverPills || '<span class="text-muted">—</span>'}</td>
          <td class="rec-cell">${pred.recommendation || '—'}</td>
        </tr>`;
    }).join('');
  }

  // ------------------------------------------------------------------ //
  // SHAP Analysis tab render
  // ------------------------------------------------------------------ //
  function renderSHAPTab(payload, result) {
    const contributions = SHAP.fromApiResult(result);
    const baseVal       = SHAP.baseLogOdds(result);
    const finalVal      = SHAP.finalLogOdds(result);
    const netShap       = parseFloat((finalVal - baseVal).toFixed(4));
    const HIGH_THRESHOLD = 0.05;

    const pct       = result.claim_summary.claim_denial_probability_pct;
    const riskClass = pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low';

    _setKPI('kpi-prob',
      `${pct.toFixed(1)}%`, riskClass,
      pct > 42 ? '↑ Above population average' : '↓ Below population average'
    );
    _setKPI('kpi-shap',
      `${netShap >= 0 ? '+' : ''}${netShap.toFixed(3)}`,
      netShap > 0 ? 'high' : 'green',
      `Base: ${baseVal.toFixed(3)}`
    );

    const highFlags  = contributions.filter(c => Math.abs(c.value) >= HIGH_THRESHOLD);
    const flagLabels = highFlags.slice(0, 3).map(c => c.name.split(' ')[0]).join(' · ') || '—';
    _setKPI('kpi-flags', String(highFlags.length), 'neutral', flagLabels);

    const topVal   = contributions[0]?.value ?? 0;
    const revisedLO = finalVal - Math.abs(topVal);
    const revisedP  = Math.round(100 / (1 + Math.exp(-revisedLO)));
    _setKPI('kpi-revised', `~${Math.max(revisedP, 2)}%`, 'green', '↓ If top driver resolved');

    Charts.waterfall('shap-waterfall-chart', contributions, baseVal, finalVal);
    Charts.featureScatter('shap-scatter-chart', contributions);

    const claimNetFeature = contributions.find(c =>
      c.name.toLowerCase().includes('net') || c.name.toLowerCase().includes('claim')
    );
    Charts.dependence('shap-dependence-chart',
      payload.claim_net ?? payload.claim_gross ?? 250,
      claimNetFeature?.value ?? null
    );

    return contributions;
  }

  // ------------------------------------------------------------------ //
  // ICD & Interactions tab
  // ------------------------------------------------------------------ //
  function renderICDTab() {
    if (!_lastPayload) return;
    const activeGroup = ICD.groupFromCode(_lastPayload.diagnosis_code);
    const captEl = document.getElementById('icd-active-caption');
    if (captEl) captEl.textContent = `ICD ${_lastPayload.diagnosis_code} → ${activeGroup}. Matching rows highlighted.`;
    const filterSel = document.getElementById('icd-filter-group');
    if (filterSel && filterSel.value === 'All') {
      for (const opt of filterSel.options) {
        if (opt.value === activeGroup) { opt.selected = true; break; }
      }
    }
    ICD.renderTable(activeGroup, document.getElementById('icd-sort')?.value ?? 'meanShap', filterSel?.value ?? 'All');
    ICD.renderHeatmap(filterSel?.value ?? 'All');
  }

  function renderInteractionsTab() {
    if (!_lastPayload) return;
    const activeIcd = ICD.groupFromCode(_lastPayload.diagnosis_code);
    const activeCpt = Interactions.cptGroupFromCode((_lastPayload.activities?.[0]?.activity_code) ?? '');
    const cap = document.getElementById('interactions-caption');
    if (cap) cap.textContent = `ICD: ${activeIcd} | CPT: ${activeCpt}`;
    Interactions.renderPairsTable(activeIcd, activeCpt);
    Interactions.renderSHAPHeatmap();
    Interactions.renderDenialHeatmap();
  }

  // ------------------------------------------------------------------ //
  // Form submit
  // ------------------------------------------------------------------ //
  async function handleSubmit(e) {
    e.preventDefault();
    hideValidationErrors();
    showSpinner(true);
    Chat.clear();

    const payload = (window.__jsonModeActive?.())
      ? window.__jsonModePayload()
      : collectPayload();

    // Validate at least one activity present
    if (!payload.activities || payload.activities.length === 0) {
      showSpinner(false);
      showValidationErrors({ message: 'At least one activity is required.', errors: [] });
      return;
    }

    const result = await API.predict(payload);
    showSpinner(false);

    if (result.ok) {
      _lastPayload = payload;
      _lastResult  = result;

      updateVerdict(result);

      document.getElementById('overview-placeholder')?.classList.add('hidden');
      document.getElementById('overview-content')?.classList.remove('hidden');
      document.getElementById('shap-placeholder')?.classList.add('hidden');
      document.getElementById('shap-content')?.classList.remove('hidden');

      renderOverview(payload, result);
      const contributions = renderSHAPTab(payload, result);

      // Switch to overview tab
      document.querySelector('.tab-btn[data-tab="overview"]')?.click();
      document.querySelectorAll('.top-nav-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === 'overview');
      });

      Chat.init(payload, result, contributions);

      // Pre-populate drill-down tabs
      ICD.renderTable(ICD.groupFromCode(payload.diagnosis_code), 'meanShap', ICD.groupFromCode(payload.diagnosis_code));
      ICD.renderHeatmap(ICD.groupFromCode(payload.diagnosis_code));
      const firstCpt = payload.activities?.[0]?.activity_code ?? '';
      Interactions.renderPairsTable(ICD.groupFromCode(payload.diagnosis_code), Interactions.cptGroupFromCode(firstCpt));
      Interactions.renderSHAPHeatmap();
      Interactions.renderDenialHeatmap();

    } else if (result.kind === 'validation') {
      clearVerdict(); _lastPayload = null; _lastResult = null;
      resetResultUI(); showValidationErrors(result); Chat.clear();
    } else {
      clearVerdict(); _lastPayload = null; _lastResult = null;
      resetResultUI();
      alert(`Error: ${result.detail}`);
    }
  }

  // ------------------------------------------------------------------ //
  // Activity row management (add / remove rows in the form)
  // ------------------------------------------------------------------ //
  function _makeActivityRow(idx, defaults = {}) {
    const row = document.createElement('tr');
    row.className = 'activity-row';
    row.innerHTML = `
      <td><input class="act-code    field-inline" type="text"   value="${defaults.activity_code    ?? '99213'}"  placeholder="e.g. 99213" /></td>
      <td><input class="act-qty     field-inline" type="number" value="${defaults.activity_quantity ?? 1}"        min="1" max="99" /></td>
      <td><input class="act-gross   field-inline" type="number" value="${defaults.activity_gross    ?? 250}"      min="0" step="10" /></td>
      <td>
        <select class="act-cpt-cat field-inline">
          ${CPT_CATEGORIES.map(c => `<option value="${c}" ${c === (defaults.cpt_category ?? 'Medicine') ? 'selected' : ''}>${c}</option>`).join('')}
        </select>
      </td>
      <td><button type="button" class="btn-remove-act" title="Remove activity">✕</button></td>`;
    row.querySelector('.btn-remove-act').addEventListener('click', () => {
      const tbody = document.getElementById('activities-tbody');
      if (tbody && tbody.querySelectorAll('.activity-row').length > 1) {
        row.remove();
        _reindexActivityRows();
      }
    });
    return row;
  }

  function _reindexActivityRows() {
    document.querySelectorAll('.activity-row').forEach((row, i) => {
      const label = row.closest('table')?.previousElementSibling?.querySelector('.act-row-count');
      if (label) label.textContent = `${i + 1}`;
    });
  }

  const CPT_CATEGORIES = [
    'Evaluation and Management', 'Surgery', 'Radiology',
    'Pathology and Laboratory', 'Medicine', 'Anesthesia', 'Other',
  ];

  function initActivityManager() {
    const tbody  = document.getElementById('activities-tbody');
    const addBtn = document.getElementById('add-activity-btn');
    if (!tbody || !addBtn) return;

    // Start with one blank row
    tbody.appendChild(_makeActivityRow(0));

    addBtn.addEventListener('click', () => {
      const rows = tbody.querySelectorAll('.activity-row');
      tbody.appendChild(_makeActivityRow(rows.length));
    });
  }

  // ------------------------------------------------------------------ //
  // Chat input wiring
  // ------------------------------------------------------------------ //
  function initChat() {
    const btn   = document.getElementById('chat-send-btn');
    const input = document.getElementById('chat-input');
    btn?.addEventListener('click', () => {
      const msg = input?.value.trim();
      if (!msg) return;
      Chat.send(msg);
      if (input) input.value = '';
    });
    input?.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); btn?.click(); }
    });
    Chat.clear();
  }

  // ------------------------------------------------------------------ //
  // ICD controls
  // ------------------------------------------------------------------ //
  function initICDControls() {
    ICD.populateGroupFilter('icd-filter-group');
    document.getElementById('icd-filter-group')?.addEventListener('change', () => {
      const activeGroup = _lastPayload ? ICD.groupFromCode(_lastPayload.diagnosis_code) : null;
      ICD.renderTable(activeGroup, document.getElementById('icd-sort')?.value ?? 'meanShap',
        document.getElementById('icd-filter-group')?.value ?? 'All');
      ICD.renderHeatmap(document.getElementById('icd-filter-group')?.value ?? 'All');
    });
    document.getElementById('icd-sort')?.addEventListener('change', () => {
      const activeGroup = _lastPayload ? ICD.groupFromCode(_lastPayload.diagnosis_code) : null;
      ICD.renderTable(activeGroup, document.getElementById('icd-sort')?.value ?? 'meanShap',
        document.getElementById('icd-filter-group')?.value ?? 'All');
    });
    ICD.renderTable(null, 'meanShap', 'All');
    ICD.renderHeatmap('All');
    Interactions.renderPairsTable(null, null);
    Interactions.renderSHAPHeatmap();
    Interactions.renderDenialHeatmap();
  }

  // ------------------------------------------------------------------ //
  // Expander toggles
  // ------------------------------------------------------------------ //
  function initExpanders() {
    document.querySelectorAll('.expander-trigger').forEach(btn => {
      btn.addEventListener('click', () => {
        const body  = btn.nextElementSibling;
        const arrow = btn.querySelector('.expander-arrow');
        body?.classList.toggle('open');
        if (arrow) arrow.textContent = body?.classList.contains('open') ? '▲' : '▼';
      });
    });
  }

  // ------------------------------------------------------------------ //
  // DOM helpers
  // ------------------------------------------------------------------ //
  function _titleCase(s) { return s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : ''; }

  function _setTableRows(tableId, rows) {
    const table = document.getElementById(tableId);
    if (!table) return;
    table.innerHTML = rows.map(([label, value]) =>
      `<tr><td>${label}</td><td>${value}</td></tr>`).join('');
  }

  function _setKPI(cardId, value, colorClass, subText) {
    const card = document.getElementById(cardId);
    if (!card) return;
    const valEl = card.querySelector('.kpi-value');
    const subEl = card.querySelector('.kpi-sub');
    if (valEl) { valEl.textContent = value; valEl.className = `kpi-value ${colorClass}`; }
    if (subEl) subEl.textContent = subText;
  }

  // ------------------------------------------------------------------ //
  // Boot
  // ------------------------------------------------------------------ //
  document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    checkHealth();
    initActivityManager();
    initChat();
    initICDControls();
    initExpanders();
    document.getElementById('claim-form')?.addEventListener('submit', handleSubmit);
  });

})();
