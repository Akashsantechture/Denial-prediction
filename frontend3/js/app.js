/**
 * js/app.js
 * ---------
 * Main application controller.
 *
 * Responsibilities
 * ----------------
 * 1. Tab routing — activates the correct panel on click.
 * 2. Backend health check on load — updates sidebar badge.
 * 3. Form submission — collects payload, calls API, dispatches result.
 * 4. Result rendering — populates Overview, SHAP, ICD, Interactions tabs.
 * 5. State management — stores last payload + result in module-level vars
 *    so switching tabs re-renders without another API call.
 *
 * Depends on (load order in index.html):
 *   api.js → shap.js → charts.js → icd.js → interactions.js → chat.js → app.js
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
        panels.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(`tab-${target}`)?.classList.add('active');

        // Lazy-render on tab switch if result exists
        if (_lastResult && _lastPayload) {
          if (target === 'icd')          renderICDTab();
          if (target === 'interactions') renderInteractionsTab();
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
    if (badge) badge.className = 'health-badge loading';
    if (badge) badge.innerHTML = '<span class="health-dot"></span> Checking…';

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
  // ------------------------------------------------------------------ //
  function collectPayload() {
    const g = id => document.getElementById(id);
    return {
      diagnosis_code:       g('f-diagnosis_code')?.value.trim().toUpperCase(),
      diagnosis_type:       g('f-diagnosis_type')?.value,
      activity_code:        g('f-activity_code')?.value.trim(),
      activity_gross:       parseFloat(g('f-activity_gross')?.value),
      activity_quantity:    parseInt(g('f-activity_quantity')?.value, 10),
      claim_gross:          parseFloat(g('f-claim_gross')?.value),
      claim_net:            parseFloat(g('f-claim_net')?.value),
      patient_age:          parseInt(g('f-patient_age')?.value, 10),
      gender:               g('f-gender')?.value,
      nationality:          g('f-nationality')?.value,
      payer_id:             g('f-payer_id')?.value.trim(),
      insurance_plan_tier:  g('f-insurance_plan_tier')?.value,
      clinician_profession: g('f-clinician_profession')?.value.trim(),
      clinician_category:   g('f-clinician_category')?.value.trim(),
      facility_type:        g('f-facility_type')?.value,
      billing_lag_days:     parseInt(g('f-billing_lag_days')?.value, 10),
      length_of_stay:       parseInt(g('f-length_of_stay')?.value, 10),
      encounter_type:       g('f-encounter_type')?.value,
    };
  }

  // ------------------------------------------------------------------ //
  // Spinner
  // ------------------------------------------------------------------ //
  function showSpinner(visible) {
    document.getElementById('spinner-overlay')?.classList.toggle('visible', visible);
  }

  // ------------------------------------------------------------------ //
  // Validation error display
  // ------------------------------------------------------------------ //
  function showValidationErrors(failure) {
    const block = document.getElementById('validation-block');
    if (!block) return;
    document.getElementById('val-title').textContent = 'Submission Could Not Be Processed';
    document.getElementById('val-msg').textContent   = failure.message;
    const list = document.getElementById('val-error-list');
    list.innerHTML = failure.errors.map(e => `
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
  // Verdict banner
  // ------------------------------------------------------------------ //
  function updateVerdict(result) {
    const banner = document.getElementById('verdict-banner');
    if (!banner) return;
    const prob = result.denial_probability_pct;
    const riskLower = result.risk_level.toLowerCase();
    const labels = { high: `DENIED — ${prob.toFixed(0)}% probability`, medium: `REVIEW — ${prob.toFixed(0)}% probability`, low: `APPROVED — ${prob.toFixed(0)}% probability` };
    banner.className = `verdict-banner visible ${riskLower}`;
    banner.textContent = labels[riskLower] ?? `${prob.toFixed(0)}% probability`;
  }

  function clearVerdict() {
    const banner = document.getElementById('verdict-banner');
    if (banner) banner.className = 'verdict-banner';
  }

  // ------------------------------------------------------------------ //
  // Overview tab render
  // ------------------------------------------------------------------ //
  function renderOverview(payload, result) {
    // Gauge
    Charts.gauge('gauge-chart', result.denial_probability_pct);

    // Risk badge
    const rb = document.getElementById('risk-badge');
    if (rb) {
      const r = result.risk_level.toLowerCase();
      rb.className = `risk-badge ${r}`;
      rb.textContent = result.risk_level;
    }

    // Patient & Claim table
    const fmt = (v, digits = 2) => parseFloat(v).toLocaleString('en', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    _setTableRows('patient-table', [
      ['Age / Gender',   `${payload.patient_age} yrs / ${_titleCase(payload.gender)}`],
      ['Nationality',    _titleCase(payload.nationality)],
      ['ICD Code',       payload.diagnosis_code],
      ['CPT Code',       payload.activity_code],
      ['Claim Net',      `AED ${fmt(payload.claim_net ?? payload.claim_gross)}`],
      ['LOS',            `${payload.length_of_stay} day${payload.length_of_stay !== 1 ? 's' : ''}`],
      ['Encounter',      payload.encounter_type],
      ['Payer',          payload.payer_id],
      ['Plan Tier',      payload.insurance_plan_tier],
    ]);

    // Action banner
    const ab = document.getElementById('action-banner');
    if (ab) {
      ab.className = `action-banner ${result.risk_level.toLowerCase()}`;
      ab.innerHTML = `<strong>${result.risk_level} Risk (${result.denial_probability_pct.toFixed(1)}%)</strong>${result.action_recommendation}`;
    }

    // Denial reason bar
    Charts.denialReasonBar('denial-reason-chart');
  }

  // ------------------------------------------------------------------ //
  // SHAP Analysis tab render
  // ------------------------------------------------------------------ //
  function renderSHAPTab(payload, result) {
    const contributions = SHAP.estimate(payload, result);
    const baseLogOdds   = SHAP.BASE_LOG_ODDS;
    const finalLO       = SHAP.finalLogOdds(result.denial_probability_pct);
    const netShap       = parseFloat((finalLO - baseLogOdds).toFixed(3));
    const HIGH_THRESHOLD = 0.08;

    // KPI cards
    const prob = result.denial_probability_pct;
    const riskClass = prob >= 50 ? 'high' : prob >= 35 ? 'medium' : 'low';

    _setKPI('kpi-prob',      `${prob.toFixed(1)}%`,   riskClass, 'vs 42% base rate');
    _setKPI('kpi-shap',      `${netShap >= 0 ? '+' : ''}${netShap}`, netShap > 0 ? 'high' : 'green', `Base: ${baseLogOdds}`);

    const highFlags = contributions.filter(c => Math.abs(c.value) >= HIGH_THRESHOLD);
    const flagLabel = highFlags.slice(0, 3).map(c => c.name.split(' ')[0]).join(' · ') || '—';
    _setKPI('kpi-flags',     String(highFlags.length), 'neutral', flagLabel);

    const revised = Math.max(prob - Math.abs(contributions[0]?.value ?? 0) * 30, 5).toFixed(0);
    _setKPI('kpi-revised',   `~${revised}%`,  'green', '↓ If top flag fixed');

    // Charts
    Charts.waterfall('shap-waterfall-chart', contributions, baseLogOdds, finalLO);
    Charts.featureScatter('shap-scatter-chart', contributions);
    Charts.dependence('shap-dependence-chart', payload.claim_net ?? payload.claim_gross ?? 250);

    return contributions;  // returned so chat can use them
  }

  // ------------------------------------------------------------------ //
  // ICD Drill-Down tab
  // ------------------------------------------------------------------ //
  function renderICDTab() {
    if (!_lastPayload) return;
    const activeGroup = ICD.groupFromCode(_lastPayload.diagnosis_code);
    document.getElementById('icd-active-caption')&&
      (document.getElementById('icd-active-caption').textContent =
        `ICD \`${_lastPayload.diagnosis_code}\` → ${activeGroup}. Matching rows highlighted.`);
    const filterSel = document.getElementById('icd-filter-group');
    if (filterSel && filterSel.value === 'All') {
      // Pre-select active group
      for (const opt of filterSel.options) {
        if (opt.value === activeGroup) { opt.selected = true; break; }
      }
    }
    const filterGroup = filterSel?.value ?? 'All';
    const sortSel     = document.getElementById('icd-sort')?.value ?? 'meanShap';
    ICD.renderTable(activeGroup, sortSel, filterGroup);
    ICD.renderHeatmap(filterGroup);
  }

  // ------------------------------------------------------------------ //
  // Interactions tab
  // ------------------------------------------------------------------ //
  function renderInteractionsTab() {
    if (!_lastPayload) return;
    const activeIcd = ICD.groupFromCode(_lastPayload.diagnosis_code);
    const activeCpt = Interactions.cptGroupFromCode(_lastPayload.activity_code);
    const cap = document.getElementById('interactions-caption');
    if (cap) cap.textContent = `ICD: ${activeIcd} | CPT: ${activeCpt}`;
    Interactions.renderPairsTable(activeIcd, activeCpt);
    Interactions.renderSHAPHeatmap();
    Interactions.renderDenialHeatmap();
  }

  // ------------------------------------------------------------------ //
  // Form submit handler
  // ------------------------------------------------------------------ //
  async function handleSubmit(e) {
    e.preventDefault();
    hideValidationErrors();
    showSpinner(true);
    Chat.clear();

    // If JSON mode is active, read payload from the textarea instead of form fields
    const payload = (window.__jsonModeActive && window.__jsonModeActive())
      ? window.__jsonModePayload()
      : collectPayload();
    const result  = await API.predict(payload);

    showSpinner(false);

    if (result.ok) {
      _lastPayload = payload;
      _lastResult  = result;

      updateVerdict(result);

      // Show content, hide placeholders
      document.getElementById('overview-placeholder')?.classList.add('hidden');
      document.getElementById('overview-content')?.classList.remove('hidden');
      document.getElementById('shap-placeholder')?.classList.add('hidden');
      document.getElementById('shap-content')?.classList.remove('hidden');

      renderOverview(payload, result);
      const contributions = renderSHAPTab(payload, result);

      // Switch to Overview tab
      document.querySelector('[data-tab="overview"]')?.click();
      // Sync top-nav
      document.querySelectorAll('.top-nav-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === 'overview');
      });

      // Init chat
      Chat.init(payload, result, contributions);

      // Pre-populate ICD & Interactions so they render when clicked
      ICD.renderTable(ICD.groupFromCode(payload.diagnosis_code), 'meanShap', ICD.groupFromCode(payload.diagnosis_code));
      ICD.renderHeatmap(ICD.groupFromCode(payload.diagnosis_code));
      Interactions.renderPairsTable(ICD.groupFromCode(payload.diagnosis_code), Interactions.cptGroupFromCode(payload.activity_code));
      Interactions.renderSHAPHeatmap();
      Interactions.renderDenialHeatmap();

    } else if (result.kind === 'validation') {
      clearVerdict();
      _lastPayload = null; _lastResult = null;
      showValidationErrors(result);
      Chat.clear();
    } else {
      clearVerdict();
      _lastPayload = null; _lastResult = null;
      alert(`Error: ${result.detail}`);
    }
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
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        btn?.click();
      }
    });

    Chat.clear();   // set disabled state on load
  }

  // ------------------------------------------------------------------ //
  // ICD tab filter/sort wiring
  // ------------------------------------------------------------------ //
  function initICDControls() {
    ICD.populateGroupFilter('icd-filter-group');

    document.getElementById('icd-filter-group')?.addEventListener('change', () => {
      const activeGroup = _lastPayload ? ICD.groupFromCode(_lastPayload.diagnosis_code) : null;
      const filterGroup = document.getElementById('icd-filter-group')?.value ?? 'All';
      const sortKey     = document.getElementById('icd-sort')?.value ?? 'meanShap';
      ICD.renderTable(activeGroup, sortKey, filterGroup);
      ICD.renderHeatmap(filterGroup);
    });

    document.getElementById('icd-sort')?.addEventListener('change', () => {
      const activeGroup = _lastPayload ? ICD.groupFromCode(_lastPayload.diagnosis_code) : null;
      const filterGroup = document.getElementById('icd-filter-group')?.value ?? 'All';
      const sortKey     = document.getElementById('icd-sort')?.value ?? 'meanShap';
      ICD.renderTable(activeGroup, sortKey, filterGroup);
    });

    // Render read-only tables on initial load
    ICD.renderTable(null, 'meanShap', 'All');
    ICD.renderHeatmap('All');
    Interactions.renderPairsTable(null, null);
    Interactions.renderSHAPHeatmap();
    Interactions.renderDenialHeatmap();
  }

  // ------------------------------------------------------------------ //
  // Expander toggle
  // ------------------------------------------------------------------ //
  function initExpanders() {
    document.querySelectorAll('.expander-trigger').forEach(btn => {
      btn.addEventListener('click', () => {
        const body = btn.nextElementSibling;
        body?.classList.toggle('open');
        btn.querySelector('.expander-arrow')&&
          (btn.querySelector('.expander-arrow').textContent = body?.classList.contains('open') ? '▲' : '▼');
      });
    });
  }

  // ------------------------------------------------------------------ //
  // Small DOM helpers
  // ------------------------------------------------------------------ //
  function _titleCase(s) { return s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : ''; }

  function _setTableRows(tableId, rows) {
    const table = document.getElementById(tableId);
    if (!table) return;
    table.innerHTML = rows.map(([label, value]) =>
      `<tr><td>${label}</td><td>${value}</td></tr>`
    ).join('');
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
    initChat();
    initICDControls();
    initExpanders();

    document.getElementById('claim-form')?.addEventListener('submit', handleSubmit);
  });

})();
