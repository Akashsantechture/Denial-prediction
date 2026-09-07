/**
 * js/app.js
 * ---------
 * Main application controller — wired to predictionapi.py v4.0
 *
 * API request shape (POST /predict — ClaimRequest in validators/schemas.py):
 * {
 *   claim_id, patient_age, gender, nationality,
 *   encounter_type, length_of_stay,
 *   claim_gross, claim_net,
 *   clinician_profession, facility_type, payer_classification,
 *   primary_diagnosis_code,
 *   activities: [{
 *     activity_code, activity_quantity, activity_gross,
 *     diagnoses: [{ diagnosis_code, diagnosis_type }]
 *   }]
 * }
 *
 * API response shape (POST /predict):
 * {
 *   claim_id,
 *   claim_summary: {
 *     claim_denial_probability_pct,  claim_risk_level,
 *     highest_activity_risk,         average_activity_risk,
 *     activity_count
 *   },
 *   predictions: [{                  ← normalised from activity_predictions by api.js
 *     activity_code,
 *     denial_probability,
 *     predicted_denial,
 *     top_drivers: [{ feature, shap_value, impact }]
 *   }]
 * }
 *
 * Note: cpt_category and recommendation are NOT returned by the API.
 * cpt_category is derived client-side from activity_code.
 * recommendation is generated client-side from top_drivers.
 */

(() => {
  // ------------------------------------------------------------------ //
  // Module state
  // ------------------------------------------------------------------ //
  let _lastPayload      = null;
  let _lastResult       = null;
  let _selectedActIdx   = 0;     // currently selected activity index on SHAP tab
  let _intActIdx        = 0;     // currently selected activity index on Interactions tab
  let _intDepth         = 2;     // current interaction depth (2, 3, or 4)

  // ------------------------------------------------------------------ //
  // CPT category — mirrors preprocessing/categorial_conversion.py
  // ------------------------------------------------------------------ //
  function _deriveCptCategory(activityCode) {
    if (!activityCode) return 'Unknown_CPT';
    const code = String(activityCode).trim();
    const n = parseInt(code, 10);
    if (isNaN(n)) return 'HCPCS_Supplies';
    if (n >= 100   && n <= 1999)  return 'Anesthesia';
    if (n >= 10000 && n <= 69999) return 'Surgery';
    if (n >= 70000 && n <= 79999) return 'Radiology';
    if (n >= 80000 && n <= 89999) return 'Pathology_Laboratory';
    if (n >= 99202 && n <= 99499) return 'Evaluation_Management';
    if (n >= 90000)               return 'Medicine';
    return 'Unknown_CPT';
  }

  // ------------------------------------------------------------------ //
  // Recommendation generator — client-side, based on top SHAP drivers
  // ------------------------------------------------------------------ //
  function _buildRecommendation(pred, primaryDiagnosisCode) {
    const drivers = pred.top_drivers || [];
    if (drivers.length === 0) return 'Review claim documentation before resubmission.';

    const topRisk = drivers
      .filter(d => d.impact === 'increase_risk')
      .sort((a, b) => b.shap_value - a.shap_value)[0];

    if (!topRisk) return 'Claim appears well-supported. Verify documentation is complete.';

    const recs = {
      activity_code:          `Verify CPT \`${pred.activity_code}\` is correctly coded for the service rendered.`,
      activity_gross:         'Review the billed amount for this activity — high values increase denial likelihood.',
      activity_quantity:      'Check if the activity quantity is clinically justified and documented.',
      primary_diagnosis_code: `Ensure ICD \`${primaryDiagnosisCode}\` is the most specific code for the condition.`,
      primary_diagnosis_category: 'Verify the primary diagnosis category aligns with the procedure type.',
      claim_gross:            'High claim gross increases scrutiny. Confirm all line items are necessary.',
      claim_net:              'Verify net claim amount and applied discounts are accurate.',
      encounter_type:         'Confirm the encounter type (OP/IP/EM) matches the care setting.',
      length_of_stay:         'Ensure length of stay is medically justified and documented.',
      clinician_profession:   'Check that the clinician specialty is appropriate for the procedure.',
      facility_type:          'Verify the facility type is consistent with the procedure performed.',
      payer_classification:   'Review payer-specific requirements and pre-authorisation rules.',
      payer_classification:   'Review payer-specific requirements and pre-authorisation rules.',
      secondary_dx_count:     'Review secondary diagnoses — additional diagnoses affect model scoring.',
    };

    return recs[topRisk.feature]
      || `Address high SHAP driver: \`${topRisk.feature}\` (${topRisk.shap_value >= 0 ? '+' : ''}${topRisk.shap_value.toFixed(3)}).`;
  }

  // Enrich predictions with client-side cpt_category and recommendation
  function _enrichPredictions(predictions, activities, primaryDiagnosisCode) {
    return predictions.map((pred, idx) => ({
      ...pred,
      cpt_category:    _deriveCptCategory(activities[idx]?.activity_code ?? pred.activity_code),
      recommendation:  _buildRecommendation(pred, primaryDiagnosisCode),
    }));
  }

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
        if (_lastResult && _lastPayload) {
          if (target === 'icd')          renderICDTab();
          if (target === 'interactions') renderInteractionsTab();
          if (target === 'shap')         _renderSHAPForActivity(_selectedActIdx);
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
    if (badge) {
      badge.className = 'health-badge loading';
      badge.innerHTML = '<span class="health-dot"></span> Checking…';
    }

    const h = await API.health();

    if (badge) {
      const allLoaded = h.model_loaded && h.encoder_loaded && h.shap_loaded;
      if (h.status === 'healthy' && allLoaded) {
        badge.className = 'health-badge online';
        badge.innerHTML = '<span class="health-dot"></span> Backend online · Model · Encoder · SHAP loaded';
      } else if (h.status === 'healthy' && h.model_loaded) {
        badge.className = 'health-badge loading';
        const missing = [
          !h.encoder_loaded && 'Encoder',
          !h.shap_loaded    && 'SHAP',
        ].filter(Boolean).join(', ');
        badge.innerHTML = `<span class="health-dot"></span> Backend online · Model loaded · ${missing} NOT loaded`;
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
    const v = id => g(id)?.value?.trim() ?? '';

    const primaryDxCode = v('f-primary_diagnosis_code').toUpperCase();

    const activityRows = document.querySelectorAll('.activity-row');
    const activities   = Array.from(activityRows).map(row => {
      // Per-activity diagnoses — primary + any secondary rows
      const diagnoses = [];

      // Always add the claim-level primary diagnosis as the activity's primary dx
      if (primaryDxCode) {
        diagnoses.push({ diagnosis_code: primaryDxCode, diagnosis_type: 'primary' });
      }

      // Secondary diagnoses entered per activity
      row.querySelectorAll('.secondary-dx-row').forEach(secRow => {
        const secCode = secRow.querySelector('.sec-dx-code')?.value?.trim().toUpperCase();
        if (secCode) {
          diagnoses.push({ diagnosis_code: secCode, diagnosis_type: 'secondary' });
        }
      });

      return {
        activity_code:     row.querySelector('.act-code')?.value?.trim()    || '',
        activity_quantity: parseFloat(row.querySelector('.act-qty')?.value)  || 1,
        activity_gross:    parseFloat(row.querySelector('.act-gross')?.value) || 0,
        diagnoses,
      };
    });

    return {
      claim_id:               `CLM-${Date.now()}`,
      primary_diagnosis_code: primaryDxCode,
      patient_age:            parseInt(v('f-patient_age'), 10),
      gender:                 v('f-gender'),
      nationality:            v('f-nationality'),
      encounter_type:         v('f-encounter_type'),
      length_of_stay:         parseInt(v('f-length_of_stay'), 10),
      claim_gross:            parseFloat(v('f-claim_gross')),
      claim_net:              parseFloat(v('f-claim_net')),
      clinician_profession:   v('f-clinician_profession'),
      facility_type:          v('f-facility_type'),
      payer_classification:   v('f-payer_id') || 'UNKNOWN',
      payer_id:               v('f-payer_id'),
      activities,
    };
  }

  // ------------------------------------------------------------------ //
  // Spinner
  // ------------------------------------------------------------------ //
  function showSpinner(visible) {
    document.getElementById('spinner-overlay')
      ?.classList.toggle('visible', visible);
  }

  // ------------------------------------------------------------------ //
  // Validation errors
  // ------------------------------------------------------------------ //
  function showValidationErrors(failure) {
    const block = document.getElementById('validation-block');
    if (!block) return;
    document.getElementById('val-title').textContent =
      'Submission Could Not Be Processed';
    document.getElementById('val-msg').textContent =
      failure.message ?? 'Please review the fields below.';
    const list = document.getElementById('val-error-list');
    list.innerHTML = (failure.errors || []).map(e => `
      <div class="val-error-item">
        <div class="val-error-field">Field: ${e.field ?? 'Unknown'}</div>
        <div class="val-error-issue">${e.issue ?? 'Invalid value'}</div>
        ${e.provided_value != null
          ? `<div class="val-error-value">Provided: <code>${e.provided_value}</code></div>`
          : ''}
      </div>`).join('');
    block.classList.add('visible');
    block.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function hideValidationErrors() {
    document.getElementById('validation-block')?.classList.remove('visible');
  }

  // ------------------------------------------------------------------ //
  // Reset result UI
  // ------------------------------------------------------------------ //
  function resetResultUI() {
    document.getElementById('overview-content')?.classList.add('hidden');
    document.getElementById('overview-placeholder')?.classList.remove('hidden');
    document.getElementById('shap-content')?.classList.add('hidden');
    document.getElementById('shap-placeholder')?.classList.remove('hidden');

    ['gauge-chart', 'shap-waterfall-chart',
     'shap-scatter-chart', 'shap-dependence-chart'].forEach(id => {
      const el = document.getElementById(id);
      if (el) Plotly.purge(el);
    });

    const ab = document.getElementById('action-banner');
    if (ab) { ab.className = 'action-banner low'; ab.innerHTML = '<strong>—</strong>'; }
    const rb = document.getElementById('risk-badge');
    if (rb) { rb.className = 'risk-badge'; rb.textContent = '—'; }
    _setTableRows('patient-table', []);
    ['kpi-prob','kpi-shap','kpi-flags','kpi-revised',
     'kpi-claim-score','kpi-highest-act','kpi-avg-act','kpi-act-count']
      .forEach(id => _setKPI(id, '—', 'neutral', id === 'kpi-claim-score' ? 'At-Risk Amount' : '—'));
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
    banner.className   = `verdict-banner visible ${level}`;
    banner.textContent = labels[level] ?? `${pct.toFixed(1)}% Denial Score`;
  }

  function clearVerdict() {
    const b = document.getElementById('verdict-banner');
    if (b) b.className = 'verdict-banner';
  }

  // ------------------------------------------------------------------ //
  // Overview tab render
  // ------------------------------------------------------------------ //
  function renderOverview(payload, result) {
    const cs    = result.claim_summary;
    const preds = result.predictions ?? [];
    const pct   = cs.claim_denial_probability_pct;
    const level = cs.claim_risk_level.toLowerCase();

    // Claim Score Hero
    const scoreEl = document.getElementById('claim-score-value');
    const levelEl = document.getElementById('claim-score-level');
    const barEl   = document.getElementById('claim-score-bar');
    if (scoreEl) scoreEl.textContent      = `${pct.toFixed(1)}%`;
    if (levelEl) {
      levelEl.textContent = cs.claim_risk_level;
      levelEl.className   = `claim-score-level ${level}`;
    }
    if (barEl) {
      barEl.style.width = `${Math.min(pct, 100)}%`;
      barEl.className   = `claim-score-fill ${level}`;
    }

    // At-Risk Amount — sum activity_gross only for hard-denied activities
    const atRiskData = preds.reduce((acc, pred, i) => {
      if (pred.predicted_denial) {
        acc.amount += (payload.activities?.[i]?.activity_gross ?? 0);
        acc.count++;
      }
      return acc;
    }, { amount: 0, count: 0 });

    const atRiskTotal = atRiskData.amount;
    const atRiskCount = atRiskData.count;
    const totalActs   = preds.length;

    const atRiskLabel = atRiskCount === 0
      ? 'No financial exposure'
      : atRiskCount === totalActs
        ? 'Full claim at risk'
        : `${atRiskCount} of ${totalActs} activities at risk`;

    const atRiskLevel = atRiskCount === 0 ? 'green'
      : atRiskCount === totalActs          ? 'high'
      : 'medium';

    const atRiskDisplay = atRiskCount === 0
      ? 'AED 0'
      : `AED ${atRiskTotal.toLocaleString('en-AE', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    _setKPI('kpi-claim-score', atRiskDisplay, atRiskLevel, atRiskLabel);
    _setKPI('kpi-highest-act',  `${(cs.highest_activity_risk * 100).toFixed(1)}%`, 'neutral', 'Highest Activity');
    _setKPI('kpi-avg-act',      `${(cs.average_activity_risk  * 100).toFixed(1)}%`, 'neutral', 'Average Activity');
    _setKPI('kpi-act-count',    String(cs.activity_count), 'neutral', 'Activities Submitted');

    Charts.gauge('gauge-chart', pct);

    const rb = document.getElementById('risk-badge');
    if (rb) { rb.className = `risk-badge ${level}`; rb.textContent = cs.claim_risk_level; }

    const fmt = v => parseFloat(v).toLocaleString('en', {
      minimumFractionDigits: 2, maximumFractionDigits: 2,
    });
    _setTableRows('patient-table', [
      ['Age / Gender',   `${payload.patient_age} yrs / ${_titleCase(payload.gender)}`],
      ['Nationality',    _titleCase(payload.nationality)],
      ['Primary ICD',    payload.primary_diagnosis_code],
      ['ICD Category',   API.deriveIcdCategory(payload.primary_diagnosis_code)],
      ['Claim Gross',    `AED ${fmt(payload.claim_gross)}`],
      ['Claim Net',      `AED ${fmt(payload.claim_net)}`],
      ['LOS',            `${payload.length_of_stay} day${payload.length_of_stay !== 1 ? 's' : ''}`],
      ['Encounter',      payload.encounter_type],
      ['Clinician',      payload.clinician_profession],
      ['Facility',       payload.facility_type],
      ['Payer',          payload.payer_id || payload.payer_classification],
    ]);

    // Action banner — top-level recommendation from highest-risk prediction
    const ab = document.getElementById('action-banner');
    if (ab) {
      const topPred = [...preds].sort((a, b) =>
        b.denial_probability - a.denial_probability)[0];
      ab.className = `action-banner ${level}`;
      ab.innerHTML = `<strong>${cs.claim_risk_level} Denial Likelihood (${pct.toFixed(1)}%)</strong>` +
        (topPred?.recommendation ? ` — ${topPred.recommendation}` : '');
    }

    _renderActivityTable(preds, payload.activities ?? []);
  }

  function _renderActivityTable(predictions, activities) {
    const tbody = document.getElementById('activity-predictions-body');
    if (!tbody) return;
    tbody.innerHTML = predictions.map((pred, idx) => {
      const pct    = (pred.denial_probability * 100).toFixed(1);
      const lvl    = pred.denial_probability >= 0.70 ? 'high'
                   : pred.denial_probability >= 0.40 ? 'medium' : 'low';
      const denied = pred.predicted_denial;
      const act    = activities[idx] ?? {};
      const gross  = act.activity_gross ?? '—';
      const qty    = act.activity_quantity ?? '—';
      const pills  = (pred.top_drivers || []).slice(0, 3).map(d => {
        const cls  = d.impact === 'increase_risk' ? 'pos' : 'neg';
        const sign = d.shap_value >= 0 ? '+' : '';
        return `<span class="shap-pill ${cls}" title="${d.feature}">${sign}${d.shap_value.toFixed(3)}</span>`;
      }).join(' ');
      return `
        <tr class="${denied ? 'row-denied' : ''}">
          <td><code>${pred.activity_code}</code></td>
          <td><span class="cpt-badge">${pred.cpt_category || '—'}</span></td>
          <td>${qty}</td>
          <td>${typeof gross === 'number'
            ? `AED ${gross.toLocaleString('en', { minimumFractionDigits: 2 })}`
            : gross}</td>
          <td>
            <div class="act-prob-row">
              <span class="act-prob-val ${lvl}">${pct}%</span>
              <div class="act-prob-bar-wrap">
                <div class="act-prob-bar ${lvl}"
                     style="width:${Math.min(parseFloat(pct), 100)}%"></div>
              </div>
            </div>
          </td>
          <td><span class="denial-verdict ${denied ? 'denied' : 'approved'}">
            ${denied ? '⚠ Likely Denied' : '✓ Likely Approved'}
          </span></td>
          <td class="driver-pills-cell">${pills || '<span class="text-muted">—</span>'}</td>
          <td class="rec-cell">${pred.recommendation || '—'}</td>
        </tr>`;
    }).join('');
  }

  // ------------------------------------------------------------------ //
  // SHAP tab — activity selector setup
  // ------------------------------------------------------------------ //
  function initSHAPActivitySelector(result) {
    const sel = document.getElementById('shap-activity-select');
    if (!sel) return;

    const preds = result.predictions ?? [];

    const sortedByRisk = [...preds]
      .map((p, i) => ({ ...p, _origIdx: i }))
      .sort((a, b) => b.denial_probability - a.denial_probability);

    sel.innerHTML = preds.map((pred, idx) => {
      const pct    = (pred.denial_probability * 100).toFixed(1);
      const flag   = pred.predicted_denial ? ' ⚠' : ' ✓';
      const isTop  = sortedByRisk[0]._origIdx === idx ? ' ★ Highest risk' : '';
      return `<option value="${idx}">
        #${idx + 1} · ${pred.activity_code} · ${pred.cpt_category} · ${pct}%${flag}${isTop}
      </option>`;
    }).join('');

    const defaultIdx = sortedByRisk[0]?._origIdx ?? 0;
    sel.value        = String(defaultIdx);
    _selectedActIdx  = defaultIdx;

    sel.onchange = () => {
      _selectedActIdx = parseInt(sel.value, 10);
      _renderSHAPForActivity(_selectedActIdx);
    };
  }

  // ------------------------------------------------------------------ //
  // SHAP tab — render for a specific activity index
  // ------------------------------------------------------------------ //
  function _renderSHAPForActivity(idx) {
    if (!_lastResult || !_lastPayload) return;

    const preds  = _lastResult.predictions ?? [];
    const pred   = preds[idx];
    if (!pred) return;

    const contributions = SHAP.fromActivityIndex(_lastResult, idx);
    const baseVal       = SHAP.baseLogOdds(_lastResult);
    const finalVal      = SHAP.finalLogOddsForActivity(_lastResult, idx);
    const netShap       = parseFloat((finalVal - baseVal).toFixed(4));
    const HIGH_THRESHOLD = 0.05;

    const actPct     = parseFloat((pred.denial_probability * 100).toFixed(1));
    const riskClass  = actPct >= 70 ? 'high' : actPct >= 40 ? 'medium' : 'low';

    _setKPI('kpi-prob',
      `${actPct.toFixed(1)}%`, riskClass,
      pred.predicted_denial ? '⚠ Predicted denied' : '✓ Predicted approved'
    );
    _setKPI('kpi-shap',
      `${netShap >= 0 ? '+' : ''}${netShap.toFixed(3)}`,
      netShap > 0 ? 'high' : 'green',
      `Base: ${baseVal.toFixed(3)}`
    );

    const highFlags  = contributions.filter(c => Math.abs(c.value) >= HIGH_THRESHOLD);
    const flagLabels = highFlags.slice(0, 3).map(c => c.name.split(' ')[0]).join(' · ') || '—';
    _setKPI('kpi-flags', String(highFlags.length), highFlags.length > 0 ? 'high' : 'neutral', flagLabels);

    const topVal    = contributions[0]?.value ?? 0;
    const revisedLO = finalVal - Math.abs(topVal);
    const revisedP  = Math.round(100 / (1 + Math.exp(-revisedLO)));
    _setKPI('kpi-revised', `~${Math.max(revisedP, 2)}%`, 'green', '↓ If top driver resolved');

    const badgeRow = document.getElementById('shap-activity-badge-row');
    if (badgeRow) {
      badgeRow.innerHTML = `
        <span class="cpt-badge">${pred.cpt_category}</span>
        <span class="denial-verdict ${pred.predicted_denial ? 'denied' : 'approved'}">
          ${pred.predicted_denial ? '⚠ Likely Denied' : '✓ Likely Approved'}
        </span>
        <span class="kpi-sub" style="font-size:.78rem;">
          Claim-level score: <strong>${_lastResult.claim_summary.claim_denial_probability_pct.toFixed(1)}%</strong>
        </span>`;
    }

    Charts.waterfall('shap-waterfall-chart', contributions, baseVal, finalVal);
    Charts.featureScatter('shap-scatter-chart', contributions);

    const actGross        = _lastPayload.activities?.[idx]?.activity_gross ?? 250;
    const actGrossFeature = contributions.find(c =>
      c.feature === 'activity_gross' || c.name.toLowerCase().includes('activity gross')
    );
    Charts.dependence('shap-dependence-chart', actGross, actGrossFeature?.value ?? null);

    _renderSHAPAllActivitiesTable(preds, idx);
  }

  function _renderSHAPAllActivitiesTable(predictions, activeIdx) {
    const tbody = document.getElementById('shap-all-acts-body');
    if (!tbody) return;

    tbody.innerHTML = predictions.map((pred, idx) => {
      const pct    = (pred.denial_probability * 100).toFixed(1);
      const lvl    = pred.denial_probability >= 0.70 ? 'high'
                   : pred.denial_probability >= 0.40 ? 'medium' : 'low';
      const denied = pred.predicted_denial;
      const drivers = pred.top_drivers || [];

      const driverCell = (i) => {
        const d = drivers[i];
        if (!d) return '<td class="text-muted">—</td>';
        const cls  = d.impact === 'increase_risk' ? 'pos' : 'neg';
        const sign = d.shap_value >= 0 ? '+' : '';
        return `<td>
          <span class="shap-pill ${cls}">${sign}${d.shap_value.toFixed(3)}</span>
          <span style="font-size:.72rem;color:var(--color-text-muted);margin-left:.25rem;">
            ${SHAP.DISPLAY_NAMES[d.feature] || d.feature}
          </span>
        </td>`;
      };

      const isActive = idx === activeIdx;
      return `
        <tr class="${isActive ? 'shap-row-active' : ''} ${denied ? 'shap-row-denied' : ''}"
            data-act-idx="${idx}" style="cursor:pointer;">
          <td>${idx + 1}</td>
          <td><code>${pred.activity_code}</code></td>
          <td><span class="cpt-badge">${pred.cpt_category || '—'}</span></td>
          <td><span class="act-prob-val ${lvl}">${pct}%</span></td>
          <td><span class="denial-verdict ${denied ? 'denied' : 'approved'}">
            ${denied ? '⚠ Denied' : '✓ Approved'}
          </span></td>
          ${driverCell(0)}
          <td>${drivers[0]
            ? `<span class="shap-pill ${drivers[0].impact === 'increase_risk' ? 'pos' : 'neg'}">
                ${drivers[0].shap_value >= 0 ? '+' : ''}${drivers[0].shap_value.toFixed(3)}
               </span>`
            : '—'}</td>
          ${driverCell(1)}
          ${driverCell(2)}
        </tr>`;
    }).join('');

    tbody.querySelectorAll('tr[data-act-idx]').forEach(row => {
      row.addEventListener('click', () => {
        const i  = parseInt(row.dataset.actIdx, 10);
        const sel = document.getElementById('shap-activity-select');
        if (sel) sel.value = String(i);
        _selectedActIdx = i;
        _renderSHAPForActivity(i);
      });
    });
  }

  // ------------------------------------------------------------------ //
  // renderSHAPTab — called once after a new submission.
  // ------------------------------------------------------------------ //
  function renderSHAPTab(payload, result) {
    initSHAPActivitySelector(result);
    _renderSHAPForActivity(_selectedActIdx);
    return SHAP.fromApiResult(result);
  }

  // ------------------------------------------------------------------ //
  // ICD & Interactions tabs
  // ------------------------------------------------------------------ //
  function renderICDTab() {
    if (!_lastPayload) return;
    const activeGroup = ICD.groupFromCode(_lastPayload.primary_diagnosis_code);
    const captEl      = document.getElementById('icd-active-caption');
    if (captEl) captEl.textContent =
      `ICD ${_lastPayload.primary_diagnosis_code} → ${activeGroup}. Matching rows highlighted.`;
    const filterSel = document.getElementById('icd-filter-group');
    if (filterSel && filterSel.value === 'All') {
      for (const opt of filterSel.options) {
        if (opt.value === activeGroup) { opt.selected = true; break; }
      }
    }
    ICD.renderTable(
      activeGroup,
      document.getElementById('icd-sort')?.value ?? 'meanShap',
      filterSel?.value ?? 'All'
    );
    ICD.renderHeatmap(filterSel?.value ?? 'All');
  }

  function renderInteractionsTab() {
    if (!_lastPayload) return;
    const activeIcd = ICD.groupFromCode(_lastPayload.primary_diagnosis_code);
    const activeCpt = Interactions.cptGroupFromCode(
      _lastPayload.activities?.[0]?.activity_code ?? ''
    );
    const cap = document.getElementById('interactions-caption');
    if (cap) cap.textContent = `ICD: ${activeIcd} | CPT: ${activeCpt}`;
    Interactions.renderPairsTable(activeIcd, activeCpt);
    Interactions.renderSHAPHeatmap();
    Interactions.renderDenialHeatmap();
    // Live section — re-render with current depth + activity index
    if (_lastResult) Interactions.renderLive(_lastResult, _intActIdx, _intDepth);
  }

  // ------------------------------------------------------------------ //
  // Interactions tab — activity selector + depth toggle controls
  // ------------------------------------------------------------------ //
  function initInteractionsControls() {
    // Populate activity selector
    const sel = document.getElementById('int-activity-select');
    if (!sel) return;

    if (_lastResult) {
      const preds = _lastResult.predictions ?? [];
      sel.innerHTML = preds.map((pred, idx) => {
        const pct  = (pred.denial_probability * 100).toFixed(1);
        const flag = pred.predicted_denial ? ' ⚠' : ' ✓';
        return `<option value="${idx}">#${idx + 1} · ${pred.activity_code} · ${pct}%${flag}</option>`;
      }).join('');
      // Default to highest-risk activity
      const topIdx = [...preds]
        .map((p, i) => ({ p, i }))
        .sort((a, b) => b.p.denial_probability - a.p.denial_probability)[0]?.i ?? 0;
      sel.value = String(topIdx);
      _intActIdx = topIdx;
    }

    sel.onchange = () => {
      _intActIdx = parseInt(sel.value, 10);
      if (_lastResult) Interactions.renderLive(_lastResult, _intActIdx, _intDepth);
    };

    // Depth toggle buttons
    document.getElementById('depth-toggle-group')
      ?.querySelectorAll('.depth-btn')
      .forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          _intDepth = parseInt(btn.dataset.depth, 10);
          if (_lastResult) Interactions.renderLive(_lastResult, _intActIdx, _intDepth);
        });
      });
  }

  // ------------------------------------------------------------------ //
  // Form submit
  // ------------------------------------------------------------------ //
  async function handleSubmit(e) {
    e.preventDefault();
    hideValidationErrors();
    showSpinner(true);
    Chat.clear();

    const rawPayload = (window.__jsonModeActive?.())
      ? window.__jsonModePayload()
      : collectPayload();

    // Normalise: support legacy JSON payloads that use diagnosis_code
    if (!rawPayload.primary_diagnosis_code && rawPayload.diagnosis_code) {
      rawPayload.primary_diagnosis_code = rawPayload.diagnosis_code;
    }

    // Normalise activity diagnoses — if activities have no diagnoses array,
    // inject the primary diagnosis automatically
    if (Array.isArray(rawPayload.activities)) {
      rawPayload.activities = rawPayload.activities.map(a => {
        const diagnoses = Array.isArray(a.diagnoses) ? a.diagnoses : [];
        if (diagnoses.length === 0 && rawPayload.primary_diagnosis_code) {
          diagnoses.push({
            diagnosis_code: rawPayload.primary_diagnosis_code,
            diagnosis_type: 'primary',
          });
        }
        return { ...a, diagnoses };
      });
    }

    if (!rawPayload.activities || rawPayload.activities.length === 0) {
      showSpinner(false);
      showValidationErrors({ message: 'At least one activity is required.', errors: [] });
      return;
    }

    const result = await API.predict(rawPayload);
    showSpinner(false);

    if (result.ok) {
      // Enrich raw predictions with client-side cpt_category + recommendation
      result.predictions = _enrichPredictions(
        result.predictions,
        rawPayload.activities,
        rawPayload.primary_diagnosis_code
      );

      _lastPayload    = rawPayload;
      _lastResult     = result;
      _selectedActIdx = 0;
      _intActIdx      = 0;
      _intDepth       = _intDepth || 2;   // preserve user's depth choice across submissions

      updateVerdict(result);

      document.getElementById('overview-placeholder')?.classList.add('hidden');
      document.getElementById('overview-content')?.classList.remove('hidden');
      document.getElementById('shap-placeholder')?.classList.add('hidden');
      document.getElementById('shap-content')?.classList.remove('hidden');

      renderOverview(rawPayload, result);
      const contributions = renderSHAPTab(rawPayload, result);

      document.querySelector('.tab-btn[data-tab="overview"]')?.click();
      document.querySelectorAll('.top-nav-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === 'overview');
      });

      Chat.init(rawPayload, result, contributions);

      ICD.renderTable(
        ICD.groupFromCode(rawPayload.primary_diagnosis_code), 'meanShap',
        ICD.groupFromCode(rawPayload.primary_diagnosis_code)
      );
      ICD.renderHeatmap(ICD.groupFromCode(rawPayload.primary_diagnosis_code));
      const firstCpt = rawPayload.activities?.[0]?.activity_code ?? '';
      Interactions.renderPairsTable(
        ICD.groupFromCode(rawPayload.primary_diagnosis_code),
        Interactions.cptGroupFromCode(firstCpt)
      );
      Interactions.renderSHAPHeatmap();
      Interactions.renderDenialHeatmap();
      // Initialise live interaction controls + render default (depth 2, highest-risk activity)
      initInteractionsControls();
      Interactions.renderLive(result, _intActIdx, _intDepth);

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
  // Activity row manager
  // ------------------------------------------------------------------ //
  function _makeActivityRow(idx, defaults = {}) {
    const row = document.createElement('tr');
    row.className = 'activity-row';
    row.innerHTML = `
      <td>
        <input class="act-code field-inline" type="text"
               value="${defaults.activity_code ?? '99213'}"
               placeholder="e.g. 99213" />
      </td>
      <td>
        <input class="act-qty field-inline" type="number"
               value="${defaults.activity_quantity ?? 1}" min="1" max="999" />
      </td>
      <td>
        <input class="act-gross field-inline" type="number"
               value="${defaults.activity_gross ?? 250}" min="0" step="10" />
      </td>
      <td class="act-secondary-dx-cell">
        <div class="secondary-dx-list"></div>
        <button type="button" class="btn-add-sec-dx" title="Add secondary diagnosis">+ Secondary DX</button>
      </td>
      <td>
        <button type="button" class="btn-remove-act" title="Remove activity">✕</button>
      </td>`;

    // Wire remove-activity button
    row.querySelector('.btn-remove-act').addEventListener('click', () => {
      const tbody = document.getElementById('activities-tbody');
      if (tbody && tbody.querySelectorAll('.activity-row').length > 1) row.remove();
    });

    // Wire add-secondary-dx button
    row.querySelector('.btn-add-sec-dx').addEventListener('click', () => {
      _addSecondaryDxRow(row.querySelector('.secondary-dx-list'));
    });

    // Restore any existing secondary diagnoses (used when populating from defaults)
    if (Array.isArray(defaults.diagnoses)) {
      const secList = row.querySelector('.secondary-dx-list');
      defaults.diagnoses
        .filter(d => d.diagnosis_type === 'secondary')
        .forEach(d => _addSecondaryDxRow(secList, d.diagnosis_code));
    }

    return row;
  }

  function _addSecondaryDxRow(container, value = '') {
    const wrapper = document.createElement('div');
    wrapper.className = 'secondary-dx-row';
    wrapper.innerHTML = `
      <input class="sec-dx-code field-inline" type="text"
             value="${value}" placeholder="e.g. E11.9" style="width:90px;" />
      <button type="button" class="btn-remove-sec-dx" title="Remove">✕</button>`;
    wrapper.querySelector('.btn-remove-sec-dx').addEventListener('click', () => wrapper.remove());
    container.appendChild(wrapper);
    wrapper.querySelector('.sec-dx-code').focus();
  }

  function initActivityManager() {
    const tbody  = document.getElementById('activities-tbody');
    const addBtn = document.getElementById('add-activity-btn');
    if (!tbody || !addBtn) return;
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
      const ag  = _lastPayload ? ICD.groupFromCode(_lastPayload.primary_diagnosis_code) : null;
      const fg  = document.getElementById('icd-filter-group')?.value ?? 'All';
      const sk  = document.getElementById('icd-sort')?.value ?? 'meanShap';
      ICD.renderTable(ag, sk, fg);
      ICD.renderHeatmap(fg);
    });
    document.getElementById('icd-sort')?.addEventListener('change', () => {
      const ag = _lastPayload ? ICD.groupFromCode(_lastPayload.primary_diagnosis_code) : null;
      ICD.renderTable(
        ag,
        document.getElementById('icd-sort')?.value ?? 'meanShap',
        document.getElementById('icd-filter-group')?.value ?? 'All'
      );
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
  function _titleCase(s) {
    return s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : '';
  }

  function _setTableRows(tableId, rows) {
    const table = document.getElementById(tableId);
    if (!table) return;
    table.innerHTML = rows
      .map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`)
      .join('');
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
    initInteractionsControls();
    initExpanders();
    document.getElementById('claim-form')?.addEventListener('submit', handleSubmit);
  });

})();
