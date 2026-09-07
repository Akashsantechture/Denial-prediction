/**
 * js/interactions.js
 * ------------------
 * Interactions tab — static population reference +
 * LIVE per-activity interaction depth explorer.
 *
 * Live section:
 *   Interactions.renderLive(result, actIdx, depth)
 *     depth 2 → 1:1     → pairwise heatmap + ranked pairs table
 *     depth 3 → 1:1:1   → triple ranked bar chart
 *     depth 4 → 1:1:1:1 → quad ranked table + dominant risk card
 *
 * Static section (unchanged):
 *   Interactions.renderPairsTable(activeIcdGroup, activeCptGroup)
 *   Interactions.renderSHAPHeatmap()
 *   Interactions.renderDenialHeatmap()
 *   Interactions.cptGroupFromCode(activityCode)
 */

const Interactions = (() => {

  // ================================================================== //
  // UTILITIES
  // ================================================================== //

  /** Generic k-combinations from array — returns array of arrays. */
  function comb(arr, k) {
    if (k === 0) return [[]];
    if (k > arr.length) return [];
    if (k === arr.length) return [arr.slice()];
    const [first, ...rest] = arr;
    return [
      ...comb(rest, k - 1).map(c => [first, ...c]),
      ...comb(rest, k),
    ];
  }

  /** CPT range → group — mirrors Interactions static + app.js */
  function cptGroupFromCode(code) {
    if (!code) return 'Unknown_CPT';
    const s = String(code).trim().toUpperCase();
    if (!/^\d/.test(s)) return 'HCPCS_Supplies';
    const n = parseInt(s, 10);
    if (n >= 100   && n <= 1999)  return 'Anesthesia';
    if (n >= 10004 && n <= 69990) return 'Surgery';
    if (n >= 70010 && n <= 79999) return 'Radiology';
    if (n >= 80047 && n <= 89398) return 'Pathology_Laboratory';
    if (n >= 99202 && n <= 99499) return 'Evaluation_Management';
    if (n >= 90281 && n <= 99607) return 'Medicine';
    return 'Unknown_CPT';
  }

  /** Human-readable feature name — reuses SHAP.DISPLAY_NAMES if loaded */
  function _label(feature) {
    return (typeof SHAP !== 'undefined' && SHAP.DISPLAY_NAMES?.[feature])
      ? SHAP.DISPLAY_NAMES[feature]
      : feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  /** Short label (first two words max) */
  function _shortLabel(feature) {
    return _label(feature).split(' ').slice(0, 2).join(' ');
  }

  /** Direction string from array of shap values */
  function _direction(shapVals) {
    const pos = shapVals.filter(v => v > 0).length;
    const neg = shapVals.filter(v => v < 0).length;
    if (pos === shapVals.length) return 'amplifying';
    if (neg === shapVals.length) return 'conflicting';
    return 'mixed';
  }

  const DIR_LABELS = {
    amplifying:  { badge: '↑ All amplify denial',    cls: 'dir-amplifying' },
    conflicting: { badge: '↓ All reduce denial',     cls: 'dir-conflicting' },
    mixed:       { badge: '⚡ Mixed directions',       cls: 'dir-mixed' },
  };

  // ================================================================== //
  // LIVE RENDERING
  // ================================================================== //

  /**
   * Main entry point.
   * Called by app.js whenever result / actIdx / depth changes.
   *
   * @param {object} result   — full API result (predictions[].top_drivers)
   * @param {number} actIdx   — selected activity index (0-based)
   * @param {number} depth    — 2, 3, or 4
   */
  function renderLive(result, actIdx, depth) {
    const pred = result?.predictions?.[actIdx];
    if (!pred) { _showLivePlaceholder(); return; }

    const drivers = (pred.top_drivers || [])
      .slice()
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, 5);                              // always top-5 source pool

    if (drivers.length < depth) {
      _showLivePlaceholder(
        `Not enough SHAP drivers for ${depth}-way interactions. ` +
        `This activity returned ${drivers.length} driver(s).`
      );
      return;
    }

    // Update live header banner
    _renderLiveHeader(pred, actIdx, depth, drivers);

    // Depth-specific render
    if (depth === 2) _renderDepth2(drivers, pred);
    if (depth === 3) _renderDepth3(drivers, pred);
    if (depth === 4) _renderDepth4(drivers, pred);
  }

  // ------------------------------------------------------------------ //
  // Live header banner
  // ------------------------------------------------------------------ //
  function _renderLiveHeader(pred, actIdx, depth, drivers) {
    const el = document.getElementById('live-interaction-header');
    if (!el) return;

    const pct   = (pred.denial_probability * 100).toFixed(1);
    const level = pred.denial_probability >= 0.70 ? 'high'
                : pred.denial_probability >= 0.40 ? 'medium' : 'low';
    const depthLabel = { 2: '1:1 — Pairwise', 3: '1:1:1 — Triple', 4: '1:1:1:1 — Quad' };

    // Find dominant combination
    const topDrivers = drivers.slice(0, depth);
    const combined   = topDrivers.reduce((s, d) => s + d.shap_value, 0);
    const dir        = _direction(topDrivers.map(d => d.shap_value));
    const topNames   = topDrivers.map(d => _shortLabel(d.feature)).join(' × ');

    el.className = `live-interaction-header ${level}`;
    el.innerHTML = `
      <div class="live-header-left">
        <span class="live-badge">⚡ LIVE</span>
        <span class="live-act-label">Activity #${actIdx + 1} · <code>${pred.activity_code}</code> · ${pct}% denial risk</span>
        <span class="live-depth-label">${depthLabel[depth] ?? depth + '-way'}</span>
      </div>
      <div class="live-header-right">
        <span class="live-dominant-label">Dominant combination:</span>
        <span class="live-dominant-combo">${topNames}</span>
        <span class="live-combined-score ${combined >= 0 ? 'pos' : 'neg'}">
          Combined SHAP: ${combined >= 0 ? '+' : ''}${combined.toFixed(3)}
        </span>
        <span class="depth-dir-badge ${DIR_LABELS[dir].cls}">${DIR_LABELS[dir].badge}</span>
      </div>`;
  }

  // ------------------------------------------------------------------ //
  // DEPTH 2 — Pairwise heatmap + pairs table
  // ------------------------------------------------------------------ //
  function _renderDepth2(drivers, pred) {
    const container = document.getElementById('live-depth-content');
    if (!container) return;

    const pairs  = comb(drivers, 2);
    const labels = drivers.map(d => _shortLabel(d.feature));
    const n      = drivers.length;

    // Build symmetric NxN matrix
    // Diagonal = own SHAP value; off-diagonal = sum of both
    const z = Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => {
        if (r === c) return parseFloat(drivers[r].shap_value.toFixed(3));
        const a = drivers[r].shap_value;
        const b = drivers[c].shap_value;
        return parseFloat((a + b).toFixed(3));
      })
    );

    // Ranked pairs list (off-diagonal, unique pairs)
    const rankedPairs = pairs
      .map(([a, b]) => ({
        featureA:   a.feature,
        featureB:   b.feature,
        labelA:     _shortLabel(a.feature),
        labelB:     _shortLabel(b.feature),
        shapA:      a.shap_value,
        shapB:      b.shap_value,
        combined:   a.shap_value + b.shap_value,
        direction:  _direction([a.shap_value, b.shap_value]),
        impactA:    a.impact,
        impactB:    b.impact,
      }))
      .sort((a, b) => Math.abs(b.combined) - Math.abs(a.combined));

    container.innerHTML = `
      <div class="live-section-row">
        <div class="live-heatmap-wrap">
          <div class="chart-title" style="font-size:.82rem;margin-bottom:.4rem;">
            Pairwise SHAP Combination Matrix
            <span class="chart-caption" style="margin-left:.5rem;">
              Off-diagonal = combined SHAP of both features · Diagonal = individual contribution
            </span>
          </div>
          <div id="live-pair-heatmap" class="plotly-container" style="min-height:340px;"></div>
        </div>
        <div class="live-pairs-table-wrap">
          <div class="chart-title" style="font-size:.82rem;margin-bottom:.4rem;">
            Ranked Pairs — ${rankedPairs.length} combinations
          </div>
          <div class="data-table-wrap">
            <table class="data-table" id="live-pairs-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Feature A</th>
                  <th>SHAP A</th>
                  <th>Feature B</th>
                  <th>SHAP B</th>
                  <th>Combined</th>
                  <th>Direction</th>
                </tr>
              </thead>
              <tbody>
                ${rankedPairs.map((p, i) => `
                  <tr class="${i === 0 ? 'dominant-pair-row' : ''}">
                    <td>${i + 1}</td>
                    <td><span class="feat-label">${p.labelA}</span></td>
                    <td><span class="shap-pill ${p.impactA === 'increase_risk' ? 'pos' : 'neg'}">
                      ${p.shapA >= 0 ? '+' : ''}${p.shapA.toFixed(3)}</span></td>
                    <td><span class="feat-label">${p.labelB}</span></td>
                    <td><span class="shap-pill ${p.impactB === 'increase_risk' ? 'pos' : 'neg'}">
                      ${p.shapB >= 0 ? '+' : ''}${p.shapB.toFixed(3)}</span></td>
                    <td><strong class="${p.combined >= 0 ? 'text-risk' : 'text-safe'}">
                      ${p.combined >= 0 ? '+' : ''}${p.combined.toFixed(3)}</strong></td>
                    <td><span class="depth-dir-badge ${DIR_LABELS[p.direction].cls}">
                      ${DIR_LABELS[p.direction].badge}</span></td>
                  </tr>`).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>`;

    // Draw Plotly heatmap
    _drawPairHeatmap('live-pair-heatmap', z, labels);
  }

  function _drawPairHeatmap(elId, z, labels) {
    const el = document.getElementById(elId);
    if (!el) return;

    // Flatten for colour scale centering
    const flat   = z.flat();
    const absMax = Math.max(...flat.map(Math.abs), 0.01);

    Plotly.react(el, [{
      type: 'heatmap',
      z,
      x: labels,
      y: labels,
      colorscale: [
        [0,   '#3B82F6'],
        [0.5, '#1E293B'],
        [1,   '#EF4444'],
      ],
      zmin: -absMax,
      zmax:  absMax,
      showscale: true,
      colorbar: {
        thickness: 10,
        tickfont:  { size: 9 },
        title:     { text: 'SHAP', font: { size: 9 }, side: 'right' },
        len:       0.7,
      },
      hoverongaps: false,
      hovertemplate:
        '<b>%{y}</b> × <b>%{x}</b><br>Combined SHAP: %{z:.3f}<extra></extra>',
      text:       z.map(row => row.map(v => (v >= 0 ? '+' : '') + v.toFixed(3))),
      texttemplate: '%{text}',
      textfont:   { size: 10, color: 'white' },
    }], {
      paper_bgcolor: 'var(--color-surface, #1e293b)',
      plot_bgcolor:  'var(--color-surface, #1e293b)',
      font:          { family: 'Inter, sans-serif', color: 'var(--color-text-primary, #e2e8f0)', size: 11 },
      height:        330,
      margin:        { l: 130, r: 80, t: 20, b: 100, pad: 4 },
      xaxis:         { tickangle: -35, automargin: true },
      yaxis:         { automargin: true },
    }, { responsive: true, displayModeBar: false });
  }

  // ------------------------------------------------------------------ //
  // DEPTH 3 — Triple ranked horizontal bar chart
  // ------------------------------------------------------------------ //
  function _renderDepth3(drivers, pred) {
    const container = document.getElementById('live-depth-content');
    if (!container) return;

    const triples = comb(drivers, 3)
      .map(([a, b, c]) => ({
        label:     [a, b, c].map(d => _shortLabel(d.feature)).join(' × '),
        features:  [a.feature, b.feature, c.feature],
        shaps:     [a.shap_value, b.shap_value, c.shap_value],
        impacts:   [a.impact, b.impact, c.impact],
        combined:  a.shap_value + b.shap_value + c.shap_value,
        direction: _direction([a.shap_value, b.shap_value, c.shap_value]),
      }))
      .sort((a, b) => Math.abs(b.combined) - Math.abs(a.combined));

    container.innerHTML = `
      <div class="live-section-col">
        <div class="chart-title" style="font-size:.82rem;margin-bottom:.4rem;">
          Triple Combinations — ${triples.length} combinations
          <span class="chart-caption" style="margin-left:.5rem;">
            Ranked by |combined SHAP|. Bar width = combined contribution.
          </span>
        </div>
        <div id="live-triple-chart" class="plotly-container" style="min-height:${Math.max(280, triples.length * 46 + 80)}px;"></div>
        <div class="data-table-wrap" style="margin-top:1rem;">
          <table class="data-table">
            <thead>
              <tr>
                <th>#</th><th>Triple Combination</th>
                <th>SHAP A</th><th>SHAP B</th><th>SHAP C</th>
                <th>Combined</th><th>Direction</th>
              </tr>
            </thead>
            <tbody>
              ${triples.map((t, i) => `
                <tr class="${i === 0 ? 'dominant-pair-row' : ''}">
                  <td>${i + 1}</td>
                  <td><span class="feat-label-triple">${t.label}</span></td>
                  ${t.shaps.map((s, si) => `
                    <td><span class="shap-pill ${t.impacts[si] === 'increase_risk' ? 'pos' : 'neg'}">
                      ${s >= 0 ? '+' : ''}${s.toFixed(3)}</span></td>`).join('')}
                  <td><strong class="${t.combined >= 0 ? 'text-risk' : 'text-safe'}">
                    ${t.combined >= 0 ? '+' : ''}${t.combined.toFixed(3)}</strong></td>
                  <td><span class="depth-dir-badge ${DIR_LABELS[t.direction].cls}">
                    ${DIR_LABELS[t.direction].badge}</span></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;

    _drawTripleChart('live-triple-chart', triples);
  }

  function _drawTripleChart(elId, triples) {
    const el = document.getElementById(elId);
    if (!el) return;

    const colors = triples.map(t =>
      t.direction === 'amplifying' ? '#EF4444'
      : t.direction === 'conflicting' ? '#3B82F6'
      : '#F59E0B'
    );

    Plotly.react(el, [{
      type:        'bar',
      orientation: 'h',
      x:           triples.map(t => t.combined),
      y:           triples.map(t => t.label),
      marker:      { color: colors, opacity: 0.85 },
      hovertemplate: '<b>%{y}</b><br>Combined SHAP: %{x:.3f}<extra></extra>',
    }], {
      paper_bgcolor: 'var(--color-surface, #1e293b)',
      plot_bgcolor:  'var(--color-chart-bg, #162032)',
      font:          { family: 'Inter, sans-serif', color: 'var(--color-text-primary, #e2e8f0)', size: 10 },
      height:        Math.max(280, triples.length * 46 + 80),
      margin:        { l: 20, r: 60, t: 20, b: 50, pad: 4 },
      xaxis: {
        title:      { text: 'Combined SHAP (log-odds)', font: { size: 10 } },
        gridcolor:  'rgba(148,163,184,0.1)',
        zerolinecolor: '#475569',
        zerolinewidth: 1.5,
      },
      yaxis:         { automargin: true, tickfont: { size: 9 } },
      showlegend:    false,
    }, { responsive: true, displayModeBar: false });
  }

  // ------------------------------------------------------------------ //
  // DEPTH 4 — Quad dominant risk card + ranked table
  // ------------------------------------------------------------------ //
  function _renderDepth4(drivers, pred) {
    const container = document.getElementById('live-depth-content');
    if (!container) return;

    const quads = comb(drivers, 4)
      .map(([a, b, c, d]) => ({
        label:     [a, b, c, d].map(x => _shortLabel(x.feature)).join(' × '),
        features:  [a, b, c, d].map(x => x.feature),
        shaps:     [a, b, c, d].map(x => x.shap_value),
        impacts:   [a, b, c, d].map(x => x.impact),
        combined:  a.shap_value + b.shap_value + c.shap_value + d.shap_value,
        direction: _direction([a.shap_value, b.shap_value, c.shap_value, d.shap_value]),
      }))
      .sort((a, b) => Math.abs(b.combined) - Math.abs(a.combined));

    const top   = quads[0];
    const level = !top ? 'neutral'
                : top.combined >= 0.5  ? 'high'
                : top.combined >= 0.2  ? 'medium' : 'low';

    container.innerHTML = `
      <div class="live-section-col">

        <!-- Dominant quad risk card -->
        ${top ? `
        <div class="dominant-quad-card ${level}">
          <div class="quad-card-title">⚡ Dominant 4-Way Combination</div>
          <div class="quad-card-combo">${top.label}</div>
          <div class="quad-card-meta">
            <span class="quad-combined ${top.combined >= 0 ? 'pos' : 'neg'}">
              Combined SHAP: ${top.combined >= 0 ? '+' : ''}${top.combined.toFixed(3)}
            </span>
            <span class="depth-dir-badge ${DIR_LABELS[top.direction].cls}">
              ${DIR_LABELS[top.direction].badge}
            </span>
          </div>
          <div class="quad-card-breakdown">
            ${top.features.map((f, i) => `
              <div class="quad-factor">
                <span class="quad-feat-name">${_shortLabel(f)}</span>
                <span class="shap-pill ${top.impacts[i] === 'increase_risk' ? 'pos' : 'neg'}">
                  ${top.shaps[i] >= 0 ? '+' : ''}${top.shaps[i].toFixed(3)}
                </span>
              </div>`).join('')}
          </div>
        </div>` : ''}

        <!-- All quads table -->
        <div class="chart-title" style="font-size:.82rem;margin:.8rem 0 .4rem;">
          All Quad Combinations — ${quads.length} combination${quads.length !== 1 ? 's' : ''}
        </div>
        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>#</th><th>Quad Combination</th>
                <th>SHAP A</th><th>SHAP B</th><th>SHAP C</th><th>SHAP D</th>
                <th>Combined</th><th>Direction</th>
              </tr>
            </thead>
            <tbody>
              ${quads.map((q, i) => `
                <tr class="${i === 0 ? 'dominant-pair-row' : ''}">
                  <td>${i + 1}</td>
                  <td><span class="feat-label-triple">${q.label}</span></td>
                  ${q.shaps.map((s, si) => `
                    <td><span class="shap-pill ${q.impacts[si] === 'increase_risk' ? 'pos' : 'neg'}">
                      ${s >= 0 ? '+' : ''}${s.toFixed(3)}</span></td>`).join('')}
                  <td><strong class="${q.combined >= 0 ? 'text-risk' : 'text-safe'}">
                    ${q.combined >= 0 ? '+' : ''}${q.combined.toFixed(3)}</strong></td>
                  <td><span class="depth-dir-badge ${DIR_LABELS[q.direction].cls}">
                    ${DIR_LABELS[q.direction].badge}</span></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }

  // ------------------------------------------------------------------ //
  // Placeholder
  // ------------------------------------------------------------------ //
  function _showLivePlaceholder(msg) {
    const header  = document.getElementById('live-interaction-header');
    const content = document.getElementById('live-depth-content');
    if (header)  header.className  = 'live-interaction-header neutral';
    if (header)  header.innerHTML  = '';
    if (content) content.innerHTML = `
      <div class="live-placeholder">
        <div class="placeholder-icon">⚡</div>
        <div>${msg ?? 'Submit a claim to activate live interaction analysis.'}</div>
      </div>`;
  }

  // ================================================================== //
  // STATIC POPULATION DATA (unchanged)
  // ================================================================== //

  const PAIRS = [
    { icd: 'External_Causes',         cpt: 'Pathology_Laboratory', claims: 2,   icdImpact: 12.56, cptImpact: 12.65, combined: 25.21 },
    { icd: 'Dermatology',             cpt: 'Pathology_Laboratory', claims: 42,  icdImpact: 10.59, cptImpact: 11.52, combined: 22.11 },
    { icd: 'Infectious',              cpt: 'Pathology_Laboratory', claims: 18,  icdImpact:  8.91, cptImpact: 10.80, combined: 19.71 },
    { icd: 'Neurology',               cpt: 'Pathology_Laboratory', claims: 25,  icdImpact:  8.52, cptImpact:  9.05, combined: 17.57 },
    { icd: 'Eye_Ear',                 cpt: 'Pathology_Laboratory', claims:  9,  icdImpact:  8.77, cptImpact:  8.44, combined: 17.21 },
    { icd: 'Trauma_Burns_Poisoning',  cpt: 'Pathology_Laboratory', claims:  6,  icdImpact:  2.36, cptImpact: 11.89, combined: 14.26 },
    { icd: 'General_Symptoms',        cpt: 'Pathology_Laboratory', claims: 316, icdImpact:  3.22, cptImpact:  9.36, combined: 12.58 },
    { icd: 'OBGYN',                   cpt: 'Pathology_Laboratory', claims:  3,  icdImpact:  5.22, cptImpact:  7.31, combined: 12.53 },
    { icd: 'Infectious',              cpt: 'Surgery',              claims:  1,  icdImpact:  7.27, cptImpact:  4.53, combined: 11.80 },
    { icd: 'Psychiatry',              cpt: 'Pathology_Laboratory', claims: 11,  icdImpact:  3.50, cptImpact:  7.13, combined: 10.62 },
    { icd: 'Oncology_Hematology',     cpt: 'Surgery',              claims:  1,  icdImpact: -3.68, cptImpact: 13.37, combined:  9.69 },
    { icd: 'Pulmonology',             cpt: 'Pathology_Laboratory', claims: 38,  icdImpact:  0.86, cptImpact:  8.80, combined:  9.66 },
    { icd: 'Gastroenterology_Dental', cpt: 'Pathology_Laboratory', claims: 87,  icdImpact:  0.06, cptImpact:  9.54, combined:  9.60 },
    { icd: 'Musculoskeletal',         cpt: 'Pathology_Laboratory', claims: 124, icdImpact: -1.50, cptImpact:  8.42, combined:  6.92 },
    { icd: 'Neurology',               cpt: 'HCPCS_Supplies',       claims:  1,  icdImpact: 11.36, cptImpact: -4.76, combined:  6.60 },
    { icd: 'Dermatology',             cpt: 'Surgery',              claims:  3,  icdImpact: 11.12, cptImpact: -4.69, combined:  6.43 },
  ];

  const ICD_GROUPS = [
    'Cardiology','Dermatology','Endocrinology','External_Causes',
    'Gastroenterology_Dental','General_Symptoms','Genitourinary',
    'Infectious','Musculoskeletal','Neurology','OBGYN',
    'Oncology_Hematology','Psychiatry','Pulmonology','Trauma_Burns_Poisoning',
  ];
  const CPT_GROUPS = [
    'Anesthesia','Evaluation_Management','HCPCS_Supplies',
    'Medicine','Pathology_Laboratory','Radiology','Surgery',
  ];

  function renderPairsTable(activeIcd, activeCpt) {
    const tbody = document.getElementById('pairs-table-body');
    if (!tbody) return;
    tbody.innerHTML = PAIRS.map((p, i) => {
      const isActive = p.icd === activeIcd;
      const icdImpactStr = (p.icdImpact >= 0 ? '+' : '') + p.icdImpact.toFixed(2) + '%';
      const cptImpactStr = (p.cptImpact >= 0 ? '+' : '') + p.cptImpact.toFixed(2) + '%';
      return `
        <tr class="${isActive ? 'active-row' : ''}">
          <td>${i + 1}</td>
          <td>${p.icd.replace(/_/g,' ')}</td>
          <td>${p.cpt.replace(/_/g,' ')}</td>
          <td>${p.claims}</td>
          <td>${icdImpactStr}</td>
          <td>${cptImpactStr}</td>
          <td><strong>${p.combined.toFixed(2)}%</strong></td>
        </tr>`;
    }).join('');
  }

  function buildMatrix(valueKey) {
    return ICD_GROUPS.map(icd =>
      CPT_GROUPS.map(cpt => {
        const found = PAIRS.find(p => p.icd === icd && p.cpt === cpt);
        return found ? found[valueKey] : 0;
      })
    );
  }

  function renderSHAPHeatmap() {
    const z = buildMatrix('combined');
    Charts.heatmap(
      'interactions-shap-heatmap',
      z,
      CPT_GROUPS.map(g => g.replace(/_/g,' ')),
      ICD_GROUPS.map(g => g.replace(/_/g,' ')),
      'ICD Group × CPT Group — Mean SHAP Interaction',
      '.2f',
    );
  }

  function renderDenialHeatmap() {
    const z = buildMatrix('combined');
    Charts.heatmap(
      'interactions-denial-heatmap',
      z,
      CPT_GROUPS.map(g => g.replace(/_/g,' ')),
      ICD_GROUPS.map(g => g.replace(/_/g,' ')),
      'ICD × CPT Group — Denial Rate Heatmap (Actual %)',
      '.2f',
    );
  }

  return {
    cptGroupFromCode,
    renderLive,
    renderPairsTable,
    renderSHAPHeatmap,
    renderDenialHeatmap,
    PAIRS,
  };

})();
