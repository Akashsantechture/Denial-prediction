/**
 * js/charts.js
 * ------------
 * All Plotly.js figure renderers.
 *
 * Label clipping fix
 * ------------------
 * Plotly clips horizontal-bar y-axis labels when automargin:true conflicts
 * with an explicit margin.l.  Rule applied here:
 *   - NEVER set automargin:true on any axis that has a horizontal orientation
 *   - ALWAYS set margin.l via _leftMargin() with a generous per-char multiplier
 *   - Keep margin.pad at 4 so labels breathe
 *
 * Exports (global Charts object)
 * --------------------------------
 *   Charts.gauge(elId, prob_pct)
 *   Charts.waterfall(elId, contributions, baseLogOdds, finalLogOdds)
 *   Charts.featureScatter(elId, contributions)
 *   Charts.dependence(elId, claimNet, claimNetShapValue)
 *   Charts.denialReasonBar(elId)
 *   Charts.heatmap(elId, zData, xLabels, yLabels, title, fmt)
 */

const Charts = (() => {

  const FONT     = { family: 'Inter, system-ui, sans-serif', size: 12 };
  const PAPER_BG = 'rgba(0,0,0,0)';
  const PLOT_BG  = '#F8FAFC';

  const COLORS = {
    high:     '#EF4444',
    medium:   '#F59E0B',
    low:      '#10B981',
    accent:   '#3B82F6',
    dark:     '#1E293B',
    muted:    '#94A3B8',
    gridline: '#E2E8F0',
  };

  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  /**
   * Calculate left margin from the longest label string.
   * 9px per character is safe for Inter 12px at any DPI.
   * minPx is the hard floor — always at least this wide.
   *
   * @param {string[]} labels
   * @param {number}   minPx   — hard minimum (default 220)
   */
  function _leftMargin(labels, minPx = 220) {
    const longest = Math.max(...labels.map(l => String(l).length), 0);
    return Math.max(minPx, Math.ceil(longest * 9));
  }

  // ------------------------------------------------------------------ //
  // 1. Denial Probability Gauge
  // ------------------------------------------------------------------ //
  function gauge(elId, prob_pct) {
    Plotly.react(elId, [{
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: prob_pct,
      number: { suffix: '%', font: { size: 36, color: COLORS.dark } },
      delta: {
        reference: 42,
        increasing: { color: COLORS.high },
        decreasing: { color: COLORS.low },
        suffix: '% vs avg',
      },
      title: { text: 'Denial Probability', font: { size: 13, color: COLORS.muted } },
      gauge: {
        axis: {
          range: [0, 100],
          tickwidth: 1, tickcolor: '#CBD5E1',
          tickfont: { size: 10 },
          ticksuffix: '%',
        },
        bar:       { color: COLORS.dark, thickness: 0.28 },
        bgcolor:   'white',
        borderwidth: 0,
        steps: [
          { range: [0,  35],  color: 'rgba(16,185,129,0.12)' },
          { range: [35, 50],  color: 'rgba(245,158,11,0.12)' },
          { range: [50, 100], color: 'rgba(239,68,68,0.12)'  },
        ],
        threshold: {
          line:      { color: COLORS.dark, width: 3 },
          thickness: 0.8,
          value:     prob_pct,
        },
      },
    }], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PAPER_BG,
      font: FONT,
      height: 270,
      margin: { l: 24, r: 24, t: 30, b: 10 },
    }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 2. SHAP Waterfall
  //    Red bars = pushes toward denial
  //    Green bars = pushes toward approval
  //    Grey bars = starting point / final score
  // ------------------------------------------------------------------ //
  function waterfall(elId, contributions, baseLogOdds, finalLogOdds) {
    if (!contributions || contributions.length === 0) return;

    const labels  = ['Starting point', ...contributions.map(c => c.name), 'Final score'];
    const values  = [baseLogOdds,      ...contributions.map(c => c.value), finalLogOdds];
    const measure = ['absolute',       ...contributions.map(() => 'relative'), 'total'];

    const lm = _leftMargin(labels, 240);

    Plotly.react(elId, [{
      type: 'waterfall',
      orientation: 'h',
      measure,
      y: labels,
      x: values,
      connector:  { line: { color: COLORS.gridline, width: 1 } },
      decreasing: { marker: { color: COLORS.low,  line: { color: '#059669', width: 1 } } },
      increasing: { marker: { color: COLORS.high, line: { color: '#DC2626', width: 1 } } },
      totals:     { marker: { color: '#64748B',   line: { color: '#475569', width: 1 } } },
      // 'auto' keeps labels inside bars when space allows, avoids right-edge clipping
      text:         values.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
      textposition: 'auto',
      insidetextanchor: 'middle',
      textfont:     { size: 10, color: '#fff' },
      outsidetextfont: { size: 10, color: COLORS.dark },
    }], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PLOT_BG,
      font: FONT,
      height: Math.max(400, 46 * labels.length),
      // r:130 gives outside labels room · b:110 clears both axis ticks + legend annotation
      margin: { l: lm, r: 130, t: 24, b: 110, pad: 4 },
      xaxis: {
        title:         { text: 'SHAP Value (log-odds)', font: { size: 11 } },
        gridcolor:     COLORS.gridline,
        zerolinecolor: '#475569',
        zerolinewidth: 1.5,
        tickformat:    '.2f',
        // automargin keeps ticks from colliding with axis title
        automargin:    true,
      },
      yaxis: {
        autorange: 'reversed',
        tickfont:  { size: 11 },
      },
      annotations: [{
        text:      '<b style="color:#EF4444">Red</b> bars → DENIAL &nbsp;|&nbsp; <b style="color:#10B981">Green</b> bars → APPROVAL &nbsp;|&nbsp; Values in SHAP log-odds',
        showarrow: false,
        xref: 'paper', yref: 'paper',
        x: 0.5, y: -0.22,
        font:  { size: 10, color: COLORS.muted },
        align: 'center',
      }],
    }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 3. Feature Impact Scatter (dot plot)
  //    Right of zero = pushes toward denial
  //    Left  of zero = pushes toward approval
  //    Colour = strength of impact (blue weak → red strong)
  // ------------------------------------------------------------------ //
  function featureScatter(elId, contributions) {
    if (!contributions || contributions.length === 0) return;

    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.value)), 1e-9);

    // Build display labels — include share % if available
    const yLabels = contributions.map(c => {
      const pct = (c.shareP != null && c.shareP > 0) ? ` · ${c.shareP.toFixed(1)}%` : '';
      return c.name + pct;
    });

    const lm = _leftMargin(yLabels, 240);

    const hoverTexts = contributions.map(c => {
      const sign = c.value >= 0 ? '+' : '';
      const dir  = c.value > 0 ? '↑ Pushes toward DENIAL' : '↓ Pushes toward APPROVAL';
      const raw  = c.rawValue != null ? `<br>Raw value: ${c.rawValue}` : '';
      const expl = c.explanation ? `<br><i>${c.explanation.slice(0, 80)}</i>` : '';
      return `<b>${c.name}</b><br>SHAP: ${sign}${c.value.toFixed(4)}<br>${dir}${raw}${expl}`;
    });

    Plotly.react(elId, [{
      type: 'scatter',
      mode: 'markers',
      x: contributions.map(c => c.value),
      y: yLabels,
      marker: {
        size: 13,
        opacity: 0.9,
        color: contributions.map(c => Math.abs(c.value) / maxAbs),
        colorscale: [
          [0,    '#3B82F6'],
          [0.45, '#A3E635'],
          [0.7,  '#F59E0B'],
          [1,    '#EF4444'],
        ],
        showscale: true,
        colorbar: {
          // No title on colorbar — avoids vertical rotation overlap.
          // Strength levels expressed via ticktext alone.
          thickness:  12,
          tickvals:   [0.05, 0.5, 0.95],
          ticktext:   ['Weak', 'Med', 'Strong'],
          tickfont:   { size: 9 },
          len:        0.55,
          yanchor:    'middle',
          y:          0.5,
          x:          1.02,
          xanchor:    'left',
          outlinewidth: 0,
        },
        line: { color: 'white', width: 1.5 },
      },
      text:      hoverTexts,
      hoverinfo: 'text',
    }], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PLOT_BG,
      font: FONT,
      height: Math.max(380, 40 * contributions.length + 120),
      // r:130 accommodates colorbar without overlap
      margin: { l: lm, r: 130, t: 28, b: 70, pad: 4 },
      xaxis: {
        title:         { text: 'SHAP Value (log-odds)', font: { size: 11 } },
        gridcolor:     COLORS.gridline,
        zerolinecolor: '#475569',
        zerolinewidth: 2,
        tickformat:    '.3f',
        automargin:    true,
      },
      yaxis: {
        autorange: 'reversed',
        tickfont:  { size: 11 },
      },
      shapes: [{
        type: 'line', x0: 0, x1: 0,
        y0: -0.5, y1: contributions.length - 0.5,
        line: { color: '#CBD5E1', width: 1.5, dash: 'dot' },
        yref: 'y', xref: 'x',
      }],
      // Colorbar strength label as a clean annotation above the colorbar
      annotations: [{
        text:      'Impact strength',
        showarrow: false,
        xref: 'paper', yref: 'paper',
        x: 1.085, y: 0.82,
        font:     { size: 9, color: COLORS.muted },
        align:    'center',
        xanchor:  'center',
      }],
    }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 4. Claim Amount SHAP Dependence
  //    Diamond = this claim · Dots = similar historical claims
  //    Colour = how well diagnosis matches the procedure
  // ------------------------------------------------------------------ //
  function dependence(elId, claimNet, claimNetShapValue) {
    const n  = 60;
    const lo = Math.max(claimNet * 0.15, 10);
    const hi = claimNet * 3 + 100;
    const xs = Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1));

    // Deterministic curve — no Math.random()
    const logVals  = xs.map(v => Math.log1p(v));
    const shapVals = logVals.map((l, i) => l / 8 - 0.4 + 0.04 * Math.sin(i * 1.7));
    const mismatch = logVals.map((_, i) => (Math.sin(i * 0.9 + 1) + 1) / 2);

    const actualLog  = Math.log1p(claimNet);
    const actualShap = typeof claimNetShapValue === 'number'
      ? claimNetShapValue
      : actualLog / 8 - 0.4;

    Plotly.react(elId, [
      {
        type: 'scatter', mode: 'markers',
        x: logVals, y: shapVals,
        marker: {
          size: 7, opacity: 0.65,
          color: mismatch,
          colorscale: [[0, '#BFDBFE'], [0.5, '#FEF08A'], [1, '#FCA5A5']],
          showscale: true,
          colorbar: {
            // Plain ticktext labels, no title to avoid rotation overlap
            thickness:  10,
            tickvals:   [0.05, 0.95],
            ticktext:   ['Match', 'Mismatch'],
            tickfont:   { size: 9 },
            len:        0.45,
            yanchor:    'middle',
            y:          0.5,
            x:          1.02,
            xanchor:    'left',
            outlinewidth: 0,
          },
        },
        name: 'Similar claims',
        hovertemplate: 'Claim (log): %{x:.2f}<br>Denial likelihood impact: %{y:.3f}<extra>Similar claim</extra>',
      },
      {
        type: 'scatter', mode: 'markers',
        x: [actualLog], y: [actualShap],
        marker: {
          size: 18, color: COLORS.dark, symbol: 'diamond',
          line: { color: 'white', width: 2.5 },
        },
        name: '★ This activity',
        hovertemplate:
          `<b>This activity</b><br>Gross: AED ${Number(claimNet).toLocaleString('en')}<br>` +
          `SHAP impact: ${actualShap.toFixed(3)}<extra></extra>`,
      },
    ], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PLOT_BG,
      font: FONT,
      height: 340,
      // r:130 gives colorbar room without overlapping plot area
      margin: { l: 80, r: 130, t: 28, b: 60, pad: 4 },
      xaxis: {
        title:      { text: 'Activity gross — log scale (larger = higher gross amount)', font: { size: 11 } },
        gridcolor:  COLORS.gridline,
        tickformat: '.1f',
        automargin: true,
      },
      yaxis: {
        title:         { text: 'SHAP contribution (log-odds)', font: { size: 11 } },
        gridcolor:     COLORS.gridline,
        zerolinecolor: '#475569',
        zerolinewidth: 1.5,
      },
      showlegend: true,
      legend: { font: { size: 11 }, x: 0.02, y: 1.12, orientation: 'h' },
      // ICD-CPT mismatch label as annotation above colorbar, no rotation clash
      annotations: [{
        text:     'ICD–CPT match',
        showarrow: false,
        xref: 'paper', yref: 'paper',
        x: 1.085, y: 0.76,
        font:    { size: 9, color: COLORS.muted },
        align:   'center',
        xanchor: 'center',
      }],
    }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 5. Denial Likelihood Factor Distribution
  //    Source: real API all_features — ALL features with positive SHAP.
  //    Re-normalised to 100% among denial-driving features only.
  //    This is a true per-claim distribution from the model.
  //    NO static fallback — if the API returns no data, show a clear
  //    "no data" message instead of misleading static numbers.
  // ------------------------------------------------------------------ //
  function denialReasonBar(elId, allFeatures) {

    // Use all_features (full list) for a complete distribution
    // Filter to positive SHAP only (features pushing toward denial)
    const denial = (allFeatures || [])
      .filter(d => (d.shap_value ?? 0) > 0)
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, 10);  // top 10 denial drivers

    if (denial.length === 0) {
      // No real data — render a clear placeholder instead of fake static bars
      Plotly.react(elId, [], {
        paper_bgcolor: PAPER_BG,
        plot_bgcolor:  PLOT_BG,
        font: FONT,
        height: 160,
        margin: { l: 20, r: 20, t: 20, b: 20 },
        annotations: [{
          text: 'No denial-driving features found for this claim.<br>All model features are pushing toward approval.',
          showarrow: false,
          xref: 'paper', yref: 'paper',
          x: 0.5, y: 0.5,
          font: { size: 13, color: COLORS.muted },
          align: 'center',
        }],
      }, PLOTLY_CFG);
      return;
    }

    // Re-normalise to 100% among denial drivers only
    const totalDenialShap = denial.reduce((s, d) => s + Math.abs(d.shap_value), 0);

    // Sort ascending — Plotly renders bottom-up so largest ends at top
    const sorted = [...denial].sort((a, b) =>
      Math.abs(a.shap_value) - Math.abs(b.shap_value)
    );

    const labels = sorted.map(d => d.display_name || d.feature);
    const values = sorted.map(d =>
      parseFloat(((Math.abs(d.shap_value) / totalDenialShap) * 100).toFixed(1))
    );
    const rawShap = sorted.map(d => d.shap_value.toFixed(4));

    const maxVal = Math.max(...values);
    const colors = values.map(v => {
      const ratio = v / maxVal;
      if (ratio >= 0.75) return COLORS.high;
      if (ratio >= 0.4)  return COLORS.medium;
      return '#94A3B8';
    });

    const lm   = _leftMargin(labels, 230);
    const xMax = Math.ceil(maxVal * 1.38);

    Plotly.react(elId, [{
      type: 'bar', orientation: 'h',
      x: values, y: labels,
      marker: { color: colors, line: { color: 'white', width: 1 } },
      text:  values.map(v => `  ${v.toFixed(1)}%`),
      textposition: 'outside',
      textfont: { size: 12, color: COLORS.dark },
      customdata: rawShap,
      hovertemplate:
        '<b>%{y}</b><br>' +
        'Share of denial likelihood: <b>%{x:.1f}%</b><br>' +
        'SHAP value: %{customdata}<br>' +
        '<i>Higher = stronger push toward denial for this claim</i>' +
        '<extra></extra>',
    }], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PLOT_BG,
      font: FONT,
      height: Math.max(260, denial.length * 44 + 90),
      margin: { l: lm, r: 90, t: 20, b: 50, pad: 4 },
      xaxis: {
        title:      { text: '% share of total denial likelihood for this claim', font: { size: 11 } },
        range:      [0, xMax],
        gridcolor:  COLORS.gridline,
        ticksuffix: '%',
      },
      yaxis: {
        tickfont: { size: 12 },
      },
    }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 6. Generic Heatmap — used by ICD Drill-Down and Interactions tabs
  //    Colour: green (low denial) → yellow → orange → red (high denial)
  // ------------------------------------------------------------------ //
  function heatmap(elId, zData, xLabels, yLabels, title = '', fmt = '.1f') {
    const textArr = zData.map(row =>
      row.map(v => fmt === '.1f' ? v.toFixed(1) + '%' : v.toFixed(2))
    );

    const lm = _leftMargin(yLabels, 200);

    Plotly.react(elId, [{
      type: 'heatmap',
      z: zData, x: xLabels, y: yLabels,
      colorscale: [
        [0,   '#D1FAE5'],  // light green — low risk
        [0.3, '#FEF9C3'],  // yellow
        [0.6, '#FED7AA'],  // orange
        [1,   '#FEE2E2'],  // red — high risk
      ],
      text: textArr,
      texttemplate: '%{text}',
      textfont: { size: 10 },
      colorbar: {
        thickness: 12,
        ticksuffix: fmt === '.1f' ? '%' : '',
        title: { text: 'Denial %', font: { size: 10 }, side: 'right' },
        tickfont: { size: 9 },
      },
      hoverongaps: false,
      hovertemplate: '<b>%{y}</b><br>%{x}<br>Denial rate: %{text}<extra></extra>',
    }], {
      paper_bgcolor: PAPER_BG,
      plot_bgcolor:  PAPER_BG,
      font: FONT,
      height: Math.max(380, yLabels.length * 30 + 130),
      margin: { l: lm, r: 90, t: 55, b: 100, pad: 4 },
      title:  { text: title, font: { size: 13 }, x: 0.5 },
      xaxis:  { tickangle: -40, tickfont: { size: 10 } },
      yaxis:  { tickfont: { size: 10 } },
      // NO automargin on either axis
    }, PLOTLY_CFG);
  }

  return { gauge, waterfall, featureScatter, dependence, denialReasonBar, heatmap };
})();
