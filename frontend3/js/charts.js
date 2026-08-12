/**
 * js/charts.js
 * ------------
 * All Plotly.js figure renderers.
 * Each function receives plain data and a DOM element id, renders into it.
 *
 * Dependencies: Plotly (loaded via CDN in index.html)
 *
 * Exports (global Charts object)
 * --------------------------------
 *   Charts.gauge(elId, prob_pct)
 *   Charts.waterfall(elId, contributions, baseLogOdds, finalLogOdds)
 *   Charts.featureScatter(elId, contributions)
 *   Charts.dependence(elId, claimNet)
 *   Charts.denialReasonBar(elId)
 *   Charts.heatmap(elId, zData, xLabels, yLabels, title, fmt)
 */

const Charts = (() => {

  const FONT     = { family: 'Inter, system-ui, sans-serif', size: 12 };
  const PAPER_BG = 'rgba(0,0,0,0)';
  const PLOT_BG  = 'rgba(0,0,0,0)';
  const BASE_LAYOUT = {
    paper_bgcolor: PAPER_BG,
    plot_bgcolor:  PLOT_BG,
    font: FONT,
    margin: { l: 28, r: 24, t: 40, b: 28 },
  };

  const COLORS = {
    high:   '#EF4444',
    medium: '#F59E0B',
    low:    '#10B981',
    accent: '#3B82F6',
    dark:   '#1E293B',
    muted:  '#94A3B8',
  };

  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ------------------------------------------------------------------ //
  // 1. Denial Probability Gauge
  // ------------------------------------------------------------------ //
  function gauge(elId, prob_pct) {
    const data = [{
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: prob_pct,
      number: { suffix: '%', font: { size: 34, color: COLORS.dark } },
      delta: {
        reference: 42,
        increasing: { color: COLORS.high },
        decreasing: { color: COLORS.low },
        suffix: '%',
      },
      title: { text: 'Denial Probability', font: { size: 13, color: COLORS.muted } },
      gauge: {
        axis: { range: [0, 100], tickwidth: 1, tickcolor: '#CBD5E1', tickfont: { size: 10 } },
        bar:  { color: COLORS.dark, thickness: 0.25 },
        bgcolor: 'white',
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
    }];
    Plotly.react(elId, data, { ...BASE_LAYOUT, height: 270 }, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 2. SHAP Waterfall
  // ------------------------------------------------------------------ //
  function waterfall(elId, contributions, baseLogOdds, finalLogOdds) {
    const labels = ['Base', ...contributions.map(c => c.name), 'Final'];
    const values = [baseLogOdds, ...contributions.map(c => c.value), finalLogOdds];
    const measure = ['absolute', ...contributions.map(() => 'relative'), 'total'];

    const data = [{
      type: 'waterfall',
      orientation: 'h',
      measure,
      y: labels,
      x: values,
      connector: { line: { color: '#E2E8F0', width: 1 } },
      decreasing: { marker: { color: COLORS.low } },
      increasing: { marker: { color: COLORS.high } },
      totals:     { marker: { color: COLORS.muted } },
      text: values.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
      textposition: 'outside',
    }];

    const layout = {
      ...BASE_LAYOUT,
      height: Math.max(320, 40 * labels.length),
      xaxis: { title: { text: 'SHAP Log-Odds Contribution', font: { size: 11 } } },
      yaxis: { autorange: 'reversed' },
      title: { text: 'SHAP Waterfall', font: { size: 14 } },
    };
    Plotly.react(elId, data, layout, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 3. Feature Value vs SHAP Scatter
  // ------------------------------------------------------------------ //
  function featureScatter(elId, contributions) {
    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.value)), 1e-9);
    const data = [{
      type: 'scatter',
      mode: 'markers',
      x: contributions.map(c => c.value),
      y: contributions.map(c => c.name),
      marker: {
        size: 10,
        opacity: 0.85,
        color: contributions.map(c => Math.abs(c.value) / maxAbs),
        colorscale: [[0, '#3B82F6'], [1, '#EF4444']],
        showscale: true,
        colorbar: { title: { text: 'Feature<br>Value', font: { size: 10 } }, thickness: 10 },
      },
      text: contributions.map(c => `${c.name}: ${c.value >= 0 ? '+' : ''}${c.value.toFixed(4)}`),
      hoverinfo: 'text',
    }];

    const layout = {
      ...BASE_LAYOUT,
      height: 380,
      title: { text: 'Feature Value vs SHAP Value', font: { size: 13 } },
      xaxis: { title: { text: 'SHAP Value (impact on log-odds)', font: { size: 11 } } },
      yaxis: { autorange: 'reversed' },
    };
    Plotly.react(elId, data, layout, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 4. Claim Amount SHAP Dependence
  // ------------------------------------------------------------------ //
  function dependence(elId, claimNet) {
    // Simulate a small neighbourhood around the actual claim amount
    const n = 60;
    const lo = Math.max(claimNet * 0.2, 1);
    const hi = claimNet * 2.5 + 1;
    const xs = Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1));
    const logVals    = xs.map(v => Math.log1p(v));
    const shapVals   = logVals.map(l => l / 8 - 0.4 + (Math.random() * 0.06 - 0.03));
    const mismatch   = xs.map(() => Math.random());

    // Actual point
    const actualLog  = Math.log1p(claimNet);
    const actualShap = actualLog / 8 - 0.4;

    const data = [
      {
        type: 'scatter', mode: 'markers',
        x: logVals, y: shapVals,
        marker: {
          size: 6, opacity: 0.7,
          color: mismatch,
          colorscale: [[0, '#3B82F6'], [1, '#EF4444']],
          showscale: true,
          colorbar: { title: { text: 'ICD–CPT<br>Mismatch', font: { size: 9 } }, thickness: 10 },
        },
        name: 'Similar claims',
      },
      {
        type: 'scatter', mode: 'markers',
        x: [actualLog], y: [actualShap],
        marker: { size: 13, color: COLORS.dark, symbol: 'diamond' },
        name: 'This claim',
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      height: 320,
      title: { text: 'SHAP Dependence — Claim Amount', font: { size: 13 } },
      xaxis: { title: { text: 'Claim Net (log)', font: { size: 11 } } },
      yaxis: { title: { text: 'SHAP Value', font: { size: 11 } } },
      showlegend: true,
      legend: { font: { size: 11 } },
    };
    Plotly.react(elId, data, layout, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 5. Denial Reason Distribution Bar
  // ------------------------------------------------------------------ //
  function denialReasonBar(elId) {
    const labels = [
      'No Prior Auth', 'ICD–CPT Mismatch', 'High Claim Value',
      'Short Billing Lag', 'Out-of-Network', 'LOS Outlier',
    ];
    const values = [38.0, 24.0, 16.0, 11.0, 7.0, 4.0];
    const colors = values.map(v => v === Math.max(...values) ? COLORS.high : '#94A3B8');

    const data = [{
      type: 'bar', orientation: 'h',
      x: values, y: labels,
      marker: { color: colors },
      text: values.map(v => v.toFixed(1) + '%'),
      textposition: 'outside',
    }];

    const layout = {
      ...BASE_LAYOUT,
      height: 260,
      title: { text: 'Denial Reason Distribution — Similar Claims', font: { size: 13 } },
      xaxis: { title: { text: '% of Similar Claims', font: { size: 11 } }, range: [0, 50] },
      yaxis: { autorange: 'reversed' },
    };
    Plotly.react(elId, data, layout, PLOTLY_CFG);
  }

  // ------------------------------------------------------------------ //
  // 6. Generic Heatmap (ICD×Encounter, ICD×CPT)
  // ------------------------------------------------------------------ //
  function heatmap(elId, zData, xLabels, yLabels, title = '', fmt = '.1f') {
    const textArr = zData.map(row => row.map(v => {
      if (fmt === '.1f')  return v.toFixed(1) + '%';
      if (fmt === '.2f')  return v.toFixed(2);
      return String(v);
    }));

    const data = [{
      type: 'heatmap',
      z: zData, x: xLabels, y: yLabels,
      colorscale: [[0, '#EFF6FF'], [0.5, '#BFDBFE'], [1, '#1E40AF']],
      text: textArr,
      texttemplate: '%{text}',
      colorbar: { thickness: 10, title: { text: '', font: { size: 10 } } },
      hoverongaps: false,
    }];

    const layout = {
      ...BASE_LAYOUT,
      height: 420,
      title: { text: title, font: { size: 13 } },
      xaxis: { tickangle: -35, tickfont: { size: 10 } },
      yaxis: { tickfont: { size: 10 } },
      margin: { l: 160, r: 24, t: 50, b: 80 },
    };
    Plotly.react(elId, data, layout, PLOTLY_CFG);
  }

  return { gauge, waterfall, featureScatter, dependence, denialReasonBar, heatmap };
})();
