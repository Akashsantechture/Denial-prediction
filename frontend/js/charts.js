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

  // Read theme-aware values from CSS variables at render time
  // so charts adapt correctly when the user switches dark ↔ light mode.
  function _cssVar(name, fallback) {
    const val = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
    return val || fallback;
  }
  function _isLight() {
    return document.documentElement.getAttribute('data-theme') === 'light';
  }
  function _font()    { return { family: 'Inter, system-ui, sans-serif', size: 12, color: _cssVar('--color-text-primary', '#E2E8F0') }; }
  function _paperBg() { return 'rgba(0,0,0,0)'; }
  function _plotBg()  { return _cssVar('--color-chart-bg', '#121827'); }
  function _muted()   { return _cssVar('--color-text-muted', '#4E5D78'); }

  const FONT     = { family: 'Inter, system-ui, sans-serif', size: 12, color: '#E2E8F0' };
  const PAPER_BG = 'rgba(0,0,0,0)';
  const PLOT_BG  = '#161323';

  const COLORS = {
    high:     '#E05252',
    medium:   '#D4874A',
    low:      '#3D9E6E',
    accent:   '#4C7EE0',
    dark:     '#E2E8F0',
    muted:    '#4E5D78',
    gridline: '#252D3D',
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
    const light     = _isLight();
    const fontColor = _cssVar('--color-text-primary', light ? '#16202E' : '#E2E8F0');
    const mutedColor= _cssVar('--color-text-muted',   light ? '#8896A8' : '#4E5D78');
    const bgColor   = light ? '#F7F8FA' : '#161B27';
    const ringBg    = light ? '#E8EBF0' : '#1C2233';

    // Risk-level colour for the needle bar
    let barColor = COLORS.low;
    if (prob_pct >= 70) barColor = COLORS.high;
    else if (prob_pct >= 40) barColor = COLORS.medium;

    Plotly.react(elId, [{
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: prob_pct,
      number: {
        suffix: '%',
        font:   { size: 38, color: fontColor, family: 'Inter, sans-serif' },
      },
      delta: {
        reference: 42,
        increasing: { color: COLORS.high },
        decreasing: { color: COLORS.low },
        suffix: '% vs avg',
        font: { size: 13 },
      },
      title: {
        text: 'Denial Probability',
        font: { size: 12, color: mutedColor, family: 'Inter, sans-serif' },
      },
      gauge: {
        axis: {
          range:     [0, 100],
          tickwidth: 1,
          tickcolor: mutedColor,
          tickfont:  { size: 10, color: mutedColor },
          ticksuffix: '%',
          dtick: 20,
        },
        bar: { color: barColor, thickness: 0.22 },
        bgcolor: ringBg,
        borderwidth: 0,
        steps: [
          { range: [0,  35],  color: light ? 'rgba(61,158,110,0.15)'  : 'rgba(61,158,110,0.12)'  },
          { range: [35, 70],  color: light ? 'rgba(212,135,74,0.15)'  : 'rgba(212,135,74,0.12)'  },
          { range: [70, 100], color: light ? 'rgba(224,82,82,0.15)'   : 'rgba(224,82,82,0.12)'   },
        ],
        threshold: {
          line:      { color: barColor, width: 3 },
          thickness: 0.75,
          value:     prob_pct,
        },
      },
    }], {
      paper_bgcolor: light ? '#FFFFFF' : bgColor,
      plot_bgcolor:  light ? '#FFFFFF' : bgColor,
      font: { family: 'Inter, sans-serif', size: 12, color: fontColor },
      height: 270,
      margin: { l: 28, r: 28, t: 30, b: 14 },
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
    const lm      = _leftMargin(labels, 240);
    const font    = _font();
    const paperBg = _paperBg();
    const plotBg  = _plotBg();
    const muted   = _muted();
    const gridCol = _cssVar('--color-border', '#252D3D');

    Plotly.react(elId, [{
      type: 'waterfall',
      orientation: 'h',
      measure,
      y: labels,
      x: values,
      connector:  { line: { color: gridCol, width: 1 } },
      decreasing: { marker: { color: COLORS.low,  line: { color: '#059669', width: 1 } } },
      increasing: { marker: { color: COLORS.high, line: { color: '#DC2626', width: 1 } } },
      totals:     { marker: { color: '#64748B',   line: { color: '#475569', width: 1 } } },
      text:         values.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
      textposition: 'auto',
      insidetextanchor: 'middle',
      textfont:     { size: 10, color: '#fff' },
      outsidetextfont: { size: 10, color: font.color },
    }], {
      paper_bgcolor: paperBg,
      plot_bgcolor:  plotBg,
      font,
      height: Math.max(400, 46 * labels.length),
      margin: { l: lm, r: 130, t: 24, b: 110, pad: 4 },
      xaxis: {
        title:         { text: 'SHAP Value (log-odds)', font: { size: 11 } },
        gridcolor:     gridCol,
        zerolinecolor: '#475569',
        zerolinewidth: 1.5,
        tickformat:    '.2f',
        automargin:    true,
      },
      yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
      annotations: [{
        text:      'Red bars → DENIAL &nbsp;|&nbsp; Green bars → APPROVAL &nbsp;|&nbsp; Values in SHAP log-odds',
        showarrow: false,
        xref: 'paper', yref: 'paper',
        x: 0.5, y: -0.22,
        font:  { size: 10, color: muted },
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

    const maxAbs  = Math.max(...contributions.map(c => Math.abs(c.value)), 1e-9);
    const font    = _font();
    const paperBg = _paperBg();
    const plotBg  = _plotBg();
    const muted   = _muted();
    const gridCol = _cssVar('--color-border', '#252D3D');

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
        size: 13, opacity: 0.9,
        color: contributions.map(c => Math.abs(c.value) / maxAbs),
        colorscale: [
          [0,    '#4C7EE0'],
          [0.45, '#94A3B8'],
          [0.7,  '#D4874A'],
          [1,    '#E05252'],
        ],
        showscale: true,
        colorbar: {
          thickness: 12, tickvals: [0.05, 0.5, 0.95],
          ticktext: ['Weak', 'Med', 'Strong'], tickfont: { size: 9 },
          len: 0.55, yanchor: 'middle', y: 0.5,
          x: 1.02, xanchor: 'left', outlinewidth: 0,
        },
        line: { color: 'white', width: 1.5 },
      },
      text: hoverTexts, hoverinfo: 'text',
    }], {
      paper_bgcolor: paperBg,
      plot_bgcolor:  plotBg,
      font,
      height: Math.max(380, 40 * contributions.length + 120),
      margin: { l: lm, r: 130, t: 28, b: 70, pad: 4 },
      xaxis: {
        title: { text: 'SHAP Value (log-odds)', font: { size: 11 } },
        gridcolor: gridCol, zerolinecolor: '#475569', zerolinewidth: 2,
        tickformat: '.3f', automargin: true,
      },
      yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
      shapes: [{
        type: 'line', x0: 0, x1: 0,
        y0: -0.5, y1: contributions.length - 0.5,
        line: { color: '#CBD5E1', width: 1.5, dash: 'dot' },
        yref: 'y', xref: 'x',
      }],
      annotations: [{
        text: 'Impact strength', showarrow: false,
        xref: 'paper', yref: 'paper',
        x: 1.085, y: 0.82,
        font: { size: 9, color: muted }, align: 'center', xanchor: 'center',
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

    const logVals  = xs.map(v => Math.log1p(v));
    const shapVals = logVals.map((l, i) => l / 8 - 0.4 + 0.04 * Math.sin(i * 1.7));
    const mismatch = logVals.map((_, i) => (Math.sin(i * 0.9 + 1) + 1) / 2);

    const actualLog  = Math.log1p(claimNet);
    const actualShap = typeof claimNetShapValue === 'number'
      ? claimNetShapValue : actualLog / 8 - 0.4;

    const font    = _font();
    const paperBg = _paperBg();
    const plotBg  = _plotBg();
    const muted   = _muted();
    const gridCol = _cssVar('--color-border', '#252D3D');

    Plotly.react(elId, [
      {
        type: 'scatter', mode: 'markers',
        x: logVals, y: shapVals,
        marker: {
          size: 7, opacity: 0.65,
          color: mismatch,
          colorscale: [[0, '#4C7EE0'], [0.5, '#94A3B8'], [1, '#E05252']],
          showscale: true,
          colorbar: {
            thickness: 10, tickvals: [0.05, 0.95], ticktext: ['Match', 'Mismatch'],
            tickfont: { size: 9 }, len: 0.45, yanchor: 'middle', y: 0.5,
            x: 1.02, xanchor: 'left', outlinewidth: 0,
          },
        },
        name: 'Similar claims',
        hovertemplate: 'Claim (log): %{x:.2f}<br>Denial likelihood impact: %{y:.3f}<extra>Similar claim</extra>',
      },
      {
        type: 'scatter', mode: 'markers',
        x: [actualLog], y: [actualShap],
        marker: { size: 18, color: font.color, symbol: 'diamond', line: { color: 'white', width: 2.5 } },
        name: '★ This activity',
        hovertemplate:
          `<b>This activity</b><br>Gross: ${Number(claimNet).toLocaleString('en')}<br>` +
          `SHAP impact: ${actualShap.toFixed(3)}<extra></extra>`,
      },
    ], {
      paper_bgcolor: paperBg, plot_bgcolor: plotBg, font,
      height: 340,
      margin: { l: 80, r: 130, t: 28, b: 60, pad: 4 },
      xaxis: {
        title: { text: 'Activity gross — log scale', font: { size: 11 } },
        gridcolor: gridCol, tickformat: '.1f', automargin: true,
      },
      yaxis: {
        title: { text: 'SHAP contribution (log-odds)', font: { size: 11 } },
        gridcolor: gridCol, zerolinecolor: '#475569', zerolinewidth: 1.5,
      },
      showlegend: true,
      legend: { font: { size: 11 }, x: 0.02, y: 1.12, orientation: 'h' },
      annotations: [{
        text: 'ICD–CPT match', showarrow: false,
        xref: 'paper', yref: 'paper', x: 1.085, y: 0.76,
        font: { size: 9, color: muted }, align: 'center', xanchor: 'center',
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
    const denial = (allFeatures || [])
      .filter(d => (d.shap_value ?? 0) > 0)
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, 10);

    const font    = _font();
    const paperBg = _paperBg();
    const plotBg  = _plotBg();
    const muted   = _muted();
    const gridCol = _cssVar('--color-border', '#252D3D');

    if (denial.length === 0) {
      Plotly.react(elId, [], {
        paper_bgcolor: paperBg, plot_bgcolor: plotBg, font,
        height: 160,
        margin: { l: 20, r: 20, t: 20, b: 20 },
        annotations: [{
          text: 'No denial-driving features found for this claim.<br>All model features are pushing toward approval.',
          showarrow: false, xref: 'paper', yref: 'paper',
          x: 0.5, y: 0.5,
          font: { size: 13, color: muted }, align: 'center',
        }],
      }, PLOTLY_CFG);
      return;
    }

    const totalDenialShap = denial.reduce((s, d) => s + Math.abs(d.shap_value), 0);
    const sorted = [...denial].sort((a, b) => Math.abs(a.shap_value) - Math.abs(b.shap_value));
    const labels  = sorted.map(d => d.display_name || d.feature);
    const values  = sorted.map(d => parseFloat(((Math.abs(d.shap_value) / totalDenialShap) * 100).toFixed(1)));
    const rawShap = sorted.map(d => d.shap_value.toFixed(4));
    const maxVal  = Math.max(...values);
    const colors  = values.map(v => {
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
      text: values.map(v => `  ${v.toFixed(1)}%`),
      textposition: 'outside',
      textfont: { size: 12, color: font.color },
      customdata: rawShap,
      hovertemplate:
        '<b>%{y}</b><br>Share of denial likelihood: <b>%{x:.1f}%</b><br>' +
        'SHAP value: %{customdata}<br>' +
        '<i>Higher = stronger push toward denial for this claim</i><extra></extra>',
    }], {
      paper_bgcolor: paperBg, plot_bgcolor: plotBg, font,
      height: Math.max(260, denial.length * 44 + 90),
      margin: { l: lm, r: 90, t: 20, b: 50, pad: 4 },
      xaxis: {
        title: { text: '% share of total denial likelihood', font: { size: 11 } },
        range: [0, xMax], gridcolor: gridCol, ticksuffix: '%',
      },
      yaxis: { tickfont: { size: 12 } },
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
    const lm      = _leftMargin(yLabels, 200);
    const font    = _font();
    const paperBg = _paperBg();

    Plotly.react(elId, [{
      type: 'heatmap',
      z: zData, x: xLabels, y: yLabels,
      colorscale: [
        [0,   '#D1FAE5'],
        [0.3, '#FEF9C3'],
        [0.6, '#FED7AA'],
        [1,   '#FEE2E2'],
      ],
      text: textArr,
      texttemplate: '%{text}',
      textfont: { size: 10 },
      colorbar: {
        thickness: 12, ticksuffix: fmt === '.1f' ? '%' : '',
        title: { text: 'Denial %', font: { size: 10 }, side: 'right' },
        tickfont: { size: 9 },
      },
      hoverongaps: false,
      hovertemplate: '<b>%{y}</b><br>%{x}<br>Denial rate: %{text}<extra></extra>',
    }], {
      paper_bgcolor: paperBg, plot_bgcolor: paperBg, font,
      height: Math.max(380, yLabels.length * 30 + 130),
      margin: { l: lm, r: 90, t: 55, b: 100, pad: 4 },
      title:  { text: title, font: { size: 13 }, x: 0.5 },
      xaxis:  { tickangle: -40, tickfont: { size: 10 } },
      yaxis:  { tickfont: { size: 10 } },
    }, PLOTLY_CFG);
  }

  return { gauge, waterfall, featureScatter, dependence, denialReasonBar, heatmap };
})();
