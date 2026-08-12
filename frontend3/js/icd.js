/**
 * js/icd.js
 * ---------
 * ICD Drill-Down tab — reference data, table builder, heatmap data.
 *
 * Exports (global ICD object)
 * ---------------------------
 *   ICD.groupFromCode(diagnosisCode)  → string  (e.g. "Musculoskeletal")
 *   ICD.renderTable(activeGroup)      → void    (populates #icd-table-body)
 *   ICD.renderHeatmap()               → void    (renders denial-rate heatmap)
 */

const ICD = (() => {

  // ------------------------------------------------------------------ //
  // ICD-10 chapter → group prefix map
  // ------------------------------------------------------------------ //
  const PREFIX_MAP = {
    A: 'Infectious',      B: 'Infectious',
    C: 'Oncology_Hematology', D: 'Oncology_Hematology',
    E: 'Endocrinology',
    F: 'Psychiatry',
    G: 'Neurology',
    H: 'Eye_Ear',
    I: 'Cardiology',
    J: 'Pulmonology',
    K: 'Gastroenterology_Dental',
    L: 'Dermatology',
    M: 'Musculoskeletal',
    N: 'Genitourinary',
    O: 'OBGYN',
    P: 'Congenital',      Q: 'Congenital',
    R: 'General_Symptoms',
    S: 'External_Causes',
    T: 'Trauma_Burns_Poisoning',
    V: 'External_Causes', W: 'External_Causes',
    X: 'External_Causes', Y: 'External_Causes',
    Z: 'Preventive',
  };

  function groupFromCode(code) {
    if (!code) return 'Unknown';
    return PREFIX_MAP[(code.trim().toUpperCase()[0])] ?? 'Unknown';
  }

  // ------------------------------------------------------------------ //
  // Denial Rate by ICD Group × Encounter Type
  // ------------------------------------------------------------------ //
  const DENIAL_RATES = {
    Dermatology:             { OP: 62.1, IP: 48.3, EM: 55.0 },
    Neurology:               { OP: 54.2, IP: 41.7, EM: 49.8 },
    Infectious:              { OP: 51.6, IP: 38.9, EM: 47.2 },
    External_Causes:         { OP: 70.4, IP: 58.1, EM: 64.5 },
    Psychiatry:              { OP: 45.3, IP: 35.8, EM: 41.0 },
    Pulmonology:             { OP: 38.7, IP: 29.4, EM: 35.1 },
    General_Symptoms:        { OP: 36.2, IP: 27.8, EM: 32.6 },
    Musculoskeletal:         { OP: 33.4, IP: 25.1, EM: 30.0 },
    Gastroenterology_Dental: { OP: 31.8, IP: 23.9, EM: 28.3 },
    Genitourinary:           { OP: 28.5, IP: 20.7, EM: 25.4 },
    Endocrinology:           { OP: 24.3, IP: 18.2, EM: 21.6 },
    Cardiology:              { OP: 22.1, IP: 16.4, EM: 19.8 },
    Oncology_Hematology:     { OP: 34.6, IP: 26.3, EM: 31.2 },
    OBGYN:                   { OP: 30.1, IP: 22.5, EM: 27.8 },
    Preventive:              { OP: 18.7, IP: 12.1, EM: 15.9 },
  };

  // ------------------------------------------------------------------ //
  // Granular ICD SHAP rows
  // ------------------------------------------------------------------ //
  const SHAP_ROWS = [
    { code: 'M54.5',   desc: 'Back Pain',              group: 'Musculoskeletal',         claims: 312, denyRate: 33.4, meanShap: -0.015, modelProb: 33.1 },
    { code: 'I10',     desc: 'Essential Hypertension',  group: 'Cardiology',              claims: 891, denyRate: 22.1, meanShap: -0.031, modelProb: 21.8 },
    { code: 'E11.22',  desc: 'Type 2 Diabetes',         group: 'Endocrinology',           claims: 524, denyRate: 24.3, meanShap: -0.024, modelProb: 24.0 },
    { code: 'J06.9',   desc: 'URTI',                    group: 'Pulmonology',             claims: 278, denyRate: 38.7, meanShap:  0.009, modelProb: 38.2 },
    { code: 'K21.0',   desc: 'GERD',                    group: 'Gastroenterology_Dental', claims: 346, denyRate: 31.8, meanShap:  0.001, modelProb: 31.5 },
    { code: 'L30.9',   desc: 'Dermatitis',               group: 'Dermatology',             claims:  89, denyRate: 62.1, meanShap:  0.106, modelProb: 61.5 },
    { code: 'G43.909', desc: 'Migraine',                 group: 'Neurology',               claims: 156, denyRate: 54.2, meanShap:  0.085, modelProb: 53.8 },
    { code: 'A09',     desc: 'Gastroenteritis',          group: 'Infectious',              claims: 204, denyRate: 51.6, meanShap:  0.089, modelProb: 51.1 },
    { code: 'F41.9',   desc: 'Anxiety Disorder',         group: 'Psychiatry',              claims: 118, denyRate: 45.3, meanShap:  0.035, modelProb: 44.9 },
    { code: 'N39.0',   desc: 'UTI',                      group: 'Genitourinary',           claims: 267, denyRate: 28.5, meanShap: -0.009, modelProb: 28.2 },
    { code: 'Z00.00',  desc: 'General Exam',             group: 'Preventive',              claims: 412, denyRate: 18.7, meanShap: -0.041, modelProb: 18.4 },
    { code: 'S06.300', desc: 'Head Injury',              group: 'External_Causes',         claims:  43, denyRate: 70.4, meanShap:  0.126, modelProb: 69.8 },
    { code: 'C50.912', desc: 'Breast Cancer',            group: 'Oncology_Hematology',     claims:  97, denyRate: 34.6, meanShap: -0.016, modelProb: 34.2 },
    { code: 'O26.899', desc: 'Complication Pregnancy',   group: 'OBGYN',                   claims:  72, denyRate: 30.1, meanShap:  0.052, modelProb: 29.8 },
  ];

  // ------------------------------------------------------------------ //
  // Render granular SHAP table
  // ------------------------------------------------------------------ //
  function renderTable(activeGroup, sortKey = 'meanShap', filterGroup = 'All') {
    const tbody = document.getElementById('icd-table-body');
    if (!tbody) return;

    let rows = [...SHAP_ROWS];

    if (filterGroup && filterGroup !== 'All') {
      rows = rows.filter(r => r.group === filterGroup);
    }

    const sortDir = { meanShap: false, denyRate: false, claims: false };
    rows.sort((a, b) => sortDir[sortKey] ? a[sortKey] - b[sortKey] : b[sortKey] - a[sortKey]);

    tbody.innerHTML = rows.map((r, i) => {
      const shapClass  = r.meanShap >= 0 ? 'pos' : 'neg';
      const shapStr    = (r.meanShap >= 0 ? '+' : '') + r.meanShap.toFixed(3);
      const activeClass = (r.group === activeGroup) ? 'active-row' : '';
      return `
        <tr class="${activeClass}">
          <td>${i + 1}</td>
          <td><code>${r.code}</code></td>
          <td>${r.desc}</td>
          <td>${r.group}</td>
          <td>${r.claims}</td>
          <td>${r.denyRate.toFixed(1)}%</td>
          <td><span class="shap-pill ${shapClass}">${shapStr}</span></td>
          <td>${r.modelProb.toFixed(1)}%</td>
        </tr>`;
    }).join('');
  }

  // ------------------------------------------------------------------ //
  // Render denial-rate heatmap
  // ------------------------------------------------------------------ //
  function renderHeatmap(filterGroup = 'All') {
    const groups = filterGroup !== 'All'
      ? [filterGroup]
      : Object.keys(DENIAL_RATES);
    const encounters = ['OP', 'IP', 'EM'];
    const z = groups.map(g => encounters.map(e => DENIAL_RATES[g]?.[e] ?? 0));

    Charts.heatmap(
      'icd-heatmap',
      z,
      encounters,
      groups,
      'Denial Rate by ICD Group × Encounter Type',
      '.1f',
    );
  }

  // ------------------------------------------------------------------ //
  // Populate group filter dropdown
  // ------------------------------------------------------------------ //
  function populateGroupFilter(selectId) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const groups = ['All', ...Object.keys(DENIAL_RATES).sort()];
    sel.innerHTML = groups.map(g => `<option value="${g}">${g.replace(/_/g,' ')}</option>`).join('');
  }

  return { groupFromCode, renderTable, renderHeatmap, populateGroupFilter, SHAP_ROWS, DENIAL_RATES };
})();
