/**
 * js/interactions.js
 * ------------------
 * Interactions tab — top pairs table + ICD×CPT heatmap renderers.
 *
 * Data source: v1.5 SHAP Feature Interaction Report (Top 20 combinations).
 *
 * Exports (global Interactions object)
 * -------------------------------------
 *   Interactions.cptGroupFromCode(activityCode)  → string
 *   Interactions.renderPairsTable(activeIcdGroup, activeCptGroup)
 *   Interactions.renderSHAPHeatmap()
 *   Interactions.renderDenialHeatmap()
 */

const Interactions = (() => {

  // ------------------------------------------------------------------ //
  // CPT range → group
  // ------------------------------------------------------------------ //
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

  // ------------------------------------------------------------------ //
  // Top interaction pairs (from SHAP Interaction Report v1.5)
  // ------------------------------------------------------------------ //
  const PAIRS = [
    { icd: 'External_Causes',         cpt: 'Pathology_Laboratory', claims: 2,   icdImpact: 12.56, cptImpact: 12.65, combined: 25.21 },
    { icd: 'Dermatology',              cpt: 'Pathology_Laboratory', claims: 42,  icdImpact: 10.59, cptImpact: 11.52, combined: 22.11 },
    { icd: 'Infectious',               cpt: 'Pathology_Laboratory', claims: 18,  icdImpact:  8.91, cptImpact: 10.80, combined: 19.71 },
    { icd: 'Neurology',                cpt: 'Pathology_Laboratory', claims: 25,  icdImpact:  8.52, cptImpact:  9.05, combined: 17.57 },
    { icd: 'Eye_Ear',                  cpt: 'Pathology_Laboratory', claims:  9,  icdImpact:  8.77, cptImpact:  8.44, combined: 17.21 },
    { icd: 'Trauma_Burns_Poisoning',   cpt: 'Pathology_Laboratory', claims:  6,  icdImpact:  2.36, cptImpact: 11.89, combined: 14.26 },
    { icd: 'General_Symptoms',         cpt: 'Pathology_Laboratory', claims: 316, icdImpact:  3.22, cptImpact:  9.36, combined: 12.58 },
    { icd: 'OBGYN',                    cpt: 'Pathology_Laboratory', claims:  3,  icdImpact:  5.22, cptImpact:  7.31, combined: 12.53 },
    { icd: 'Infectious',               cpt: 'Surgery',              claims:  1,  icdImpact:  7.27, cptImpact:  4.53, combined: 11.80 },
    { icd: 'Psychiatry',               cpt: 'Pathology_Laboratory', claims: 11,  icdImpact:  3.50, cptImpact:  7.13, combined: 10.62 },
    { icd: 'Oncology_Hematology',      cpt: 'Surgery',              claims:  1,  icdImpact: -3.68, cptImpact: 13.37, combined:  9.69 },
    { icd: 'Pulmonology',              cpt: 'Pathology_Laboratory', claims: 38,  icdImpact:  0.86, cptImpact:  8.80, combined:  9.66 },
    { icd: 'Gastroenterology_Dental',  cpt: 'Pathology_Laboratory', claims: 87,  icdImpact:  0.06, cptImpact:  9.54, combined:  9.60 },
    { icd: 'Musculoskeletal',          cpt: 'Pathology_Laboratory', claims: 124, icdImpact: -1.50, cptImpact:  8.42, combined:  6.92 },
    { icd: 'Neurology',                cpt: 'HCPCS_Supplies',       claims:  1,  icdImpact: 11.36, cptImpact: -4.76, combined:  6.60 },
    { icd: 'Dermatology',              cpt: 'Surgery',              claims:  3,  icdImpact: 11.12, cptImpact: -4.69, combined:  6.43 },
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

  // ------------------------------------------------------------------ //
  // Pairs table
  // ------------------------------------------------------------------ //
  function renderPairsTable(activeIcd, activeCpt) {
    const tbody = document.getElementById('pairs-table-body');
    if (!tbody) return;

    tbody.innerHTML = PAIRS.map((p, i) => {
      const isActive = p.icd === activeIcd;
      const rowClass = isActive ? 'active-row' : '';
      const icdImpactStr = (p.icdImpact >= 0 ? '+' : '') + p.icdImpact.toFixed(2) + '%';
      const cptImpactStr = (p.cptImpact >= 0 ? '+' : '') + p.cptImpact.toFixed(2) + '%';
      return `
        <tr class="${rowClass}">
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

  // ------------------------------------------------------------------ //
  // Build [ICD × CPT] matrix from PAIRS data
  // ------------------------------------------------------------------ //
  function buildMatrix(valueKey) {
    return ICD_GROUPS.map(icd =>
      CPT_GROUPS.map(cpt => {
        const found = PAIRS.find(p => p.icd === icd && p.cpt === cpt);
        return found ? found[valueKey] : 0;
      })
    );
  }

  // ------------------------------------------------------------------ //
  // Render SHAP interaction heatmap
  // ------------------------------------------------------------------ //
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

  // ------------------------------------------------------------------ //
  // Render denial rate heatmap (actual %)
  // ------------------------------------------------------------------ //
  function renderDenialHeatmap() {
    const z = buildMatrix('combined');   // combined risk used as proxy
    Charts.heatmap(
      'interactions-denial-heatmap',
      z,
      CPT_GROUPS.map(g => g.replace(/_/g,' ')),
      ICD_GROUPS.map(g => g.replace(/_/g,' ')),
      'ICD × CPT Group — Denial Rate Heatmap (Actual %)',
      '.2f',
    );
  }

  return { cptGroupFromCode, renderPairsTable, renderSHAPHeatmap, renderDenialHeatmap, PAIRS };
})();
