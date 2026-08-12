/**
 * js/shap.js
 * ----------
 * SHAP estimation logic — JavaScript port of components/shap_charts.py
 *
 * Because the v1.5 model artifact is not available in the browser, SHAP values
 * are *estimated* from the v1.5 gain percentages (SHAP Interaction Report)
 * scaled by heuristic signals derived from the claim payload.
 *
 * The estimates are deterministic for a given payload and are scaled so the
 * waterfall always closes exactly on the true prediction log-odds.
 *
 * Exports (via global SHAP object)
 * ---------------------------------
 *   SHAP.estimate(payload, result)
 *     → Array of { name, value } sorted by |value| descending
 *
 *   SHAP.BASE_LOG_ODDS   (0.42)
 *   SHAP.finalLogOdds(prob_pct)
 */

const SHAP = (() => {
  const BASE_LOG_ODDS = 0.42;

  function finalLogOdds(prob_pct) {
    const p = Math.min(Math.max(prob_pct / 100, 0.0001), 0.9999);
    return Math.log(p / (1 - p));
  }

  /**
   * Derive approximate per-feature SHAP contributions for a claim.
   * @param {Object} payload  — raw claim payload
   * @param {Object} result   — PredictionSuccess from api.js
   * @returns {Array<{name:string, value:number}>}
   */
  function estimate(payload, result) {
    const totalMass = finalLogOdds(result.denial_probability_pct) - BASE_LOG_ODDS;

    // --- Helper signals derived from payload ---
    const claimNet       = Math.max(payload.claim_net ?? payload.claim_gross ?? 1, 0);
    const activityGross  = Math.max(payload.activity_gross ?? 1, 0);
    const los            = payload.length_of_stay ?? 0;
    const billingLag     = payload.billing_lag_days ?? 1;
    const enc            = payload.encounter_type ?? 'OP';
    const diag           = (payload.diagnosis_code ?? '').toUpperCase();
    const act            = String(payload.activity_code ?? '');
    const nat            = (payload.nationality ?? 'OTHERS').toUpperCase();

    const actGrossLog = Math.log1p(activityGross);
    const netLog      = Math.log1p(claimNet);

    // ICD–CPT mismatch heuristic
    const cptInt = parseInt(act, 10);
    let mismatch = 1.0;
    if (diag[0] === 'M' && cptInt >= 27000 && cptInt <= 27999) mismatch = 0.1;
    else if ('IJEK'.includes(diag[0]) && cptInt > 10000)       mismatch = 0.3;

    const natSignal = ['INDIAN','OTHERS'].includes(nat) ? 0.6 : 0.2;
    const lagSignal = billingLag <= 3 ? 0.8 : 0.2;
    const losSignal = (los >= 4 && enc !== 'IP') ? 0.7 : (los > 7 ? 0.4 : 0.1);
    const ageSignal = (payload.patient_age > 60 || payload.patient_age < 18) ? 0.6 : 0.2;

    // [label, gain%, direction_signal]
    const rawFeatures = [
      ['Activity Gross (log)',   17.53, actGrossLog / 10],
      ['CPT Category',           16.98, mismatch],
      ['Activity Code',          14.19, mismatch * 0.9],
      ['Claim Net (log)',         11.44, netLog / 10],
      ['Claim Gross (log)',        8.98, 1 - (netLog / 12)],
      ['Clinician Category',       6.52, mismatch * 0.5],
      ['Billing Lag',              5.56, lagSignal],
      ['ICD Category',             1.83, mismatch * 0.6],
      ['Nationality',              1.33, natSignal],
      ['Length of Stay',           1.13, losSignal],
      ['Patient Age',              0.73, ageSignal],
    ];

    const totalGain = rawFeatures.reduce((s, [, g]) => s + g, 0);

    let contributions = rawFeatures.map(([label, gain, signal]) => {
      const weight    = (gain / totalGain) * Math.abs(totalMass);
      const direction = signal > 0.5 ? 1 : -1;
      return { name: label, value: weight * direction * signal * 2 };
    });

    // Rescale so sum === totalMass (waterfall closes exactly)
    const currentSum = contributions.reduce((s, c) => s + c.value, 0);
    if (Math.abs(currentSum) > 1e-9) {
      const scale = totalMass / currentSum;
      contributions = contributions.map(c => ({ name: c.name, value: parseFloat((c.value * scale).toFixed(4)) }));
    }

    // Sort by |value| descending
    contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return contributions;
  }

  return { estimate, finalLogOdds, BASE_LOG_ODDS };
})();
