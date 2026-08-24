/**
 * js/shap.js
 * ----------
 * SHAP bridge — converts predictionapi.py top_drivers[] into the
 * contribution array that charts.js, app.js, and chat.js expect.
 *
 * predictionapi.py / shap_engine.py output shape (per activity):
 *   top_drivers: [{
 *     feature:    string,   raw feature name e.g. "claim_gross"
 *     shap_value: float,    signed SHAP value
 *     impact:     "increase_risk" | "decrease_risk"
 *   }]
 *
 * For the Overview + SHAP tab we aggregate across all activities
 * (mean absolute SHAP per feature) so the charts show claim-level
 * rather than single-activity signals.
 *
 * Exports (global SHAP object)
 * ----------------------------
 *   SHAP.fromApiResult(result)
 *     → Array<{ name, value, displayName, direction, impactLevel,
 *               shareP, rawValue, explanation }>
 *       sorted by |value| descending — aggregated across activities
 *
 *   SHAP.fromActivity(prediction)
 *     → same shape, single-activity view
 *
 *   SHAP.baseLogOdds(result)   — derived from claim_denial_probability
 *   SHAP.finalLogOdds(result)  — derived from claim_denial_probability
 *   SHAP.BASE_LOG_ODDS         — fallback constant
 */

const SHAP = (() => {

  const BASE_LOG_ODDS = -0.32;   // logit(0.42) ≈ -0.32 — population base

  // Human-readable display names for raw feature names from the model
  const DISPLAY_NAMES = {
    activity_code:        'Activity / CPT Code',
    activity_quantity:    'Activity Quantity',
    activity_gross:       'Activity Gross Amount',
    patient_age:          'Patient Age',
    gender:               'Gender',
    nationality:          'Nationality',
    claim_gross:          'Claim Gross Amount',
    claim_net:            'Net Claim Amount',
    encounter_type:       'Encounter Type',
    length_of_stay:       'Length of Stay',
    clinician_profession: 'Clinician Profession',
    clinician_category:   'Clinician Category',
    facility_type:        'Facility Type',
    payer_classification: 'Payer Classification',
    diagnosis_code:       'Diagnosis Code',
    billing_lag_days:     'Billing Lag',
    icd_category:         'ICD Category',
    cpt_category:         'CPT Category',
    icd_cpt_domain_match: 'ICD–CPT Domain Match',
  };

  function _displayName(feature) {
    return DISPLAY_NAMES[feature] || feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function _direction(impact) {
    return impact === 'increase_risk' ? 'increases_denial_likelihood' : 'reduces_denial_likelihood';
  }

  function _explanation(feature, shap_value) {
    const dir = shap_value > 0 ? 'increasing denial likelihood' : 'reducing denial likelihood';
    return `${_displayName(feature)} contributed to ${dir} for this claim.`;
  }

  /**
   * Convert a single activity's top_drivers to contribution shape.
   */
  function fromActivity(prediction) {
    if (!prediction || !Array.isArray(prediction.top_drivers)) return [];

    const drivers = prediction.top_drivers;
    const totalAbs = drivers.reduce((s, d) => s + Math.abs(d.shap_value), 0) || 1;

    return drivers
      .map(d => ({
        name:        _displayName(d.feature),
        value:       d.shap_value,
        displayName: _displayName(d.feature),
        direction:   _direction(d.impact),
        impactLevel: Math.abs(d.shap_value) >= 0.1 ? 'HIGH' : Math.abs(d.shap_value) >= 0.04 ? 'MODERATE' : 'LOW',
        shareP:      parseFloat(((Math.abs(d.shap_value) / totalAbs) * 100).toFixed(1)),
        rawValue:    null,
        explanation: _explanation(d.feature, d.shap_value),
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  /**
   * Aggregate top_drivers across ALL activities.
   * Uses mean signed SHAP per feature so same-direction signals
   * reinforce each other and opposing ones partially cancel.
   * This gives a true claim-level SHAP picture.
   */
  function fromApiResult(result) {
    const predictions = result?.predictions;
    if (!Array.isArray(predictions) || predictions.length === 0) return [];

    // Accumulate: { feature → { sum, count, impact } }
    const acc = {};
    for (const pred of predictions) {
      for (const d of (pred.top_drivers || [])) {
        if (!acc[d.feature]) acc[d.feature] = { sum: 0, count: 0, impact: d.impact };
        acc[d.feature].sum   += d.shap_value;
        acc[d.feature].count += 1;
        // If any activity increases risk, flag the feature as risk-increasing
        if (d.impact === 'increase_risk') acc[d.feature].impact = 'increase_risk';
      }
    }

    const entries = Object.entries(acc).map(([feature, { sum, count, impact }]) => ({
      feature,
      shap_value: parseFloat((sum / count).toFixed(4)),
      impact,
    }));

    const totalAbs = entries.reduce((s, e) => s + Math.abs(e.shap_value), 0) || 1;

    return entries
      .map(d => ({
        name:        _displayName(d.feature),
        value:       d.shap_value,
        displayName: _displayName(d.feature),
        direction:   _direction(d.impact),
        impactLevel: Math.abs(d.shap_value) >= 0.1 ? 'HIGH' : Math.abs(d.shap_value) >= 0.04 ? 'MODERATE' : 'LOW',
        shareP:      parseFloat(((Math.abs(d.shap_value) / totalAbs) * 100).toFixed(1)),
        rawValue:    null,
        explanation: _explanation(d.feature, d.shap_value),
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  /**
   * Convert claim denial probability to log-odds (used for SHAP waterfall).
   */
  function _probToLogOdds(prob_pct) {
    const p = Math.min(Math.max((prob_pct ?? 50) / 100, 0.0001), 0.9999);
    return parseFloat(Math.log(p / (1 - p)).toFixed(4));
  }

  /** Final log-odds derived from claim_denial_probability_pct. */
  function finalLogOdds(result) {
    return _probToLogOdds(result?.claim_summary?.claim_denial_probability_pct);
  }

  /** Base log-odds — fixed population constant. */
  function baseLogOdds(_result) {
    return BASE_LOG_ODDS;
  }

  return { fromApiResult, fromActivity, finalLogOdds, baseLogOdds, BASE_LOG_ODDS };
})();
