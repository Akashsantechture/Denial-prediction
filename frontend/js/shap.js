/**
 * js/shap.js
 * ----------
 * SHAP bridge — converts predictionapi.py top_drivers[] into the
 * contribution arrays that charts.js, app.js, and chat.js expect.
 *
 * predictionapi.py / shap_engine.py output shape (per activity):
 *   top_drivers: [{
 *     feature:    string,   e.g. "claim_gross"
 *     shap_value: float,    signed SHAP value
 *     impact:     "increase_risk" | "decrease_risk"
 *   }]
 *
 * Feature set mirrors FEATURE_ORDER in predictionapi.py (38 features):
 *   activity_code, activity_quantity, activity_gross,
 *   patient_age, gender, nationality,
 *   claim_gross, claim_net, encounter_type, length_of_stay,
 *   clinician_profession, facility_type, payer_classification,
 *   primary_diagnosis_code, primary_diagnosis_category,
 *   secondary_dx_count,
 *   secondary_infectious, secondary_oncology, secondary_oncology_hematology,
 *   secondary_endocrinology, secondary_psychiatry, secondary_neurology,
 *   secondary_eye_ear, secondary_cardiology, secondary_pulmonology,
 *   secondary_gastroenterology_dental, secondary_dermatology,
 *   secondary_musculoskeletal, secondary_genitourinary, secondary_obgyn,
 *   secondary_pediatrics, secondary_congenital, secondary_general_symptoms,
 *   secondary_trauma_burns_poisoning, secondary_external_causes,
 *   secondary_factors_influencing_health_status,
 *   secondary_unknown_icd_category,
 *   cpt_category
 *
 * Exports (global SHAP object)
 * ----------------------------
 *   SHAP.fromActivityIndex(result, idx)
 *     → true per-activity contributions from predictions[idx].top_drivers
 *
 *   SHAP.fromApiResult(result)
 *     → aggregated claim-level view (mean across all activities)
 *       used only for chat.js claim-level summaries
 *
 *   SHAP.fromActivity(prediction)
 *     → contributions for a single prediction object
 *
 *   SHAP.baseLogOdds(result)     — fixed population constant −0.32
 *   SHAP.finalLogOddsForActivity(result, idx)
 *     → log-odds derived from predictions[idx].denial_probability
 *   SHAP.finalLogOdds(result)
 *     → log-odds derived from claim_denial_probability_pct (claim level)
 *   SHAP.BASE_LOG_ODDS
 *   SHAP.DISPLAY_NAMES           — feature key → human-readable label
 */

const SHAP = (() => {

  const BASE_LOG_ODDS = -0.32;   // logit(~42% population base)

  // ── Display names — covers all 38 features in FEATURE_ORDER ─────────
  const DISPLAY_NAMES = {
    // Activity features
    activity_code:          'Activity code',
    activity_quantity:      'Activity Quantity',
    activity_gross:         'Activity Gross Amount',

    // Patient features
    patient_age:            'Patient Age',
    gender:                 'Gender',
    nationality:            'Nationality',

    // Claim features
    claim_gross:            'Claim Gross Amount',
    claim_net:              'Net Claim Amount',
    encounter_type:         'Encounter Type',
    length_of_stay:         'Length of Stay',

    // Provider features
    clinician_profession:   'Clinician Profession',
    facility_type:          'Facility Type',
    payer_classification:   'Payer Classification',

    // Primary diagnosis features
    primary_diagnosis_code:     'Primary Diagnosis Code',
    primary_diagnosis_category: 'Primary Diagnosis Category',

    // Secondary diagnosis count
    secondary_dx_count:     'Secondary Diagnosis Count',

    // Secondary diagnosis category flags (binary 0/1)
    secondary_infectious:                        'Secondary: Infectious Disease',
    secondary_oncology:                          'Secondary: Oncology',
    secondary_oncology_hematology:               'Secondary: Oncology / Hematology',
    secondary_endocrinology:                     'Secondary: Endocrinology',
    secondary_psychiatry:                        'Secondary: Psychiatry',
    secondary_neurology:                         'Secondary: Neurology',
    secondary_eye_ear:                           'Secondary: Eye / Ear',
    secondary_cardiology:                        'Secondary: Cardiology',
    secondary_pulmonology:                       'Secondary: Pulmonology',
    secondary_gastroenterology_dental:           'Secondary: Gastroenterology / Dental',
    secondary_dermatology:                       'Secondary: Dermatology',
    secondary_musculoskeletal:                   'Secondary: Musculoskeletal',
    secondary_genitourinary:                     'Secondary: Genitourinary',
    secondary_obgyn:                             'Secondary: OB/GYN',
    secondary_pediatrics:                        'Secondary: Pediatrics',
    secondary_congenital:                        'Secondary: Congenital',
    secondary_general_symptoms:                  'Secondary: General Symptoms',
    secondary_trauma_burns_poisoning:            'Secondary: Trauma / Burns / Poisoning',
    secondary_external_causes:                   'Secondary: External Causes',
    secondary_factors_influencing_health_status: 'Secondary: Health Status Factors',
    secondary_unknown_icd_category:              'Secondary: Unknown ICD Category',

    // CPT category
    cpt_category:           'CPT Category',
  };

  function _displayName(feature) {
    return DISPLAY_NAMES[feature]
      || feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function _direction(impact) {
    return impact === 'increase_risk'
      ? 'increases_denial_likelihood'
      : 'reduces_denial_likelihood';
  }

  function _explanation(feature, shap_value, activityCode) {
    const dir = shap_value > 0 ? 'increasing' : 'reducing';
    const act = activityCode ? ` for activity \`${activityCode}\`` : '';
    return `${_displayName(feature)} is ${dir} denial likelihood${act}.`;
  }

  function _toContribution(d, totalAbs, activityCode) {
    return {
      name:        _displayName(d.feature),
      feature:     d.feature,
      value:       d.shap_value,
      displayName: _displayName(d.feature),
      direction:   _direction(d.impact),
      impactLevel: Math.abs(d.shap_value) >= 0.10 ? 'HIGH'
                 : Math.abs(d.shap_value) >= 0.04 ? 'MODERATE' : 'LOW',
      shareP:      totalAbs > 0
                     ? parseFloat(((Math.abs(d.shap_value) / totalAbs) * 100).toFixed(1))
                     : 0,
      rawValue:    null,
      explanation: _explanation(d.feature, d.shap_value, activityCode),
    };
  }

  // ── True per-activity SHAP by index ─────────────────────────────────
  // Primary method used by the SHAP tab selector.
  // Returns top_drivers for predictions[idx] with no aggregation.
  function fromActivityIndex(result, idx) {
    const pred = result?.predictions?.[idx];
    if (!pred || !Array.isArray(pred.top_drivers) || pred.top_drivers.length === 0) return [];
    const drivers  = pred.top_drivers;
    const totalAbs = drivers.reduce((s, d) => s + Math.abs(d.shap_value), 0) || 1;
    return drivers
      .map(d => _toContribution(d, totalAbs, pred.activity_code))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Single prediction object → contributions ─────────────────────────
  function fromActivity(prediction) {
    if (!prediction || !Array.isArray(prediction.top_drivers)) return [];
    const drivers  = prediction.top_drivers;
    const totalAbs = drivers.reduce((s, d) => s + Math.abs(d.shap_value), 0) || 1;
    return drivers
      .map(d => _toContribution(d, totalAbs, prediction.activity_code))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Aggregated claim-level view (mean across activities) ─────────────
  // Used by chat.js for claim-level summaries only.
  function fromApiResult(result) {
    const predictions = result?.predictions;
    if (!Array.isArray(predictions) || predictions.length === 0) return [];

    const acc = {};
    for (const pred of predictions) {
      for (const d of (pred.top_drivers || [])) {
        if (!acc[d.feature]) acc[d.feature] = { sum: 0, count: 0, impact: d.impact };
        acc[d.feature].sum   += d.shap_value;
        acc[d.feature].count += 1;
        if (d.impact === 'increase_risk') acc[d.feature].impact = 'increase_risk';
      }
    }

    const entries = Object.entries(acc).map(([feature, { sum, count, impact }]) => ({
      feature, shap_value: parseFloat((sum / count).toFixed(4)), impact,
    }));

    const totalAbs = entries.reduce((s, e) => s + Math.abs(e.shap_value), 0) || 1;

    return entries
      .map(d => _toContribution(d, totalAbs, null))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Log-odds helpers ─────────────────────────────────────────────────
  function _probToLogOdds(prob_pct) {
    const p = Math.min(Math.max((prob_pct ?? 50) / 100, 0.0001), 0.9999);
    return parseFloat(Math.log(p / (1 - p)).toFixed(4));
  }

  // Log-odds for a specific activity (uses denial_probability 0–1)
  function finalLogOddsForActivity(result, idx) {
    const prob = result?.predictions?.[idx]?.denial_probability;
    if (prob == null) return BASE_LOG_ODDS;
    return _probToLogOdds(prob * 100);
  }

  // Claim-level log-odds
  function finalLogOdds(result) {
    return _probToLogOdds(result?.claim_summary?.claim_denial_probability_pct);
  }

  function baseLogOdds(_result) { return BASE_LOG_ODDS; }

  return {
    fromActivityIndex,
    fromActivity,
    fromApiResult,
    finalLogOddsForActivity,
    finalLogOdds,
    baseLogOdds,
    BASE_LOG_ODDS,
    DISPLAY_NAMES,
  };
})();
