/**
 * js/shap.js
 * ----------
 * SHAP bridge — converts main.py multimodal response into the
 * contribution arrays that charts.js, app.js, and chat.js expect.
 *
 * main.py response shape (per activity):
 *   {
 *     activity_code:           string,
 *     denial_probability:      float,     0–1  (from denial prediction model)
 *     predicted_denial:        bool,
 *     predicted_denial_reason: string|null,    (from reason model, only when denied)
 *     reason_confidence:       float|null,     (max class probability from reason model)
 *     reason_drivers: [{                       (SHAP from reason model for predicted class)
 *       feature:    string,
 *       shap_value: float,
 *       impact:     "increase_reason_probability"|"decrease_reason_probability"
 *     }]
 *   }
 *
 * NOTE: top_drivers (from denial model SHAP) is NOT present in main.py.
 * The denial model SHAP engine is commented out. All SHAP charts now
 * use reason_drivers from the multiclass reason model.
 *
 * Feature set mirrors FEATURE_ORDER in main.py (38 features).
 *
 * Exports (global SHAP object)
 * ----------------------------
 *   SHAP.fromActivityIndex(result, idx)
 *     → reason_drivers for predictions[idx] as contribution objects
 *
 *   SHAP.fromReasonDrivers(prediction)
 *     → reason_drivers for a single prediction object
 *
 *   SHAP.fromApiResult(result)
 *     → aggregated claim-level view (mean across denied activities)
 *       used by chat.js claim-level summaries
 *
 *   SHAP.fromActivity(prediction)
 *     → alias for fromReasonDrivers (backward compat for chat.js)
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
    activity_code:          'Activity Code',
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

  // ── Reason model impact → direction ─────────────────────────────────
  // reason model uses "increase_reason_probability" / "decrease_reason_probability"
  // denial model used "increase_risk" / "decrease_risk"
  // Both map to the same display direction — positive SHAP = pushes toward this outcome
  function _direction(impact) {
    if (impact === 'increase_reason_probability' || impact === 'increase_risk') {
      return 'increases_denial_likelihood';
    }
    return 'reduces_denial_likelihood';
  }

  function _explanation(feature, shap_value, activityCode) {
    const dir = shap_value > 0 ? 'increasing' : 'reducing';
    const act = activityCode ? ` for activity \`${activityCode}\`` : '';
    return `${_displayName(feature)} is ${dir} the predicted denial reason probability${act}.`;
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
      // preserve original impact string so downstream code can check it
      impact:      d.impact,
    };
  }

  // ── Reason drivers for a specific activity by index ──────────────────
  // Primary method used by SHAP tab selector.
  // Uses reason_drivers (multiclass reason model SHAP).
  function fromActivityIndex(result, idx) {
    const pred = result?.predictions?.[idx];
    if (!pred) return [];
    const drivers = pred.reason_drivers ?? pred.top_drivers ?? [];
    if (drivers.length === 0) return [];
    const totalAbs = drivers.reduce((s, d) => s + Math.abs(d.shap_value), 0) || 1;
    return drivers
      .map(d => _toContribution(d, totalAbs, pred.activity_code))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Reason drivers for a single prediction object ────────────────────
  function fromReasonDrivers(prediction) {
    if (!prediction) return [];
    const drivers = prediction.reason_drivers ?? prediction.top_drivers ?? [];
    if (drivers.length === 0) return [];
    const totalAbs = drivers.reduce((s, d) => s + Math.abs(d.shap_value), 0) || 1;
    return drivers
      .map(d => _toContribution(d, totalAbs, prediction.activity_code))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Backward-compat alias used by chat.js ────────────────────────────
  const fromActivity = fromReasonDrivers;

  // ── Aggregated claim-level view ──────────────────────────────────────
  // Mean reason_drivers across denied activities only.
  // Used by chat.js for claim-level summaries.
  function fromApiResult(result) {
    const predictions = result?.predictions;
    if (!Array.isArray(predictions) || predictions.length === 0) return [];

    // Use denied activities first; fall back to all if none denied
    const source = predictions.filter(p => p.predicted_denial);
    const pool   = source.length > 0 ? source : predictions;

    const acc = {};
    for (const pred of pool) {
      const drivers = pred.reason_drivers ?? pred.top_drivers ?? [];
      for (const d of drivers) {
        if (!acc[d.feature]) acc[d.feature] = { sum: 0, count: 0, impact: d.impact };
        acc[d.feature].sum   += d.shap_value;
        acc[d.feature].count += 1;
        // keep the dominant impact direction
        if (d.impact === 'increase_reason_probability' || d.impact === 'increase_risk') {
          acc[d.feature].impact = d.impact;
        }
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

  function finalLogOddsForActivity(result, idx) {
    const prob = result?.predictions?.[idx]?.denial_probability;
    if (prob == null) return BASE_LOG_ODDS;
    return _probToLogOdds(prob * 100);
  }

  function finalLogOdds(result) {
    return _probToLogOdds(result?.claim_summary?.claim_denial_probability_pct);
  }

  function baseLogOdds(_result) { return BASE_LOG_ODDS; }

  return {
    fromActivityIndex,
    fromReasonDrivers,
    fromActivity,        // alias → fromReasonDrivers
    fromApiResult,
    finalLogOddsForActivity,
    finalLogOdds,
    baseLogOdds,
    BASE_LOG_ODDS,
    DISPLAY_NAMES,
  };
})();
