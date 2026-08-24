/**
 * js/api.js
 * ---------
 * HTTP layer for predictionapi.py (FastAPI + XGBoost, port 8000).
 *
 * Endpoint
 * --------
 *   POST /predict
 *
 * Request shape — ClaimRequest
 * ----------------------------
 *   {
 *     claim_id:             string,
 *     patient_age:          int,
 *     gender:               string,
 *     nationality:          string,
 *     encounter_type:       string,
 *     length_of_stay:       int,
 *     claim_gross:          float,
 *     claim_net:            float,
 *     billing_lag_days:     int,
 *     clinician_profession: string,
 *     clinician_category:   string,
 *     facility_type:        string,
 *     payer_classification: string,
 *     diagnosis_code:       string,
 *     icd_category:         string,   ← derived from diagnosis_code prefix
 *     activities: [{
 *       activity_code:   string,
 *       activity_quantity: float,
 *       activity_gross:  float,
 *       cpt_category:    string,      ← entered by user
 *     }]
 *   }
 *
 * Response shape — PredictSuccess
 * --------------------------------
 *   {
 *     ok: true,
 *     claim_id:      string,
 *     claim_summary: {
 *       claim_denial_probability:     float,   0–1
 *       claim_denial_probability_pct: float,   0–100
 *       claim_risk_level:             "HIGH"|"MEDIUM"|"LOW"
 *       highest_activity_risk:        float,
 *       average_activity_risk:        float,
 *       activity_count:               int,
 *     },
 *     predictions: [{
 *       activity_code:      string,
 *       cpt_category:       string,
 *       denial_probability: float,    0–1
 *       predicted_denial:   bool,
 *       top_drivers: [{
 *         feature:    string,
 *         shap_value: float,
 *         impact:     "increase_risk"|"decrease_risk"
 *       }],
 *       recommendation: string,
 *     }]
 *   }
 *
 * Error shapes
 * ------------
 *   ApiError  { ok: false, kind: 'connection'|'timeout'|'http'|'server', detail }
 */

const API = (() => {
  const BASE_URL    = 'http://127.0.0.1:8000';
  const PREDICT_URL = `${BASE_URL}/predict`;
  const HEALTH_URL  = `${BASE_URL}/health`;
  const TIMEOUT_MS  = 25_000;

  // ── ICD category derivation ──────────────────────────────────────────
  // Maps the first letter of an ICD-10 code to a broad clinical category.
  // This avoids sending a blank icd_category to the API.
  const ICD_CATEGORY_MAP = {
    A: 'Infectious', B: 'Infectious',
    C: 'Neoplasms',  D: 'Neoplasms',
    E: 'Endocrine',
    F: 'Mental',
    G: 'Neurological',
    H: 'Sensory',
    I: 'Circulatory',
    J: 'Respiratory',
    K: 'Digestive',
    L: 'Skin',
    M: 'Musculoskeletal',
    N: 'Genitourinary',
    O: 'Obstetric',
    P: 'Perinatal',
    Q: 'Congenital',
    R: 'Symptoms',
    S: 'Injury', T: 'Injury',
    V: 'External', W: 'External', X: 'External', Y: 'External',
    Z: 'Health Status',
  };

  function deriveIcdCategory(diagnosisCode) {
    if (!diagnosisCode) return 'Other';
    return ICD_CATEGORY_MAP[diagnosisCode[0].toUpperCase()] ?? 'Other';
  }

  // ── Fetch with timeout ───────────────────────────────────────────────
  async function _fetchTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Send a claim to POST /predict.
   *
   * @param {Object} payload  — flat claim object collected from the form
   *                            Must include an `activities` array.
   * @returns PredictSuccess | ApiError
   */
  async function predict(payload) {
    // Build ClaimRequest — map flat form fields to API schema
    const claimRequest = {
      claim_id:             payload.claim_id || `CLM-${Date.now()}`,
      patient_age:          payload.patient_age,
      gender:               payload.gender,
      nationality:          payload.nationality,
      encounter_type:       payload.encounter_type,
      length_of_stay:       payload.length_of_stay,
      claim_gross:          payload.claim_gross,
      claim_net:            payload.claim_net,
      billing_lag_days:     payload.billing_lag_days,
      clinician_profession: payload.clinician_profession,
      clinician_category:   payload.clinician_category,
      facility_type:        payload.facility_type,
      payer_classification: payload.payer_classification || payload.payer_id || 'UNKNOWN',
      diagnosis_code:       payload.diagnosis_code,
      icd_category:         payload.icd_category || deriveIcdCategory(payload.diagnosis_code),
      activities:           payload.activities || [],
    };

    let response;
    try {
      response = await _fetchTimeout(PREDICT_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(claimRequest),
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        return {
          ok: false, kind: 'timeout',
          detail: `Request timed out after ${TIMEOUT_MS / 1000}s. SHAP calculation can be slow on first run — please retry.`,
        };
      }
      return {
        ok: false, kind: 'connection',
        detail: `Cannot reach the backend at ${PREDICT_URL}. Make sure predictionapi.py is running: uvicorn predictionapi:app --reload --port 8000`,
      };
    }

    if (response.status === 200) {
      const data = await response.json();
      return {
        ok:            true,
        claim_id:      data.claim_id,
        claim_summary: data.claim_summary,
        predictions:   data.predictions ?? [],
      };
    }

    if (response.status === 422) {
      const body = await response.json().catch(() => ({}));
      return {
        ok: false, kind: 'validation',
        message: body.message || body.detail || 'One or more fields failed validation.',
        errors:  body.errors  || [],
      };
    }

    const body = await response.json().catch(() => ({}));
    return {
      ok: false, kind: 'http',
      detail: body.detail || `Unexpected HTTP ${response.status} from the API.`,
    };
  }

  async function health() {
    try {
      const response = await _fetchTimeout(HEALTH_URL);
      if (response.status === 200) return response.json();
      return { status: 'unhealthy', model_loaded: false };
    } catch (_) {
      return { status: 'unreachable', model_loaded: false };
    }
  }

  return { predict, health, deriveIcdCategory, PREDICT_URL, BASE_URL };
})();


/**
 * AnalystAPI
 * ----------
 * HTTP layer for the AI Analyst microservice (port 8060).
 */
const AnalystAPI = (() => {
  const BASE_URL    = 'http://127.0.0.1:8060';
  const SESSION_URL = `${BASE_URL}/analyst/session`;
  const CHAT_URL    = `${BASE_URL}/analyst/chat`;
  const TIMEOUT_MS  = 30_000;

  async function _fetchTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async function createSession(claimIntelligence) {
    try {
      const response = await _fetchTimeout(SESSION_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ claim_intelligence: claimIntelligence }),
      });
      if (response.status === 200) {
        const data = await response.json();
        return { ok: true, session_id: data.session_id };
      }
      const body = await response.json().catch(() => ({}));
      return { ok: false, detail: body.detail ?? `HTTP ${response.status}` };
    } catch (err) {
      if (err.name === 'AbortError') return { ok: false, detail: 'Session request timed out.' };
      return { ok: false, detail: `Cannot reach AI Analyst at ${BASE_URL}.` };
    }
  }

  async function chat(sessionId, message) {
    try {
      const response = await _fetchTimeout(CHAT_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: sessionId, message }),
      });
      if (response.status === 200) {
        const data = await response.json();
        return { ok: true, answer: data.answer };
      }
      const body = await response.json().catch(() => ({}));
      return { ok: false, detail: body.detail ?? `HTTP ${response.status}` };
    } catch (err) {
      if (err.name === 'AbortError') return { ok: false, detail: 'Analyst response timed out.' };
      return { ok: false, detail: `Cannot reach AI Analyst at ${BASE_URL}.` };
    }
  }

  async function deleteSession(sessionId) {
    if (!sessionId) return;
    try {
      await fetch(`${BASE_URL}/analyst/session/${sessionId}`, { method: 'DELETE' });
    } catch (_) { /* fire-and-forget */ }
  }

  async function health() {
    try {
      const r = await _fetchTimeout(`${BASE_URL}/health`);
      if (r.status === 200) return r.json();
      return { status: 'unhealthy', llm_loaded: false };
    } catch (_) {
      return { status: 'unreachable', llm_loaded: false };
    }
  }

  return { createSession, chat, deleteSession, health, BASE_URL };
})();
