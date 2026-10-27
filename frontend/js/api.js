/**
 * js/api.js
 * ---------
 * HTTP layer for predictionapi.py (FastAPI + XGBoost, port 8000).
 *
 * Endpoint
 * --------
 *   POST /predict
 *
 * Request shape — ClaimRequest (validators/schemas.py)
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
 *     clinician_profession: string,
 *     facility_type:        string,
 *     payer_classification: string,
 *     primary_diagnosis_code: string,
 *     activities: [{
 *       activity_code:      string,
 *       activity_quantity:  float,
 *       activity_gross:     float,
 *       diagnoses: [{
 *         diagnosis_code:   string,
 *         diagnosis_type:   string   // "primary" | "secondary"
 *       }]
 *     }]
 *   }
 *
 * Response shape — POST /predict
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
 *       activity_code:           string,
 *       denial_probability:      float,    0–1
 *       predicted_denial:        bool,
 *       predicted_denial_reason: string|null,   — reason label (only when predicted_denial=true)
 *       reason_confidence:       float|null,    — max class probability from reason model
 *       reason_drivers: [{                      — SHAP from reason model (multiclass, for predicted class)
 *         feature:    string,
 *         shap_value: float,
 *         impact:     "increase_reason_probability"|"decrease_reason_probability"
 *       }],
 *     }]
 *   }
 *
 * Error shapes
 * ------------
 *   ApiError  { ok: false, kind: 'connection'|'timeout'|'http'|'server', detail }
 */

const API = (() => {
  const BASE_URL        = 'http://127.0.0.1:8000';
  const PREDICT_URL     = `${BASE_URL}/predict`;
  const KSA_PREDICT_URL = `${BASE_URL}/ksa/predict`;
  const HEALTH_URL      = `${BASE_URL}/health`;
  const MODEL_INFO_URL  = `${BASE_URL}/model-info`;
  const TIMEOUT_MS      = 25_000;

  // ── ICD category derivation ──────────────────────────────────────────
  // Maps the first letter of an ICD-10 code to a clinical category
  // that matches preprocessing/categorial_conversion.py ICD_CATEGORY_MAP.
  const ICD_CATEGORY_MAP = {
    A: 'Infectious',                    B: 'Infectious',
    C: 'Oncology',                      D: 'Oncology_Hematology',
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
    P: 'Pediatrics',
    Q: 'Congenital',
    R: 'General_Symptoms',
    S: 'Trauma_Burns_Poisoning',        T: 'Trauma_Burns_Poisoning',
    V: 'External_Causes',               W: 'External_Causes',
    X: 'External_Causes',               Y: 'External_Causes',
    Z: 'Factors_Influencing_Health_Status',
  };

  function deriveIcdCategory(diagnosisCode) {
    if (!diagnosisCode) return 'Unknown_ICD';
    return ICD_CATEGORY_MAP[diagnosisCode[0].toUpperCase()] ?? 'Unknown_ICD';
  }

  // ── Fetch with timeout + region header ──────────────────────────────
  async function _fetchTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    
    // Inject X-Region header from region manager
    const headers = { ...options.headers };
    if (window.regionManager) {
      headers['X-Region'] = window.regionManager.getRegionHeader();
    }
    
    try {
      return await fetch(url, { ...options, headers, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Route to UAE or KSA predict based on the active region toggle.
   */
  async function predict(payload) {
    const region = window.regionManager?.getRegionCode() || 'uae';
    return region === 'ksa' ? _predictKSA(payload) : _predictUAE(payload);
  }

  // ── UAE predict (/predict) ───────────────────────────────────────────
  async function _predictUAE(payload) {

    // Build ClaimRequest — map flat form fields to exact API schema
    const claimRequest = {
      claim_id:               payload.claim_id || `CLM-${Date.now()}`,
      patient_age:            payload.patient_age,
      gender:                 payload.gender,
      encounter_type:         payload.encounter_type,
      length_of_stay:         payload.length_of_stay,
      claim_gross:            payload.claim_gross,
      claim_net:              payload.claim_net,
      clinician_profession:   payload.clinician_profession,
      facility_type:          payload.facility_type,
      payer_classification:   payload.payer_classification || 'Insurance',
      payer_name:             payload.payer_name           || 'Unknown',
      payer_id:               payload.payer_id             || 'Unknown',
      receiver_id:            payload.receiver_id          || 'Unknown',
      network:                payload.network              || 'Unknown Network',
      primary_diagnosis_code: payload.primary_diagnosis_code || '',
      activities:             (payload.activities || []).map(a => ({
        activity_code:      a.activity_code,
        activity_quantity:  a.activity_quantity,
        activity_gross:     a.activity_gross,
        diagnoses: Array.isArray(a.diagnoses) && a.diagnoses.length > 0
          ? a.diagnoses
          : [{ diagnosis_code: payload.primary_diagnosis_code || '', diagnosis_type: 'Principal' }],
      })),
    };

    return _sendPredict(PREDICT_URL, claimRequest);
  }

  // ── KSA predict (/ksa/predict) ───────────────────────────────────────
  // KSA uses services[] not activities[].
  // Maps service_code/service_quantity/service_gross from each service item.
  // Extra claim-level fields: facility, department, claim_type, claim_subtype, policy_name.
  async function _predictKSA(payload) {

    const claimRequest = {
      claim_id:               payload.claim_id || `CLM-${Date.now()}`,
      patient_age:            payload.patient_age,
      gender:                 payload.gender,
      encounter_type:         payload.encounter_type,
      length_of_stay:         payload.length_of_stay,
      claim_gross:            payload.claim_gross,
      claim_net:              payload.claim_net,
      facility:               payload.facility              || 'Unknown',
      facility_type:          payload.facility_type,
      clinician_category:     payload.clinician_category    || 'Unknown',
      department:             payload.department            || 'Unknown',
      claim_type:             payload.claim_type            || 'Unknown',
      claim_subtype:          payload.claim_subtype         || 'Unknown',
      payer_classification:   payload.payer_classification  || 'Unknown',
      payer_id:               payload.payer_id              || 'Unknown',
      receiver_id:            payload.receiver_id           || 'Unknown',
      policy_name:            payload.policy_name           || 'Unknown',
      primary_diagnosis_code: payload.primary_diagnosis_code || '',
      services:               (payload.services || []).map(s => ({
        service_code:     s.service_code,
        service_quantity: s.service_quantity,
        service_gross:    s.service_gross,
        diagnoses: Array.isArray(s.diagnoses) && s.diagnoses.length > 0
          ? s.diagnoses
          : [{ diagnosis_code: payload.primary_diagnosis_code || '', diagnosis_type: 'Principal' }],
      })),
    };

    return _sendPredict(KSA_PREDICT_URL, claimRequest);
  }

  // ── Shared fetch + response handler ─────────────────────────────────
  async function _sendPredict(url, claimRequest) {
    let response;
    try {
      response = await _fetchTimeout(url, {
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
        // API returns "activity_predictions" — normalise to "predictions" for the frontend
        predictions:   data.activity_predictions ?? data.predictions ?? [],
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
      return { status: 'unhealthy', model_loaded: false, encoder_loaded: false, shap_loaded: false };
    } catch (_) {
      return { status: 'unreachable', model_loaded: false, encoder_loaded: false, shap_loaded: false };
    }
  }

  async function modelInfo() {
    try {
      const response = await _fetchTimeout(MODEL_INFO_URL);
      if (response.status === 200) return response.json();
      return null;
    } catch (_) {
      return null;
    }
  }

  return { predict, health, modelInfo, deriveIcdCategory, PREDICT_URL, BASE_URL };
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
        return {
          ok:              true,
          session_id:      data.session_id,
          initial_summary: data.initial_summary ?? null,
          rag_used:        data.rag_used ?? false,
        };
      }
      const body = await response.json().catch(() => ({}));
      return { ok: false, detail: body.detail ?? `HTTP ${response.status}` };
    } catch (err) {
      if (err.name === 'AbortError') return { ok: false, detail: 'Session request timed out.' };
      return { ok: false, detail: `Cannot reach AI Analyst at ${BASE_URL}.` };
    }
  }

  async function chat(sessionId, message, useRag = true) {
    try {
      const response = await _fetchTimeout(CHAT_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: sessionId, message, use_rag: useRag }),
      });
      if (response.status === 200) {
        const data = await response.json();
        return { ok: true, answer: data.answer, rag_used: data.rag_used ?? false };
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
