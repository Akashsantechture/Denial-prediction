/**
 * js/api.js
 * ---------
 * All HTTP communication with the genericmain.py FastAPI backend.
 *
 * Exports
 * -------
 *   API.predict(payload)   → Promise<PredictionSuccess | ValidationFailure | ApiError>
 *   API.health()           → Promise<HealthResult>
 *
 * Result shapes
 * -------------
 *   PredictionSuccess  { ok: true,  denial_probability_pct, is_high_risk, risk_level, action_recommendation }
 *   ValidationFailure  { ok: false, kind: 'validation', message, errors: [{field, issue, provided_value}] }
 *   ApiError           { ok: false, kind: 'connection'|'timeout'|'http', detail }
 */

const API = (() => {
  const BASE_URL        = 'http://127.0.0.1:8000';
  //  const BASE_URL        = 'http://127.0.0.1:8059';
  const PREDICT_URL     = `${BASE_URL}/predict_user_claim`;
  const HEALTH_URL      = `${BASE_URL}/health`;
  const TIMEOUT_MS      = 10_000;

  /** Wrap fetch with an AbortController timeout. */
  async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * POST the claim payload to /predict_user_claim.
   * @param {Object} payload  — must match UserClaimInput schema
   * @returns {Promise<Object>}
   */
  async function predict(payload) {
    let response;
    try {
      response = await fetchWithTimeout(PREDICT_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        return {
          ok: false, kind: 'timeout',
          detail: `Request timed out after ${TIMEOUT_MS / 1000}s. The server may be under load — please retry.`,
        };
      }
      return {
        ok: false, kind: 'connection',
        detail: `Cannot reach the backend at ${PREDICT_URL}. Ensure genericmain.py is running (uvicorn genericmain:app).`,
      };
    }

    if (response.status === 200) {
      const data = await response.json();
      return {
        ok: true,
        denial_probability_pct: parseFloat(data.denial_probability_pct),
        is_high_risk:           Boolean(data.is_high_risk),
        risk_level:             String(data.risk_level),
        action_recommendation:  String(data.action_recommendation),
      };
    }

    if (response.status === 422) {
      const body = await response.json();
      return {
        ok: false, kind: 'validation',
        message: body.message || 'One or more fields failed validation.',
        errors:  body.errors  || [],
      };
    }

    return {
      ok: false, kind: 'http',
      detail: `Unexpected HTTP ${response.status} from the API. Check the backend logs.`,
    };
  }

  /**
   * GET /health — returns parsed JSON or a fallback object.
   * @returns {Promise<Object>}
   */
  async function health() {
    try {
      const response = await fetchWithTimeout(HEALTH_URL);
      if (response.status === 200) return response.json();
      return { status: 'unhealthy', model_loaded: false };
    } catch (_) {
      return { status: 'unreachable', model_loaded: false };
    }
  }

  return { predict, health, PREDICT_URL };
})();
