/**
 * js/api.js
 * ---------
 * HTTP layer for model2main.py FastAPI backend.
 *
 * KEY CHANGE: The API now returns real SHAP values inside result.explanation.
 * The full explanation object is passed through UNTOUCHED so shap.js and
 * charts.js can use actual TreeExplainer values — no frontend estimation.
 *
 * Result shapes
 * -------------
 *   PredictionSuccess {
 *     ok: true,
 *     denial_probability_pct, is_high_risk, risk_level, action_recommendation,
 *     explanation: {
 *       base_value,            — real SHAP base value (log-odds)
 *       final_model_value,     — base + sum(shap_values)
 *       model_output_space,    — "raw_margin_log_odds"
 *       top_drivers:        [{ feature, display_name, shap_value, absolute_shap,
 *                              direction, impact_level, contribution_share_pct,
 *                              raw_value, explanation }],
 *       supporting_factors: [ ...same shape... ],
 *       all_features:       [ ...same shape... ],
 *       interactions:       [{ feature_1, feature_1_display_name,
 *                              feature_2, feature_2_display_name,
 *                              interaction_value, direction, explanation }]
 *     }
 *   }
 *   ValidationFailure { ok: false, kind: 'validation', message, errors }
 *   ApiError          { ok: false, kind: 'connection'|'timeout'|'http', detail }
 */

const API = (() => {
  const BASE_URL    = 'http://127.0.0.1:8000';
  const PREDICT_URL = `${BASE_URL}/predict_user_claim`;
  const HEALTH_URL  = `${BASE_URL}/health`;
  // Increased to 20s — SHAP TreeExplainer is slow on the first call
  const TIMEOUT_MS  = 20_000;

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
          detail: `Request timed out after ${TIMEOUT_MS / 1000}s. SHAP calculation can be slow on first run — please retry.`,
        };
      }
      return {
        ok: false, kind: 'connection',
        detail: `Cannot reach the backend at ${PREDICT_URL}. Ensure model2main.py is running: uvicorn model2main:app --reload`,
      };
    }

    if (response.status === 200) {
      const data = await response.json();
      return {
        ok:                     true,
        denial_probability_pct: parseFloat(data.denial_probability_pct),
        is_high_risk:           Boolean(data.is_high_risk),
        risk_level:             String(data.risk_level),
        action_recommendation:  String(data.action_recommendation),
        // Pass full SHAP explanation straight through — no transformation
        explanation:            data.explanation ?? null,
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
