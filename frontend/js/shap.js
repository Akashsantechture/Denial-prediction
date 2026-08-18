/**
 * js/shap.js
 * ----------
 * SHAP bridge — converts the REAL API explanation into the contribution
 * array that charts.js, app.js, and chat.js all expect.
 *
 * The frontend NO LONGER estimates SHAP values.
 * All values come from model2main.py TreeExplainer (actual model output).
 *
 * Exports (global SHAP object)
 * ----------------------------
 *   SHAP.fromApiResult(result)
 *     → Array<{ name, value, displayName, direction, impactLevel,
 *               shareP, rawValue, explanation }>
 *       sorted by |value| descending
 *
 *   SHAP.finalLogOdds(result)   — real final_model_value from API
 *   SHAP.baseLogOdds(result)    — real base_value from API
 *   SHAP.BASE_LOG_ODDS          — fallback constant only
 */

const SHAP = (() => {

  // Fallback constant — used only when API returns no explanation
  const BASE_LOG_ODDS = 0.42;

  /**
   * Build the standard contributions array from the API explanation.
   * Uses result.explanation.all_features (real TreeExplainer values).
   *
   * @param {Object} result — PredictionSuccess from api.js
   * @returns {Array<{name, value, displayName, direction, impactLevel, shareP, rawValue, explanation}>}
   */
  function fromApiResult(result) {
    const expl = result?.explanation;

    if (!expl || !Array.isArray(expl.all_features) || expl.all_features.length === 0) {
      console.warn('[SHAP] No explanation in API result — charts will be empty.');
      return [];
    }

    return expl.all_features
      .map(f => ({
        name:        f.display_name || f.feature,
        value:       typeof f.shap_value === 'number' ? f.shap_value : 0,
        displayName: f.display_name || f.feature,
        direction:   f.direction    || 'neutral',
        impactLevel: f.impact_level || 'LOW',
        shareP:      typeof f.contribution_share_pct === 'number' ? f.contribution_share_pct : 0,
        rawValue:    f.raw_value   ?? null,
        explanation: f.explanation || '',
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  /** Real final model value (log-odds) returned by the API. */
  function finalLogOdds(result) {
    return result?.explanation?.final_model_value ?? _probToLogOdds(result?.denial_probability_pct ?? 50);
  }

  /** Real SHAP base value returned by the API. */
  function baseLogOdds(result) {
    return result?.explanation?.base_value ?? BASE_LOG_ODDS;
  }

  /** Convert probability % to log-odds (fallback only). */
  function _probToLogOdds(prob_pct) {
    const p = Math.min(Math.max(prob_pct / 100, 0.0001), 0.9999);
    return Math.log(p / (1 - p));
  }

  return { fromApiResult, finalLogOdds, baseLogOdds, BASE_LOG_ODDS };
})();
