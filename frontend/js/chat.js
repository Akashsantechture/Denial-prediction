/**
 * js/chat.js
 * ----------
 * AI Claims Analyst chat engine.
 *
 * Strategy (LLM-first with rule-based fallback)
 * ----------------------------------------------
 * 1. On Chat.init() a session is created on the AI Analyst microservice
 *    (port 8060) by posting the full claim intelligence (payload + SHAP result).
 *    If the microservice is unreachable the module stays in RULE_BASED mode.
 *
 * 2. On every Chat.send() the message is first forwarded to the microservice
 *    via AnalystAPI.chat().  If the call succeeds the Gemini LLM answer is
 *    displayed directly.
 *
 * 3. If the LLM call fails for any reason (network error, timeout, Gemini
 *    quota, 500 from the service) the rule-based engine in _generateRuleBased()
 *    produces an answer locally and appends a subtle fallback notice so the
 *    user knows they are seeing a local response.
 *
 * Mode tracking
 * -------------
 *   _mode === 'LLM'        — session created, LLM answers preferred
 *   _mode === 'RULE_BASED' — microservice unavailable, all answers local
 *   _mode === 'INACTIVE'   — no claim submitted yet
 *
 * Exports (global Chat object)
 * ----------------------------
 *   Chat.init(payload, result, contributions)  — sets context, renders greeting
 *   Chat.send(userMessage)                     — sends message, handles fallback
 *   Chat.clear()                               — resets all state
 */

const Chat = (() => {

  // ------------------------------------------------------------------ //
  // Module state
  // ------------------------------------------------------------------ //
  let _payload       = null;
  let _result        = null;
  let _contributions = [];
  let _sessionId     = null;          // analyst microservice session id
  let _mode          = 'INACTIVE';    // 'INACTIVE' | 'LLM' | 'RULE_BASED'

  // ------------------------------------------------------------------ //
  // DOM helpers
  // ------------------------------------------------------------------ //
  function _messagesEl() { return document.getElementById('chat-messages'); }
  function _inputEl()    { return document.getElementById('chat-input'); }
  function _sendBtn()    { return document.getElementById('chat-send-btn'); }
  function _idleEl()     { return document.getElementById('chat-idle'); }
  function _modeEl()     { return document.getElementById('chat-mode-badge'); }

  function _appendBubble(role, html) {
    const el = _messagesEl();
    if (!el) return;
    _idleEl()?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = `chat-bubble ${role}`;
    div.innerHTML = html;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    return div;
  }

  // Convert **bold** and `code` markdown to safe HTML
  function _md(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n- /g, '</p><ul><li>')
      .replace(/\n/g, '<br>');
  }

  function _wrap(text) { return `<p>${_md(text)}</p>`; }

  // Render a "thinking…" placeholder that can be replaced later
  function _appendThinking() {
    const el = _messagesEl();
    if (!el) return null;
    _idleEl()?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = 'chat-bubble assistant thinking';
    div.innerHTML = '<span class="thinking-dots"><em>Thinking…</em></span>';
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    return div;
  }

  // Update the mode badge in the chat header (if present)
  function _updateModeBadge() {
    const el = _modeEl();
    if (!el) return;
    if (_mode === 'LLM') {
      el.textContent  = '✦ LLM connected ';
      el.className    = 'chat-mode-badge llm';
      el.title        = 'Responses powered by Gemini via the AI Analyst microservice';
    } else if (_mode === 'RULE_BASED') {
      el.textContent  = '⚡ Rule-based';
      el.className    = 'chat-mode-badge rule';
      el.title        = 'AI Analyst microservice unavailable — using local rule engine';
    } else {
      el.textContent  = '';
      el.className    = 'chat-mode-badge';
    }
  }

  // ------------------------------------------------------------------ //
  // Build claim intelligence object sent to the AI Analyst microservice.
  // This merges the raw claim payload with the full SHAP result so Gemini
  // has complete context: claim facts + model prediction + SHAP evidence.
  // ------------------------------------------------------------------ //
  function _buildClaimIntelligence(payload, result, contributions) {
    return {
      // ---- Claim facts ----
      claim: {
        diagnosis_code:       payload.diagnosis_code,
        diagnosis_type:       payload.diagnosis_type,
        activity_code:        payload.activity_code,
        activity_gross:       payload.activity_gross,
        activity_quantity:    payload.activity_quantity,
        claim_gross:          payload.claim_gross,
        claim_net:            payload.claim_net,
        patient_age:          payload.patient_age,
        gender:               payload.gender,
        nationality:          payload.nationality,
        payer_id:             payload.payer_id,
        insurance_plan_tier:  payload.insurance_plan_tier,
        clinician_profession: payload.clinician_profession,
        clinician_category:   payload.clinician_category,
        facility_type:        payload.facility_type,
        billing_lag_days:     payload.billing_lag_days,
        length_of_stay:       payload.length_of_stay,
        encounter_type:       payload.encounter_type,
      },
      // ---- Model prediction ----
      prediction: {
        denial_probability_pct:  result.denial_probability_pct,
        risk_level:              result.risk_level,
        is_high_risk:            result.is_high_risk,
        action_recommendation:   result.action_recommendation,
      },
      // ---- SHAP evidence (full API explanation if available) ----
      shap_explanation: result.explanation ?? null,
      // ---- Top contributions derived by shap.js (always available) ----
      top_shap_contributions: contributions.slice(0, 10).map(c => ({
        feature:    c.name,
        shap_value: c.value,
        direction:  c.value >= 0 ? 'increases_denial_likelihood' : 'reduces_denial_likelihood',
      })),
    };
  }

  // ------------------------------------------------------------------ //
  // Rule-based fallback engine
  // Identical logic to the previous standalone engine — preserved in full
  // so the fallback is feature-complete.
  // ------------------------------------------------------------------ //
  function _generateRuleBased(msg) {
    const m = msg.toLowerCase();

    const diag  = _payload?.diagnosis_code  ?? '?';
    const cpt   = _payload?.activity_code   ?? '?';
    const prob  = _result?.denial_probability_pct ?? 0;
    const level = _result?.risk_level ?? '?';
    const los   = _payload?.length_of_stay  ?? 0;
    const lag   = _payload?.billing_lag_days ?? 1;
    const net   = _payload?.claim_net ?? _payload?.claim_gross ?? 0;
    const enc   = _payload?.encounter_type  ?? 'OP';
    const nat   = (_payload?.nationality ?? 'UNKNOWN').toUpperCase();

    const top      = _contributions[0] ?? { name: 'unknown', value: 0 };
    const probVal  = Math.min(Math.max(prob / 100, 0.0001), 0.9999);
    const realBase = SHAP.baseLogOdds(_result);
    const netShap  = parseFloat((Math.log(probVal / (1 - probVal)) - realBase).toFixed(3));

    if (/\b(why|reason|driver|cause|top|main)\b/.test(m)) {
      return `The dominant SHAP driver is **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Combined with the ICD–CPT pairing \`${diag}\` × \`${cpt}\`, the model arrives at **${prob.toFixed(1)}% denial probability**.

Net SHAP log-odds shift from base: **${netShap >= 0 ? '+' : ''}${netShap}** (base = ${SHAP.BASE_LOG_ODDS}).`;
    }

    if (/\b(icd|diagnos|dx|code)\b/.test(m)) {
      return `ICD code \`${diag}\` contributes **${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}** to the log-odds when paired with CPT \`${cpt}\`.

The ICD category feature carries **1.83% gain** in the v1.5 model. Its interaction with the CPT category can amplify denial likelihood significantly — especially for Pathology_Laboratory and Radiology pairings.

Check the **ICD Drill-Down** tab for the full denial-rate table for this ICD group.`;
    }

    if (/\b(los|length.of.stay|stay)\b/.test(m)) {
      const losShap = _contributions.find(c => c.name.toLowerCase().includes('stay'))?.value ?? 0.06;
      const ctx = los <= 5
        ? `LOS of ${los} days is within the typical 0–5 day cluster — near-zero SHAP effect.`
        : `LOS of ${los} days is an outlier — the model flags extended stays as elevated denial likelihood (SHAP ≈ +${losShap.toFixed(2)}).`;
      return `LOS SHAP for this claim: **${losShap >= 0 ? '+' : ''}${losShap.toFixed(3)}**. ${ctx}

The LOS effect is conditional on the diagnosis — for mismatched ICD–CPT pairs, even moderate LOS amplifies denial likelihood. See the **Interactions** tab.`;
    }

    if (/\b(billing|lag|submission|delay)\b/.test(m)) {
      const dir = lag <= 3 ? 'increases' : 'reduces';
      return `Billing lag of **${lag} days** ${dir} denial likelihood for this claim.

The v1.5 model (gain: 5.56%) shows short lags (≤ 3 days) push predictions toward denial — likely reflecting rushed or incomplete documentation. Long lags reduce it.`;
    }

    if (/\b(amount|claim|net|gross|cost|financial|aed)\b/.test(m)) {
      const logVal   = Math.log1p(Math.max(net, 0)).toFixed(3);
      const direction = parseFloat(logVal) > 5 ? 'positive (pushing toward denial)' : 'near-neutral to negative';
      return `Claim net = **AED ${parseFloat(net).toLocaleString('en', { minimumFractionDigits: 2 })}** (log: ${logVal}).

\`claim_net_log\` carries **11.44% gain** in the model. At log ≈ ${parseFloat(logVal).toFixed(1)}, the SHAP contribution is ${direction}. Surgery and Unknown CPT claims show the sharpest positive shifts at high claim values.`;
    }

    if (/\b(fix|resolv|appeal|action|recommend|improv|reduc)\b/.test(m)) {
      const revised = Math.max(prob - Math.abs(top.value) * 30, 5).toFixed(0);
      return `To reduce denial likelihood, address the top driver first: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Estimated revised probability if resolved: **~${revised}%** (down from ${prob.toFixed(1)}%).

Additional steps:
- Verify ICD \`${diag}\` is clinically consistent with CPT \`${cpt}\`.
- Confirm encounter type (\`${enc}\`) matches the actual service setting.
- Review billing documentation before resubmission.`;
    }

    if (/\b(likelihood|risk|score|probabilit|percent|high|medium|low)\b/.test(m)) {
      return `This claim is rated **${level}** with a denial probability of **${prob.toFixed(1)}%**.

The model base: **${realBase.toFixed(3)}** log-odds. This claim sits ${prob > 42 ? 'above' : 'below'} the average by ${Math.abs(prob - 42).toFixed(1)} pp. Net SHAP shift: **${netShap >= 0 ? '+' : ''}${netShap}** log-odds.

Likelihood bands: LOW < 35% · MEDIUM 35–50% · HIGH ≥ 50%.`;
    }

    if (/\b(interact|combination|pair|together)\b/.test(m)) {
      return `The strongest interaction for \`${diag}\` × \`${cpt}\` involves the ICD and CPT category features — they jointly, not individually, drive part of the denial signal.

Pathology_Laboratory procedures paired with Dermatology, Neurology, or Infectious diagnoses show the highest combined denial likelihood (up to 25.21% combined SHAP).

See the **Interactions** tab for the full heatmap.`;
    }

    if (/\b(nationalit|nation|country|emirati|indian|expat)\b/.test(m)) {
      return `Nationality is \`${nat}\`. In the v1.5 model, nationality carries **1.33% gain** and is a mild contributor.

INDIAN and OTHERS categories show a slight positive SHAP shift compared to EMIRATI in the training distribution, but it is rarely the primary denial driver.`;
    }

    if (/\b(model|algorithm|xgboost|auc|accuracy|performance)\b/.test(m)) {
      return `The backend uses **XGBoost v1.5** trained on a behavioral healthcare claims dataset.

Performance:
- ROC-AUC: **0.8466**
- Accuracy: **76%** (525,574 test claims)
- Denied recall: 75% | Denied precision: 67%

Top features by gain: \`activity_gross_log\` (17.53%), \`cpt_category\` (16.98%), \`activity_code\` (14.19%), \`claim_net_log\` (11.44%).`;
    }

    // Default
    return `I've analysed this claim (\`${diag}\` / \`${cpt}\`). The model predicts **${prob.toFixed(1)}% denial probability** (${level} likelihood).

Top driver: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)} SHAP).

Try asking about: *why denied* · *ICD flagging* · *LOS signal* · *billing lag* · *claim amount* · *fix / recommendation* · *model performance*.`;
  }

  // ------------------------------------------------------------------ //
  // Core send logic
  // Tries the LLM, falls back to rule-based on any failure.
  // ------------------------------------------------------------------ //
  async function _generateResponse(userMessage) {
    // LLM path
    if (_mode === 'LLM' && _sessionId) {
      const llmResult = await AnalystAPI.chat(_sessionId, userMessage);
      if (llmResult.ok) {
        return { text: llmResult.answer, source: 'llm' };
      }
      // LLM failed — fall through to rule-based with a notice
      console.warn('[Chat] LLM call failed, falling back to rule-based:', llmResult.detail);
    }

    // Rule-based path
    const text = _generateRuleBased(userMessage);
    return { text, source: 'fallback' };
  }

  // ------------------------------------------------------------------ //
  // Public API
  // ------------------------------------------------------------------ //

  /**
   * Initialise the chat for a new claim result.
   * Called by app.js after a successful prediction.
   */
  async function init(payload, result, contributions) {
    _payload       = payload;
    _result        = result;
    _contributions = contributions;
    _sessionId     = null;
    _mode          = 'RULE_BASED';   // safe default until session confirmed

    // Enable input immediately so the UI isn't blocked
    const inp = _inputEl();
    const btn = _sendBtn();
    if (inp) { inp.disabled = false; inp.placeholder = 'Ask about this claim…'; }
    if (btn) btn.disabled = false;

    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.add('hidden');

    // Build greeting from local data (always fast, no network dependency)
    const baseVal  = SHAP.baseLogOdds(result);
    const finalVal = SHAP.finalLogOdds(result);
    const netShap  = (finalVal - baseVal).toFixed(3);

    const top3 = contributions.slice(0, 3).map(c => {
      const sign = c.value >= 0 ? '+' : '';
      const dir  = c.value >= 0 ? '↑ denial' : '↓ approval';
      return `**${c.name}** (${sign}${c.value.toFixed(3)}, ${dir})`;
    }).join(', ');

    const greeting =
      `I've analysed this claim using the **actual model SHAP values**.\n\n` +
      `The model predicts **${result.denial_probability_pct.toFixed(1)}% denial probability** ` +
      `(${result.risk_level} likelihood). Net SHAP shift: **${netShap >= 0 ? '+' : ''}${netShap}** from base ${baseVal.toFixed(3)}.\n\n` +
      `Top drivers: ${top3}.\n\nWhat would you like to explore?`;

    _appendBubble('assistant', _wrap(greeting));

    // Attempt to create an LLM session in the background — does not block UI
    _initLLMSession(payload, result, contributions);
  }

  /**
   * Attempt to create an AI Analyst session.
   * Updates _mode and _sessionId when done.
   * Runs asynchronously after the greeting is already shown.
   */
  async function _initLLMSession(payload, result, contributions) {
    const claimIntelligence = _buildClaimIntelligence(payload, result, contributions);
    const sessionResult = await AnalystAPI.createSession(claimIntelligence);

    if (sessionResult.ok) {
      _sessionId = sessionResult.session_id;
      _mode      = 'LLM';
      console.log('[Chat] AI Analyst session created:', _sessionId);
    } else {
      _mode = 'RULE_BASED';
      console.warn('[Chat] AI Analyst unavailable, using rule-based fallback:', sessionResult.detail);
    }

    _updateModeBadge();
  }

  /**
   * Send a user message.
   * Appends user bubble → thinking indicator → awaits response → replaces indicator.
   */
  async function send(userMessage) {
    if (!userMessage.trim() || !_result) return;

    // Disable input while waiting to prevent double-sends
    const inp = _inputEl();
    const btn = _sendBtn();
    if (inp) inp.disabled = true;
    if (btn) btn.disabled = true;

    _appendBubble('user', _wrap(userMessage));

    const thinking = _appendThinking();

    const { text, source } = await _generateResponse(userMessage);

    // Replace thinking bubble with real answer
    if (thinking && thinking.parentNode) {
      thinking.className = `chat-bubble assistant`;
      let html = _wrap(text);
      // Append a subtle notice when the fallback was used unexpectedly
      if (source === 'fallback' && _mode === 'LLM') {
        html += `<p class="chat-fallback-notice">⚡ Local response — AI Analyst temporarily unavailable.</p>`;
      }
      thinking.innerHTML = html;
      _messagesEl().scrollTop = _messagesEl().scrollHeight;
    }

    // Re-enable input
    if (inp) { inp.disabled = false; inp.focus(); }
    if (btn) btn.disabled = false;
  }

  /**
   * Reset all state and UI.
   * Called by app.js on new form submission or error.
   */
  function clear() {
    // Delete the remote session (fire-and-forget)
    if (_sessionId) {
      AnalystAPI.deleteSession(_sessionId);
    }

    _payload       = null;
    _result        = null;
    _contributions = [];
    _sessionId     = null;
    _mode          = 'INACTIVE';

    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.remove('hidden');

    const inp = _inputEl();
    const btn = _sendBtn();
    if (inp) { inp.disabled = true; inp.placeholder = 'Submit a claim to activate…'; }
    if (btn) btn.disabled = true;

    _updateModeBadge();
  }

  return { init, send, clear };
})();
