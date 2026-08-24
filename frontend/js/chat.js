/**
 * js/chat.js
 * ----------
 * AI Claims Analyst chat engine — wired to predictionapi.py response shape.
 *
 * API response shape used here:
 * {
 *   claim_id,
 *   claim_summary: {
 *     claim_denial_probability_pct,
 *     claim_risk_level,
 *     highest_activity_risk,
 *     average_activity_risk,
 *     activity_count
 *   },
 *   predictions: [{
 *     activity_code, cpt_category, denial_probability,
 *     predicted_denial, top_drivers, recommendation
 *   }]
 * }
 *
 * Strategy — LLM-first, rule-based fallback
 * ------------------------------------------
 * 1. Chat.init() creates a session on the AI Analyst (port 8060) in the
 *    background. Greeting is shown immediately without waiting.
 * 2. Every Chat.send() tries AnalystAPI.chat() first. If that fails for
 *    any reason the local _generateRuleBased() runs instead.
 * 3. _mode tracks 'LLM' | 'RULE_BASED' | 'INACTIVE' and the mode badge
 *    in the drawer header reflects the current engine.
 */

const Chat = (() => {

  let _payload       = null;
  let _result        = null;   // full API result { claim_summary, predictions }
  let _contributions = [];     // SHAP.fromApiResult() output
  let _sessionId     = null;
  let _mode          = 'INACTIVE';

  // ── DOM helpers ──────────────────────────────────────────────────────
  const _messagesEl = () => document.getElementById('chat-messages');
  const _inputEl    = () => document.getElementById('chat-input');
  const _sendBtn    = () => document.getElementById('chat-send-btn');
  const _idleEl     = () => document.getElementById('chat-idle');
  const _modeEl     = () => document.getElementById('chat-mode-badge');

  function _appendBubble(role, html) {
    const el = _messagesEl();
    if (!el) return null;
    _idleEl()?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = `chat-bubble ${role}`;
    div.innerHTML = html;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    return div;
  }

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

  function _md(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n- /g, '</p><ul><li>')
      .replace(/\n/g, '<br>');
  }
  const _wrap = text => `<p>${_md(text)}</p>`;

  function _updateModeBadge() {
    const el = _modeEl();
    if (!el) return;
    if (_mode === 'LLM') {
      el.textContent = '✦ Gemini AI';
      el.className   = 'chat-mode-badge llm';
      el.title       = 'Powered by Gemini via the AI Analyst microservice';
    } else if (_mode === 'RULE_BASED') {
      el.textContent = '⚡ Rule-based';
      el.className   = 'chat-mode-badge rule';
      el.title       = 'AI Analyst unavailable — using local rule engine';
    } else {
      el.textContent = '';
      el.className   = 'chat-mode-badge';
    }
  }

  // ── Claim intelligence builder ───────────────────────────────────────
  // Maps new API shape → a structured object the AI Analyst understands.
  function _buildClaimIntelligence(payload, result, contributions) {
    const cs   = result.claim_summary ?? {};
    const preds = result.predictions  ?? [];
    return {
      claim: {
        claim_id:             result.claim_id,
        diagnosis_code:       payload.diagnosis_code,
        icd_category:         payload.icd_category,
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
        payer_classification: payload.payer_classification,
        activities:           payload.activities ?? [],
      },
      prediction: {
        claim_denial_probability_pct: cs.claim_denial_probability_pct,
        claim_risk_level:             cs.claim_risk_level,
        highest_activity_risk_pct:    (cs.highest_activity_risk ?? 0) * 100,
        average_activity_risk_pct:    (cs.average_activity_risk  ?? 0) * 100,
        activity_count:               cs.activity_count,
        activity_predictions:         preds.map(p => ({
          activity_code:      p.activity_code,
          cpt_category:       p.cpt_category,
          denial_probability: p.denial_probability,
          predicted_denial:   p.predicted_denial,
          recommendation:     p.recommendation,
        })),
      },
      top_shap_contributions: contributions.slice(0, 10).map(c => ({
        feature:    c.name,
        shap_value: c.value,
        direction:  c.value >= 0 ? 'increases_denial' : 'reduces_denial',
      })),
    };
  }

  // ── Rule-based fallback engine ───────────────────────────────────────
  // Uses the new claim_summary shape throughout.
  function _generateRuleBased(msg) {
    const m = msg.toLowerCase();

    const cs    = _result?.claim_summary ?? {};
    const preds = _result?.predictions   ?? [];

    const prob  = cs.claim_denial_probability_pct ?? 0;
    const level = (cs.claim_risk_level ?? 'UNKNOWN').toLowerCase();
    const diag  = _payload?.diagnosis_code    ?? '?';
    const los   = _payload?.length_of_stay    ?? 0;
    const lag   = _payload?.billing_lag_days  ?? 1;
    const net   = _payload?.claim_net ?? _payload?.claim_gross ?? 0;
    const enc   = _payload?.encounter_type    ?? 'OP';
    const nat   = (_payload?.nationality ?? 'UNKNOWN').toUpperCase();
    const acts  = _payload?.activities ?? [];

    // Highest-risk activity
    const topPred  = [...preds].sort((a, b) => b.denial_probability - a.denial_probability)[0];
    const topActCode = topPred?.activity_code ?? (acts[0]?.activity_code ?? '?');
    const topActCat  = topPred?.cpt_category  ?? (acts[0]?.cpt_category  ?? '?');

    // SHAP top driver
    const top      = _contributions[0] ?? { name: 'unknown', value: 0 };
    const probVal  = Math.min(Math.max(prob / 100, 0.0001), 0.9999);
    const realBase = SHAP.baseLogOdds(_result);
    const netShap  = parseFloat((Math.log(probVal / (1 - probVal)) - realBase).toFixed(3));

    if (/\b(why|reason|driver|cause|top|main)\b/.test(m)) {
      return `The dominant SHAP driver for this claim is **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

The highest-risk activity is \`${topActCode}\` (${topActCat}) with a denial probability of **${((topPred?.denial_probability ?? 0) * 100).toFixed(1)}%**.

Net SHAP log-odds shift from base: **${netShap >= 0 ? '+' : ''}${netShap}** (base = ${realBase.toFixed(3)}).`;
    }

    if (/\b(activit|cpt|procedure|line)\b/.test(m)) {
      const actLines = preds.map(p =>
        `- \`${p.activity_code}\` (${p.cpt_category}): **${(p.denial_probability * 100).toFixed(1)}%** — ${p.predicted_denial ? '⚠ likely denied' : '✓ likely approved'}`
      ).join('\n');
      return `This claim has **${preds.length} activit${preds.length === 1 ? 'y' : 'ies'}**:\n\n${actLines || 'No activity detail available.'}\n\nOverall claim denial score: **${prob.toFixed(1)}%** (${cs.claim_risk_level}).`;
    }

    if (/\b(icd|diagnos|dx|code)\b/.test(m)) {
      return `ICD code \`${diag}\` is categorised as **${API.deriveIcdCategory(diag)}**.

The ICD category is one of the 19 model features. Its interaction with the CPT category can amplify denial likelihood — especially for Pathology_Laboratory and Radiology pairings.

Check the **ICD Drill-Down** tab for the population-level denial rates for this ICD group.`;
    }

    if (/\b(los|length.of.stay|stay)\b/.test(m)) {
      const losShap = _contributions.find(c => c.name.toLowerCase().includes('stay'))?.value ?? 0;
      const ctx = los <= 5
        ? `LOS of ${los} days is within the typical 0–5 day cluster — near-zero SHAP effect.`
        : `LOS of ${los} days is an outlier — the model flags extended stays as elevated denial likelihood (SHAP ≈ +${losShap.toFixed(2)}).`;
      return `LOS SHAP for this claim: **${losShap >= 0 ? '+' : ''}${losShap.toFixed(3)}**. ${ctx}`;
    }

    if (/\b(billing|lag|submission|delay)\b/.test(m)) {
      const dir = lag <= 3 ? 'increases' : 'reduces';
      return `Billing lag of **${lag} day${lag !== 1 ? 's' : ''}** ${dir} denial likelihood for this claim.

Short lags (≤ 3 days) push predictions toward denial — likely reflecting rushed or incomplete documentation. Longer lags reduce it.`;
    }

    if (/\b(amount|claim|gross|net|cost|financial|aed)\b/.test(m)) {
      const fmtAED = v => parseFloat(v).toLocaleString('en', { minimumFractionDigits: 2 });
      return `Claim gross = **AED ${fmtAED(payload?.claim_gross ?? 0)}** · Claim net = **AED ${fmtAED(payload?.claim_net ?? 0)}**.

\`claim_gross\` and \`claim_net\` are key model features. Higher claim amounts generally push the prediction toward denial.`;
    }

    if (/\b(fix|resolv|appeal|action|recommend|improv|reduc)\b/.test(m)) {
      const rec = topPred?.recommendation ?? 'Review claim documentation before resubmission.';
      const revised = Math.max(prob - Math.abs(top.value) * 30, 5).toFixed(0);
      return `**Recommendation for highest-risk activity (\`${topActCode}\`):** ${rec}

To reduce overall denial likelihood, address the top SHAP driver: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Estimated revised score if resolved: **~${revised}%** (from ${prob.toFixed(1)}%).

Steps:
- Verify ICD \`${diag}\` is clinically consistent with CPT \`${topActCode}\`.
- Confirm encounter type (\`${enc}\`) matches the actual service setting.
- Review billing documentation before resubmission.`;
    }

    if (/\b(likelihood|risk|score|probabilit|percent|high|medium|low)\b/.test(m)) {
      return `This claim scores **${prob.toFixed(1)}%** — rated **${cs.claim_risk_level}**.

Scoring formula: **0.7 × highest activity risk + 0.3 × average activity risk**.
- Highest activity: **${((cs.highest_activity_risk ?? 0) * 100).toFixed(1)}%**
- Average activity: **${((cs.average_activity_risk  ?? 0) * 100).toFixed(1)}%**

Likelihood bands: LOW < 40% · MEDIUM 40–70% · HIGH ≥ 70%.`;
    }

    if (/\b(interact|combination|pair|together)\b/.test(m)) {
      return `The strongest interaction for ICD \`${diag}\` × CPT \`${topActCat}\` involves the ICD and CPT category features — they jointly drive part of the denial signal.

See the **Interactions** tab for the full heatmap of denial rates across ICD × CPT combinations.`;
    }

    if (/\b(nationalit|nation|country|emirati|indian|expat)\b/.test(m)) {
      return `Nationality is \`${nat}\`. It is one of the 19 model features and acts as a mild modifier. INDIAN and OTHERS categories show a slight positive SHAP shift compared to EMIRATI but it is rarely the primary driver.`;
    }

    if (/\b(model|algorithm|xgboost|auc|accuracy|performance)\b/.test(m)) {
      return `The backend uses **XGBoost** with a **Target Encoder** trained on UAE healthcare claims.

Features (19 total): activity_code, activity_quantity, activity_gross, patient_age, gender, nationality, claim_gross, claim_net, encounter_type, length_of_stay, clinician_profession, clinician_category, facility_type, payer_classification, diagnosis_code, billing_lag_days, icd_category, cpt_category, icd_cpt_domain_match.

Claim-level score = **0.7 × max activity risk + 0.3 × average activity risk**.`;
    }

    return `This claim scores **${prob.toFixed(1)}% denial likelihood** (${cs.claim_risk_level ?? '?'}) across ${preds.length} activit${preds.length === 1 ? 'y' : 'ies'}.

Top SHAP driver: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Try asking about: *why denied* · *activities* · *ICD code* · *LOS* · *billing lag* · *claim amount* · *fix / recommendation* · *model info*.`;
  }

  // ── Core response logic ──────────────────────────────────────────────
  async function _generateResponse(userMessage) {
    if (_mode === 'LLM' && _sessionId) {
      const llmResult = await AnalystAPI.chat(_sessionId, userMessage);
      if (llmResult.ok) return { text: llmResult.answer, source: 'llm' };
      console.warn('[Chat] LLM fallback:', llmResult.detail);
    }
    return { text: _generateRuleBased(userMessage), source: 'fallback' };
  }

  // ── Public API ───────────────────────────────────────────────────────

  async function init(payload, result, contributions) {
    _payload       = payload;
    _result        = result;
    _contributions = contributions;
    _sessionId     = null;
    _mode          = 'RULE_BASED';

    const inp = _inputEl(), btn = _sendBtn();
    if (inp) { inp.disabled = false; inp.placeholder = 'Ask about this claim…'; }
    if (btn) btn.disabled = false;

    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.add('hidden');

    // Greeting — built from local data, instant
    const cs       = result.claim_summary ?? {};
    const preds    = result.predictions ?? [];
    const baseVal  = SHAP.baseLogOdds(result);
    const finalVal = SHAP.finalLogOdds(result);
    const netShap  = (finalVal - baseVal).toFixed(3);
    const pct      = cs.claim_denial_probability_pct ?? 0;
    const topPred  = [...preds].sort((a, b) => b.denial_probability - a.denial_probability)[0];

    const top3 = contributions.slice(0, 3).map(c => {
      const sign = c.value >= 0 ? '+' : '';
      const dir  = c.value >= 0 ? '↑ denial' : '↓ approval';
      return `**${c.name}** (${sign}${c.value.toFixed(3)}, ${dir})`;
    }).join(', ');

    const greeting =
      `Claim analysed — **${pct.toFixed(1)}% denial score** (${cs.claim_risk_level ?? '?'}).\n\n` +
      `${preds.length} activit${preds.length === 1 ? 'y' : 'ies'} evaluated` +
      (topPred ? `. Highest-risk: \`${topPred.activity_code}\` at **${(topPred.denial_probability * 100).toFixed(1)}%**.` : '.') + `\n\n` +
      `Top SHAP drivers: ${top3 || '—'}.\n\nWhat would you like to explore?`;

    _appendBubble('assistant', _wrap(greeting));
    _initLLMSession(payload, result, contributions);
  }

  async function _initLLMSession(payload, result, contributions) {
    const ci = _buildClaimIntelligence(payload, result, contributions);
    const sr = await AnalystAPI.createSession(ci);
    if (sr.ok) {
      _sessionId = sr.session_id;
      _mode      = 'LLM';
    } else {
      _mode = 'RULE_BASED';
      console.warn('[Chat] AI Analyst unavailable:', sr.detail);
    }
    _updateModeBadge();
  }

  async function send(userMessage) {
    if (!userMessage.trim() || !_result) return;
    const inp = _inputEl(), btn = _sendBtn();
    if (inp) inp.disabled = true;
    if (btn) btn.disabled = true;

    _appendBubble('user', _wrap(userMessage));
    const thinking = _appendThinking();
    const { text, source } = await _generateResponse(userMessage);

    if (thinking && thinking.parentNode) {
      thinking.className = 'chat-bubble assistant';
      let html = _wrap(text);
      if (source === 'fallback' && _mode === 'LLM') {
        html += `<p class="chat-fallback-notice">⚡ Local response — AI Analyst temporarily unavailable.</p>`;
      }
      thinking.innerHTML = html;
      _messagesEl().scrollTop = _messagesEl().scrollHeight;
    }
    if (inp) { inp.disabled = false; inp.focus(); }
    if (btn) btn.disabled = false;
  }

  function clear() {
    if (_sessionId) AnalystAPI.deleteSession(_sessionId);
    _payload = null; _result = null; _contributions = [];
    _sessionId = null; _mode = 'INACTIVE';
    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.remove('hidden');
    const inp = _inputEl(), btn = _sendBtn();
    if (inp) { inp.disabled = true; inp.placeholder = 'Submit a claim to activate…'; }
    if (btn) btn.disabled = true;
    _updateModeBadge();
  }

  return { init, send, clear };
})();
