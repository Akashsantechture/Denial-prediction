/**
 * js/chat.js
 * ----------
 * Rule-based AI Claims Analyst chat engine.
 *
 * No external LLM required — responses are derived deterministically
 * from the SHAP report data and the live prediction result.
 * Extend by adding entries to the RULES array.
 *
 * Exports (global Chat object)
 * ----------------------------
 *   Chat.init(payload, result, contributions)  — sets context, renders greeting
 *   Chat.send(userMessage)                     — generates + appends response
 *   Chat.clear()                               — resets history
 */

const Chat = (() => {

  // Internal state — set by init()
  let _payload       = null;
  let _result        = null;
  let _contributions = [];

  // ------------------------------------------------------------------ //
  // DOM helpers
  // ------------------------------------------------------------------ //
  function _messagesEl()  { return document.getElementById('chat-messages'); }
  function _inputEl()     { return document.getElementById('chat-input'); }
  function _sendBtn()     { return document.getElementById('chat-send-btn'); }
  function _idleEl()      { return document.getElementById('chat-idle'); }

  function _appendBubble(role, html) {
    const el = _messagesEl();
    if (!el) return;
    _idleEl()?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = `chat-bubble ${role}`;
    div.innerHTML = html;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
  }

  // Convert plain text with **bold** and `code` to safe HTML
  function _md(text) {
    return text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n- /g, '</p><ul><li>')
      .replace(/\n/g, '<br>');
  }

  function _wrap(text) { return `<p>${_md(text)}</p>`; }

  // ------------------------------------------------------------------ //
  // Response engine
  // ------------------------------------------------------------------ //
  function _generate(msg) {
    const m = msg.toLowerCase();

    const diag  = _payload?.diagnosis_code  ?? '?';
    const cpt   = _payload?.activity_code   ?? '?';
    const prob  = _result?.denial_probability_pct ?? 0;
    const risk  = _result?.risk_level ?? '?';
    const los   = _payload?.length_of_stay  ?? 0;
    const lag   = _payload?.billing_lag_days ?? 1;
    const net   = _payload?.claim_net ?? _payload?.claim_gross ?? 0;
    const enc   = _payload?.encounter_type  ?? 'OP';
    const nat   = (_payload?.nationality ?? 'UNKNOWN').toUpperCase();

    const top   = _contributions[0] ?? { name: 'unknown', value: 0 };
    const probVal = Math.min(Math.max(prob / 100, 0.0001), 0.9999);
    const netShap = parseFloat((Math.log(probVal / (1 - probVal)) - SHAP.BASE_LOG_ODDS).toFixed(3));

    // Rule: why / reason / driver
    if (/\b(why|reason|driver|cause|top|main)\b/.test(m)) {
      return `The dominant SHAP driver is **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Combined with the ICD–CPT pairing \`${diag}\` × \`${cpt}\`, the model arrives at **${prob.toFixed(1)}% denial probability**.

Net SHAP log-odds shift from base: **${netShap >= 0 ? '+' : ''}${netShap}** (base = ${SHAP.BASE_LOG_ODDS}).`;
    }

    // Rule: ICD / diagnosis
    if (/\b(icd|diagnos|dx|code)\b/.test(m)) {
      return `ICD code \`${diag}\` contributes **${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}** to the log-odds when paired with CPT \`${cpt}\`.

The ICD category feature carries **1.83% gain** in the v1.5 model. Its interaction with the CPT category can amplify denial risk significantly — especially for Pathology_Laboratory and Radiology pairings.

Check the **ICD Drill-Down** tab for the full denial-rate table for this ICD group.`;
    }

    // Rule: LOS
    if (/\b(los|length.of.stay|stay)\b/.test(m)) {
      const losShap = _contributions.find(c => c.name.toLowerCase().includes('stay'))?.value ?? 0.06;
      const ctx = los <= 5
        ? `LOS of ${los} days is within the typical 0–5 day cluster — near-zero SHAP effect.`
        : `LOS of ${los} days is an outlier — the model flags extended stays as elevated denial risk (SHAP ≈ +${losShap.toFixed(2)}).`;
      return `LOS SHAP for this claim: **${losShap >= 0 ? '+' : ''}${losShap.toFixed(3)}**. ${ctx}

The LOS effect is conditional on the diagnosis — for mismatched ICD–CPT pairs, even moderate LOS amplifies risk. See the **Interactions** tab.`;
    }

    // Rule: billing lag
    if (/\b(billing|lag|submission|delay)\b/.test(m)) {
      const dir = lag <= 3 ? 'increases' : 'reduces';
      return `Billing lag of **${lag} days** ${dir} denial risk for this claim.

The v1.5 model (gain: 5.56%) shows short lags (≤ 3 days) push predictions toward denial — likely reflecting rushed or incomplete documentation. Long lags reduce it.`;
    }

    // Rule: claim amount / financial
    if (/\b(amount|claim|net|gross|cost|financial|aed)\b/.test(m)) {
      const logVal = Math.log1p(Math.max(net, 0)).toFixed(3);
      const direction = parseFloat(logVal) > 5 ? 'positive (pushing toward denial)' : 'near-neutral to negative';
      return `Claim net = **AED ${parseFloat(net).toLocaleString('en', {minimumFractionDigits:2})}** (log: ${logVal}).

\`claim_net_log\` carries **11.44% gain** in the model. At log ≈ ${parseFloat(logVal).toFixed(1)}, the SHAP contribution is ${direction}. Surgery and Unknown CPT claims show the sharpest positive shifts at high claim values.`;
    }

    // Rule: fix / recommendation / appeal
    if (/\b(fix|resolv|appeal|action|recommend|improv|reduc)\b/.test(m)) {
      const revised = Math.max(prob - Math.abs(top.value) * 30, 5).toFixed(0);
      return `To reduce denial risk, address the top driver first: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}).

Estimated revised probability if resolved: **~${revised}%** (down from ${prob.toFixed(1)}%).

Additional steps:
- Verify ICD \`${diag}\` is clinically consistent with CPT \`${cpt}\`.
- Confirm encounter type (\`${enc}\`) matches the actual service setting.
- Review billing documentation before resubmission.`;
    }

    // Rule: risk / score / probability
    if (/\b(risk|score|probabilit|percent|high|medium|low)\b/.test(m)) {
      const diff = (prob - 42).toFixed(1);
      const dir  = prob > 42 ? 'above' : 'below';
      return `This claim is rated **${risk}** with a denial probability of **${prob.toFixed(1)}%**.

The model base rate is 42% — this claim sits ${dir} average by ${Math.abs(parseFloat(diff))} pp. Net SHAP shift: **${netShap >= 0 ? '+' : ''}${netShap}** log-odds.

Risk bands: LOW < 35% · MEDIUM 35–50% · HIGH ≥ 50%.`;
    }

    // Rule: interaction / combination
    if (/\b(interact|combination|pair|together)\b/.test(m)) {
      return `The strongest interaction for \`${diag}\` × \`${cpt}\` involves the ICD and CPT category features — they jointly, not individually, drive part of the denial signal.

Pathology_Laboratory procedures paired with Dermatology, Neurology, or Infectious diagnoses show the highest combined risk (up to 25.21% combined SHAP).

See the **Interactions** tab for the full heatmap.`;
    }

    // Rule: nationality
    if (/\b(nationalit|nation|country|emirati|indian|expat)\b/.test(m)) {
      return `Nationality is \`${nat}\`. In the v1.5 model, nationality carries **1.33% gain** and is a mild contributor.

INDIAN and OTHERS categories show a slight positive SHAP shift compared to EMIRATI in the training distribution, but it is rarely the primary denial driver.`;
    }

    // Rule: model / algorithm
    if (/\b(model|algorithm|xgboost|auc|accuracy|performance)\b/.test(m)) {
      return `The backend uses **XGBoost v1.5** trained on a behavioral healthcare claims dataset.

Performance:
- ROC-AUC: **0.8466**
- Accuracy: **76%** (525,574 test claims)
- Denied recall: 75% | Denied precision: 67%

Top features by gain: \`activity_gross_log\` (17.53%), \`cpt_category\` (16.98%), \`activity_code\` (14.19%), \`claim_net_log\` (11.44%).`;
    }

    // Default
    return `I've analysed this claim (\`${diag}\` / \`${cpt}\`). The model predicts **${prob.toFixed(1)}% denial probability** (${risk} risk).

Top driver: **${top.name}** (${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)} SHAP).

Try asking about: *why denied* · *ICD flagging* · *LOS signal* · *billing lag* · *claim amount* · *fix / recommendation* · *model performance*.`;
  }

  // ------------------------------------------------------------------ //
  // Public API
  // ------------------------------------------------------------------ //
  function init(payload, result, contributions) {
    _payload       = payload;
    _result        = result;
    _contributions = contributions;

    // Enable input
    const inp = _inputEl();
    const btn = _sendBtn();
    if (inp) { inp.disabled = false; inp.placeholder = 'Ask about this claim…'; }
    if (btn) btn.disabled = false;

    // Clear old messages
    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.add('hidden');

    const top3 = contributions.slice(0, 3)
      .map(c => `${c.name} (${c.value >= 0 ? '+' : ''}${c.value.toFixed(3)})`)
      .join(', ');

    const greeting = `I've analysed this claim. The model predicts **${result.denial_probability_pct.toFixed(1)}%** denial probability.

The dominant SHAP drivers are ${top3}.

What would you like to explore?`;

    _appendBubble('assistant', _wrap(greeting));
  }

  function send(userMessage) {
    if (!userMessage.trim() || !_result) return;
    _appendBubble('user', _wrap(userMessage));
    const response = _generate(userMessage);
    // Small delay for natural feel
    setTimeout(() => _appendBubble('assistant', _wrap(response)), 120);
  }

  function clear() {
    _payload = null; _result = null; _contributions = [];
    const el = _messagesEl();
    if (el) el.innerHTML = '';
    _idleEl()?.classList.remove('hidden');
    const inp = _inputEl();
    const btn = _sendBtn();
    if (inp) { inp.disabled = true; inp.placeholder = 'Submit a claim to activate…'; }
    if (btn) btn.disabled = true;
  }

  return { init, send, clear };
})();
