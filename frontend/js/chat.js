/**
 * js/chat.js
 * ----------
 * AI Claims Analyst — LLM-first with deep per-activity rule-based fallback.
 *
 * Schema alignment (predictionapi.py v4.0 / validators/schemas.py)
 * ----------------------------------------------------------------
 *   Payload fields used:
 *     primary_diagnosis_code  (was: diagnosis_code)
 *     patient_age, gender, nationality
 *     encounter_type, length_of_stay
 *     claim_gross, claim_net
 *     clinician_profession, facility_type, payer_classification
 *     activities[].{ activity_code, activity_quantity, activity_gross,
 *                    diagnoses[].{ diagnosis_code, diagnosis_type } }
 *
 *   Removed fields (no longer in ClaimRequest):
 *     billing_lag_days, clinician_category, icd_category,
 *     insurance_plan_tier, cpt_category (on activity)
 *
 *   Result fields:
 *     predictions[].{ activity_code, cpt_category*, denial_probability,
 *                     predicted_denial, top_drivers, recommendation* }
 *     (* enriched client-side by app.js _enrichPredictions)
 *
 * Rule engine coverage
 * --------------------
 *   Claim-level:
 *     overall score, scoring formula, risk bands, claim financials,
 *     patient demographics, primary ICD code, LOS,
 *     clinician, facility, payer, model info
 *
 *   Activity-level:
 *     "activity 99213"   → full breakdown for a specific activity code
 *     "activity 2"       → breakdown by position number
 *     "highest risk"     → deepest analysis of the most dangerous activity
 *     "why denied"       → ranked list of all activities + top SHAP drivers each
 *     "shap drivers"     → per-activity SHAP driver deep-dive for every activity
 *     "compare"          → side-by-side comparison table of all activities
 *     "fix 99213"        → specific remediation for a named activity
 *     "fix all"          → remediation steps for every denied activity
 *     named feature      → what a SHAP feature means + how it affects this claim
 *
 *   LLM path:
 *     Tries AnalystAPI.chat() first — Gemini gets full claim intelligence.
 *     Falls back silently to rule engine on any failure.
 */

const Chat = (() => {

  // ── State ────────────────────────────────────────────────────────────
  let _payload       = null;
  let _result        = null;
  let _contributions = [];  // aggregated claim-level SHAP from SHAP.fromApiResult()
  let _sessionId     = null;
  let _mode          = 'INACTIVE';

  // ── DOM helpers ──────────────────────────────────────────────────────
  const _el    = id => document.getElementById(id);
  const _msgs  = () => _el('chat-messages');
  const _input = () => _el('chat-input');
  const _btn   = () => _el('chat-send-btn');
  const _idle  = () => _el('chat-idle');
  const _badge = () => _el('chat-mode-badge');

  function _appendBubble(role, html) {
    const el = _msgs();
    if (!el) return null;
    _idle()?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = `chat-bubble ${role}`;
    div.innerHTML = html;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    return div;
  }

  function _appendThinking() {
    const el = _msgs();
    if (!el) return null;
    _idle()?.classList.add('hidden');
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
  const _wrap = t => `<p>${_md(t)}</p>`;

  function _updateBadge() {
    const el = _badge();
    if (!el) return;
    if (_mode === 'LLM') {
      el.textContent = 'Intelligence connected';
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

  // ── Shorthand accessors ──────────────────────────────────────────────
  function _cs()    { return _result?.claim_summary ?? {}; }
  function _preds() { return _result?.predictions   ?? []; }
  function _acts()  { return _payload?.activities   ?? []; }

  // Primary diagnosis code — new field name
  function _primaryDx() {
    return _payload?.primary_diagnosis_code ?? '?';
  }

  // All secondary diagnoses across all activities (deduplicated)
  function _secondaryDxList() {
    const seen = new Set();
    const codes = [];
    for (const act of _acts()) {
      for (const dx of (act.diagnoses || [])) {
        if (dx.diagnosis_type === 'secondary' && !seen.has(dx.diagnosis_code)) {
          seen.add(dx.diagnosis_code);
          codes.push(dx.diagnosis_code);
        }
      }
    }
    return codes;
  }

  // Highest-risk prediction object
  function _topPred() {
    return [..._preds()].sort((a, b) => b.denial_probability - a.denial_probability)[0];
  }

  // Find a prediction by activity code or 1-based number string
  function _findPred(token) {
    const preds = _preds();
    const num   = parseInt(token, 10);
    if (!isNaN(num) && num >= 1 && num <= preds.length) {
      return { pred: preds[num - 1], idx: num - 1 };
    }
    const idx = preds.findIndex(p =>
      p.activity_code.toLowerCase() === token.toLowerCase()
    );
    if (idx >= 0) return { pred: preds[idx], idx };
    return null;
  }

  // Format an activity's SHAP drivers as a readable list
  function _formatDrivers(pred, maxDrivers = 5) {
    const drivers = (pred.top_drivers || []).slice(0, maxDrivers);
    if (drivers.length === 0) return 'No SHAP driver data available.';
    return drivers.map((d, i) => {
      const dir  = d.impact === 'increase_risk' ? '↑ pushes toward denial' : '↓ pushes toward approval';
      const name = SHAP.DISPLAY_NAMES[d.feature] || d.feature;
      const sign = d.shap_value >= 0 ? '+' : '';
      return `${i + 1}. **${name}** — SHAP ${sign}${d.shap_value.toFixed(4)} · ${dir}`;
    }).join('\n');
  }

  // Full single-activity breakdown block
  function _activityBlock(pred, idx, actPayload) {
    const pct     = (pred.denial_probability * 100).toFixed(1);
    const verdict = pred.predicted_denial ? '⚠ LIKELY DENIED' : '✓ LIKELY APPROVED';
    const gross   = actPayload?.activity_gross != null
      ? `AED ${parseFloat(actPayload.activity_gross).toLocaleString('en', { minimumFractionDigits: 2 })}`
      : '—';
    const qty     = actPayload?.activity_quantity ?? '—';

    // Show secondary diagnoses for this activity
    const secDx = (actPayload?.diagnoses || [])
      .filter(d => d.diagnosis_type === 'secondary')
      .map(d => `\`${d.diagnosis_code}\``)
      .join(', ') || '—';

    const drivers = _formatDrivers(pred, 5);
    const rec     = pred.recommendation || 'No specific recommendation.';

    return `**Activity #${idx + 1}: \`${pred.activity_code}\`** (${pred.cpt_category || 'CPT'})

Denial probability: **${pct}%** — ${verdict}
Gross amount: ${gross} · Quantity: ${qty}
Secondary diagnoses: ${secDx}

**SHAP drivers (what the model saw):**
${drivers}

**Recommendation:** ${rec}`;
  }

  // ── Claim intelligence for LLM ───────────────────────────────────────
  function _buildCI() {
    const cs    = _cs();
    const preds = _preds();
    return {
      claim: {
        claim_id:               _result?.claim_id,
        primary_diagnosis_code: _primaryDx(),
        icd_category:           API.deriveIcdCategory(_primaryDx()),
        patient_age:            _payload?.patient_age,
        gender:                 _payload?.gender,
        nationality:            _payload?.nationality,
        encounter_type:         _payload?.encounter_type,
        length_of_stay:         _payload?.length_of_stay,
        claim_gross:            _payload?.claim_gross,
        claim_net:              _payload?.claim_net,
        clinician_profession:   _payload?.clinician_profession,
        facility_type:          _payload?.facility_type,
        payer_classification:   _payload?.payer_classification,
        activities:             _acts(),
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
          top_drivers:        p.top_drivers,
        })),
      },
      top_shap_contributions: _contributions.slice(0, 10).map(c => ({
        feature:   c.name,
        shap_value: c.value,
        direction: c.value >= 0 ? 'increases_denial' : 'reduces_denial',
      })),
    };
  }

  // ── Rule-based engine ────────────────────────────────────────────────
  function _rule(msg) {
    const m     = msg.toLowerCase();
    const cs    = _cs();
    const preds = _preds();
    const acts  = _acts();
    const diag  = _primaryDx();
    const prob  = cs.claim_denial_probability_pct ?? 0;
    const level = (cs.claim_risk_level ?? '?').toUpperCase();
    const los   = _payload?.length_of_stay   ?? 0;
    const nat   = (_payload?.nationality     ?? 'UNKNOWN').toUpperCase();
    const enc   = _payload?.encounter_type   ?? 'OP';
    const top   = _contributions[0] ?? { name: 'unknown', value: 0 };
    const topP  = _topPred();

    // ── 1. Named activity deep-dive: "activity 99213" or "activity 2" ──
    const actMatch = m.match(/\b(?:activity|act|line|cpt)\s+([a-z0-9]+)\b/i);
    if (actMatch) {
      const token = actMatch[1];
      const found = _findPred(token);
      if (found) {
        const { pred, idx } = found;
        return _activityBlock(pred, idx, acts[idx]);
      }
      return `No activity matching **"${token}"** found on this claim.\n\n` +
        `Available activities: ${preds.map((p, i) => `#${i + 1} \`${p.activity_code}\``).join(', ')}.`;
    }

    // ── 2. "highest risk" / "worst" / "most likely denied" ─────────────
    if (/\b(highest|worst|most.?(dangerous|risk|denial|denied)|top.?activit)\b/.test(m)) {
      if (!topP) return 'No predictions available yet.';
      const idx = preds.indexOf(topP);
      return _activityBlock(topP, idx, acts[idx]) +
        `\n\nThis is the highest-risk activity on the claim. Its probability of **` +
        `${(topP.denial_probability * 100).toFixed(1)}%** is the primary driver of the ` +
        `overall claim score of **${prob.toFixed(1)}%**.`;
    }

    // ── 3. "compare activities" / "all activities" ──────────────────────
    if (/\b(compare|all.?activit|each activit|list activit|every activit)\b/.test(m)) {
      if (preds.length === 0) return 'No activity predictions available.';
      const rows = preds.map((p, i) => {
        const pct  = (p.denial_probability * 100).toFixed(1);
        const vrd  = p.predicted_denial ? '⚠ Denied' : '✓ Approved';
        const top3 = (p.top_drivers || []).slice(0, 3)
          .map(d => {
            const n = SHAP.DISPLAY_NAMES[d.feature] || d.feature;
            const s = d.shap_value >= 0 ? '+' : '';
            return `${n} (${s}${d.shap_value.toFixed(3)})`;
          }).join(', ') || '—';
        return `- **#${i + 1} \`${p.activity_code}\`** (${p.cpt_category || 'CPT'}): **${pct}%** ${vrd}\n  Drivers: ${top3}`;
      }).join('\n\n');
      return `**All ${preds.length} activit${preds.length === 1 ? 'y' : 'ies'} on this claim:**\n\n${rows}\n\n` +
        `Overall claim score: **${prob.toFixed(1)}%** (${level}) = 0.7 × max + 0.3 × avg.`;
    }

    // ── 4. "shap drivers" / "shap analysis" / "feature breakdown" ───────
    if (/\b(shap.?driver|shap.?analys|feature.?break|feature.?import|shap.?value)\b/.test(m)) {
      if (preds.length === 0) return 'No SHAP data available.';
      const sections = preds.map((p, i) => {
        const pct     = (p.denial_probability * 100).toFixed(1);
        const drivers = _formatDrivers(p, 5);
        return `**Activity #${i + 1} \`${p.activity_code}\`** — ${pct}%\n${drivers}`;
      }).join('\n\n─────\n\n');
      return `**SHAP drivers by activity:**\n\n${sections}\n\n` +
        `Positive SHAP = feature pushes toward denial. Negative = pushes toward approval. ` +
        `Magnitude = strength of that push.`;
    }

    // ── 5. "fix" / "recommend" / "appeal" ───────────────────────────────
    if (/\b(fix|resolv|appeal|action|recommend|improv|reduc|prevent)\b/.test(m)) {
      const fixMatch = m.match(/\b(?:fix|resolv|appeal)\s+(?:activity\s+)?([a-z0-9]+)\b/i);
      if (fixMatch) {
        const found = _findPred(fixMatch[1]);
        if (found) {
          const { pred, idx } = found;
          const topD  = pred.top_drivers?.[0];
          const fname = topD ? (SHAP.DISPLAY_NAMES[topD.feature] || topD.feature) : 'unknown';
          const fval  = topD ? `${topD.shap_value >= 0 ? '+' : ''}${topD.shap_value.toFixed(4)}` : '—';
          return `**Remediation for Activity #${idx + 1} \`${pred.activity_code}\`** (${pred.cpt_category || 'CPT'}):\n\n` +
            `Current denial probability: **${(pred.denial_probability * 100).toFixed(1)}%**\n\n` +
            `Top driver to address: **${fname}** (SHAP ${fval})\n\n` +
            `${pred.recommendation || 'Review claim documentation before resubmission.'}\n\n` +
            `Steps:\n` +
            `- Verify primary ICD \`${diag}\` is clinically consistent with CPT \`${pred.activity_code}\`.\n` +
            `- Confirm CPT category **${pred.cpt_category || 'Unknown'}** matches the service rendered.\n` +
            `- Confirm encounter type \`${enc}\` is correct for this service.\n` +
            `- Ensure all secondary diagnoses are accurate and documented.`;
        }
      }

      const denied = preds.filter(p => p.predicted_denial);
      if (denied.length === 0) {
        return `No activities on this claim are predicted to be denied. ` +
          `Overall claim score is **${prob.toFixed(1)}%** (${level}) — low action required.`;
      }
      const steps = denied.map((p, i) => {
        const topD  = p.top_drivers?.[0];
        const fname = topD ? (SHAP.DISPLAY_NAMES[topD.feature] || topD.feature) : '—';
        return `**${i + 1}. \`${p.activity_code}\`** (${(p.denial_probability * 100).toFixed(1)}%): ` +
          `${p.recommendation || 'Review documentation.'}\n   Top driver: **${fname}**`;
      }).join('\n\n');
      const revised = Math.max(prob - Math.abs(top.value) * 30, 5).toFixed(0);
      return `**${denied.length} of ${preds.length} activit${preds.length === 1 ? 'y' : 'ies'} predicted denied.` +
        `** Estimated revised claim score if top drivers resolved: **~${revised}%**.\n\n` +
        `${steps}\n\n` +
        `Common to all: Verify primary ICD \`${diag}\` (${API.deriveIcdCategory(diag)}) ` +
        `aligns with all CPT codes. Confirm encounter type \`${enc}\` matches actual setting.`;
    }

    // ── 6. "why" / "reason" / "driver" ──────────────────────────────────
    if (/\b(why|reason|driver|cause|explain|factor|what.?push)\b/.test(m)) {
      const ranked = [...preds]
        .sort((a, b) => b.denial_probability - a.denial_probability)
        .map((p, rank) => {
          const pct   = (p.denial_probability * 100).toFixed(1);
          const topD  = p.top_drivers?.[0];
          const fname = topD ? (SHAP.DISPLAY_NAMES[topD.feature] || topD.feature) : '—';
          const fval  = topD ? `${topD.shap_value >= 0 ? '+' : ''}${topD.shap_value.toFixed(3)}` : '—';
          return `${rank + 1}. \`${p.activity_code}\` **${pct}%** — top driver: **${fname}** (${fval})`;
        }).join('\n');

      const baseVal  = SHAP.baseLogOdds(_result);
      const finalVal = SHAP.finalLogOdds(_result);
      const netShap  = parseFloat((finalVal - baseVal).toFixed(3));

      return `**Why is this claim flagged?**\n\n` +
        `Claim score: **${prob.toFixed(1)}%** (${level})\n` +
        `Formula: 0.7 × highest activity (${(cs.highest_activity_risk * 100).toFixed(1)}%) ` +
        `+ 0.3 × average (${(cs.average_activity_risk * 100).toFixed(1)}%)\n` +
        `Net SHAP shift from base: **${netShap >= 0 ? '+' : ''}${netShap}**\n\n` +
        `**Activities ranked by denial probability:**\n${ranked}\n\n` +
        `Overall top SHAP driver across all activities: **${top.name}** ` +
        `(${top.value >= 0 ? '+' : ''}${top.value.toFixed(3)}) — ` +
        `${top.value > 0 ? 'pushes toward denial' : 'pushes toward approval'}.`;
    }

    // ── 7. Individual SHAP feature explanation ───────────────────────────
    const featureKeywords = Object.entries(SHAP.DISPLAY_NAMES).map(([k, v]) => ({ k, v }));
    const featMatch = featureKeywords.find(({ k, v }) =>
      m.includes(k.replace(/_/g, ' ')) || m.includes(v.toLowerCase())
    );
    if (featMatch) {
      const fname = featMatch.v;
      const fkey  = featMatch.k;
      const occurrences = preds.flatMap((p, i) =>
        (p.top_drivers || [])
          .filter(d => d.feature === fkey)
          .map(d => ({ actCode: p.activity_code, actIdx: i, d }))
      );
      if (occurrences.length > 0) {
        const lines = occurrences.map(({ actCode, d }) => {
          const dir  = d.impact === 'increase_risk' ? '↑ pushes toward denial' : '↓ pushes toward approval';
          const sign = d.shap_value >= 0 ? '+' : '';
          return `- \`${actCode}\`: SHAP **${sign}${d.shap_value.toFixed(4)}** · ${dir}`;
        }).join('\n');
        return `**${fname}** — Feature explanation\n\n${_featureDesc(fkey, diag)}\n\n` +
          `**Impact on this claim:**\n${lines}`;
      }
      return `**${fname}** is one of the model features. ` +
        `It did not appear in the top SHAP drivers for any activity on this claim, ` +
        `meaning its contribution was below the top-K threshold for all activities.`;
    }

    // ── 8. ICD / diagnosis ───────────────────────────────────────────────
    if (/\b(icd|diagnos|dx|secondary)\b/.test(m)) {
      const icdCat  = API.deriveIcdCategory(diag);
      const cptCats = [...new Set(preds.map(p => p.cpt_category).filter(Boolean))].join(', ');
      const secDx   = _secondaryDxList();
      return `**Primary ICD: \`${diag}\`** → Category: **${icdCat}**\n\n` +
        (secDx.length
          ? `Secondary diagnoses across activities: **${secDx.join(', ')}**\n\n`
          : 'No secondary diagnoses submitted.\n\n') +
        `CPT categories on this claim: **${cptCats || '—'}**\n\n` +
        `Secondary diagnoses are encoded as binary \`secondary_*\` features in the model ` +
        `(one column per ICD category). A higher \`secondary_dx_count\` and certain ` +
        `combinations of secondary categories can increase denial likelihood.\n\n` +
        `Check the **ICD Drill-Down** tab for population-level denial rates for \`${icdCat}\`.`;
    }

    // ── 9. LOS ───────────────────────────────────────────────────────────
    if (/\b(los|length.of.stay|stay)\b/.test(m)) {
      const losFeature = _contributions.find(c =>
        c.feature === 'length_of_stay' || c.name.toLowerCase().includes('stay')
      );
      const losShap = losFeature?.value ?? 0;
      const ctx = los <= 5
        ? `${los} days is within the typical 0–5 day cluster — near-zero SHAP effect for most activities.`
        : `${los} days is an outlier — extended stays are flagged as elevated denial likelihood (SHAP ≈ +${losShap.toFixed(3)}).`;
      return `**Length of Stay: ${los} day${los !== 1 ? 's' : ''}**\n\n${ctx}\n\n` +
        `Aggregated LOS SHAP across activities: **${losShap >= 0 ? '+' : ''}${losShap.toFixed(4)}**\n\n` +
        `LOS effect is amplified for inpatient (IP) claims and when the primary diagnosis ` +
        `category does not typically require extended hospitalisation.`;
    }

    // ── 10. Financials ───────────────────────────────────────────────────
    if (/\b(amount|gross|net|cost|financial|aed|money)\b/.test(m)) {
      const fmt = v => parseFloat(v || 0).toLocaleString('en', { minimumFractionDigits: 2 });
      const actBreakdown = acts.map((a, i) => {
        const pred = preds[i];
        const pct  = pred ? `${(pred.denial_probability * 100).toFixed(1)}%` : '—';
        return `- \`${a.activity_code}\`: AED ${fmt(a.activity_gross)} × ${a.activity_quantity} → ${pct}`;
      }).join('\n');
      return `**Claim Financials**\n\n` +
        `Claim gross: **AED ${fmt(_payload?.claim_gross)}**\n` +
        `Claim net:   **AED ${fmt(_payload?.claim_net)}**\n\n` +
        `**Activity breakdown:**\n${actBreakdown}\n\n` +
        `Higher claim amounts and activity gross values push the model toward denial. ` +
        `\`activity_gross\` is consistently one of the highest-gain features in the model.`;
    }

    // ── 11. Overall score / probability ─────────────────────────────────
    if (/\b(score|probabilit|likelihood|percent|risk|band|high|medium|low)\b/.test(m)) {
      return `**Claim Denial Score: ${prob.toFixed(1)}%** — Risk Level: **${level}**\n\n` +
        `**Scoring formula (predictionapi.py):**\n` +
        `Claim score = **0.7 × max activity** + **0.3 × average activity**\n\n` +
        `- Highest activity: **${(cs.highest_activity_risk * 100).toFixed(1)}%** ` +
        `(\`${topP?.activity_code ?? '—'}\`)\n` +
        `- Average activity: **${(cs.average_activity_risk * 100).toFixed(1)}%**\n` +
        `- Activities evaluated: **${cs.activity_count}**\n\n` +
        `Likelihood bands: **LOW < 40%** · **MEDIUM 40–70%** · **HIGH ≥ 70%**`;
    }

    // ── 12. Interaction / combination ────────────────────────────────────
    if (/\b(interact|combination|pair|together|icd.?cpt)\b/.test(m)) {
      const cptCats = [...new Set(preds.map(p => p.cpt_category).filter(Boolean))];
      return `**ICD–CPT Interaction Analysis**\n\n` +
        `Primary ICD: \`${diag}\` (${API.deriveIcdCategory(diag)}) × ` +
        `CPT categories: **${cptCats.join(', ') || '—'}**\n\n` +
        `The model encodes this relationship through the \`primary_diagnosis_category\` and ` +
        `\`cpt_category\` features. Mismatches between the diagnosis domain and the procedure ` +
        `category are one of the strongest denial signals.\n\n` +
        `See the **Interactions** tab for the full heatmap of denial rates across ` +
        `ICD × CPT category combinations from the training population.`;
    }

    // ── 13. Nationality ──────────────────────────────────────────────────
    if (/\b(nationalit|nation|emirati|indian|expat)\b/.test(m)) {
      const natFeature = _contributions.find(c => c.feature === 'nationality');
      const natShap    = natFeature?.value ?? 0;
      return `**Nationality: \`${nat}\`**\n\n` +
        `SHAP contribution: **${natShap >= 0 ? '+' : ''}${natShap.toFixed(4)}** ` +
        `(${natShap > 0 ? 'pushes toward denial' : 'pushes toward approval'})\n\n` +
        `Nationality is a mild population-level signal and is rarely the primary denial driver.`;
    }

    // ── 14. Clinician / facility / payer ─────────────────────────────────
    if (/\b(clinician|doctor|physician|facility|hospital|clinic|payer|insurance)\b/.test(m)) {
      const clinFeat = _contributions.find(c => c.feature === 'clinician_profession');
      const facFeat  = _contributions.find(c => c.feature === 'facility_type');
      const payFeat  = _contributions.find(c => c.feature === 'payer_classification');
      const lines    = [
        clinFeat && `**Clinician** (${_payload?.clinician_profession}): SHAP ${clinFeat.value >= 0 ? '+' : ''}${clinFeat.value.toFixed(4)}`,
        facFeat  && `**Facility** (${_payload?.facility_type}): SHAP ${facFeat.value >= 0 ? '+' : ''}${facFeat.value.toFixed(4)}`,
        payFeat  && `**Payer** (${_payload?.payer_classification}): SHAP ${payFeat.value >= 0 ? '+' : ''}${payFeat.value.toFixed(4)}`,
      ].filter(Boolean).join('\n');
      return `**Provider & Payer SHAP signals:**\n\n${lines || 'No provider features in top SHAP drivers.'}\n\n` +
        `Specialty–procedure alignment is the key check. Mismatches between clinician ` +
        `profession and CPT codes raise denial likelihood.`;
    }

    // ── 15. Model info ───────────────────────────────────────────────────
    if (/\b(model|algorithm|xgboost|feature|train|accuracy|auc)\b/.test(m)) {
      return `**Model: XGBoost + Target Encoder**\n\n` +
        `Trained on UAE healthcare claims data. 38 features including:\n` +
        `\`activity_code\`, \`activity_quantity\`, \`activity_gross\`, \`patient_age\`, \`gender\`, ` +
        `\`nationality\`, \`claim_gross\`, \`claim_net\`, \`encounter_type\`, \`length_of_stay\`, ` +
        `\`clinician_profession\`, \`facility_type\`, \`payer_classification\`, ` +
        `\`primary_diagnosis_code\`, \`primary_diagnosis_category\`, ` +
        `\`secondary_dx_count\`, 21× \`secondary_*\` category flags, \`cpt_category\`.\n\n` +
        `**Top feature groups by impact:**\n` +
        `1. Activity-level: \`activity_gross\`, \`activity_code\`, \`activity_quantity\`\n` +
        `2. Claim-level: \`claim_net\`, \`claim_gross\`\n` +
        `3. Diagnosis: \`primary_diagnosis_code\`, \`primary_diagnosis_category\`, secondary flags\n` +
        `4. Provider: \`cpt_category\`, \`clinician_profession\`, \`facility_type\`, \`payer_classification\`\n\n` +
        `Claim score = **0.7 × max activity risk + 0.3 × average activity risk**.`;
    }

    // ── Default ──────────────────────────────────────────────────────────
    const actList = preds.map((p, i) =>
      `#${i + 1} \`${p.activity_code}\` ${(p.denial_probability * 100).toFixed(1)}%` +
      ` ${p.predicted_denial ? '⚠' : '✓'}`
    ).join(' · ');
    return `This claim scores **${prob.toFixed(1)}%** (${level}) across ` +
      `${preds.length} activit${preds.length === 1 ? 'y' : 'ies'}:\n${actList}\n\n` +
      `**Try asking:**\n` +
      `- *"activity 99213"* or *"activity 2"* — deep-dive any specific activity\n` +
      `- *"highest risk"* — full analysis of the most dangerous activity\n` +
      `- *"compare activities"* — side-by-side table of all activities\n` +
      `- *"shap drivers"* — SHAP breakdown for every activity\n` +
      `- *"why denied"* — ranked reasons with SHAP evidence\n` +
      `- *"fix 99213"* or *"fix all"* — specific remediation steps\n` +
      `- *"ICD code"*, *"secondary diagnoses"*, *"claim amount"* — feature deep-dives\n` +
      `- *"model info"* — algorithm, features, gain percentages`;
  }

  // ── Feature description lookup ───────────────────────────────────────
  function _featureDesc(fkey, primaryDx) {
    const descs = {
      activity_code:
        'The specific CPT/activity code submitted. Code selection directly influences the model — rare or mismatched codes increase denial likelihood.',
      activity_quantity:
        'Number of times the activity was performed. High quantities relative to the diagnosis can trigger elevated SHAP values.',
      activity_gross:
        "The billed gross amount for this activity. Higher amounts increase the model's denial prediction, especially for certain CPT categories.",
      patient_age:
        'Patient age is a demographic signal. Certain age groups have different denial rate patterns for specific procedures.',
      gender:
        'Gender interacts with CPT category — some procedures have gender-specific medical necessity requirements.',
      nationality:
        'Nationality is a mild population-level signal in the training data. Rarely the primary driver.',
      claim_gross:
        'Total gross amount of the entire claim. High claim values correlate with increased scrutiny from payers.',
      claim_net:
        'Net amount after discounts. Large divergence between gross and net can be a signal.',
      encounter_type:
        'OP/IP/EM classification. Mismatches between encounter type and procedures are flagged by the model.',
      length_of_stay:
        'For IP claims, extended stays increase denial likelihood. LOS of 0 is expected for OP.',
      clinician_profession:
        'The type of clinician. Specialty–procedure alignment is checked by the model.',
      facility_type:
        'Clinic vs Hospital vs Day Surgery. The model checks procedure–facility alignment.',
      payer_classification:
        'Payer identifier. Certain payers have historically higher denial rates for specific claim types.',
      primary_diagnosis_code:
        `The primary ICD-10 code (\`${primaryDx}\`). Drives primary_diagnosis_category and all secondary_* feature interactions.`,
      primary_diagnosis_category:
        'Broad clinical category derived from the primary ICD prefix. Interacts strongly with CPT category.',
      secondary_dx_count:
        'Total number of secondary diagnoses. More secondary diagnoses can increase complexity scoring.',
      cpt_category:
        'Procedure category. The primary diagnosis–CPT category pairing is one of the strongest signals in the model.',
    };
    // Handle secondary_* features generically
    if (fkey.startsWith('secondary_')) {
      const cat = fkey.replace('secondary_', '').replace(/_/g, ' ');
      return `Binary flag (0/1) indicating whether a secondary diagnosis in the **${cat}** ` +
        `category is present on any activity. The presence of certain secondary categories ` +
        `can significantly shift denial likelihood when combined with the primary diagnosis.`;
    }
    return descs[fkey] || `\`${fkey}\` is one of the 38 model features.`;
  }

  // ── LLM + fallback ───────────────────────────────────────────────────
  async function _respond(userMessage) {
    if (_mode === 'LLM' && _sessionId) {
      const r = await AnalystAPI.chat(_sessionId, userMessage);
      if (r.ok) return { text: r.answer, source: 'llm' };
      console.warn('[Chat] LLM fallback:', r.detail);
    }
    return { text: _rule(userMessage), source: 'fallback' };
  }

  // ── Public API ───────────────────────────────────────────────────────
  async function init(payload, result, contributions) {
    _payload       = payload;
    _result        = result;
    _contributions = contributions;
    _sessionId     = null;
    _mode          = 'RULE_BASED';

    const inp = _input(), btn = _btn();
    if (inp) { inp.disabled = false; inp.placeholder = 'Ask about this claim…'; }
    if (btn) btn.disabled = false;

    const el = _msgs();
    if (el) el.innerHTML = '';
    _idle()?.classList.add('hidden');

    const cs      = result.claim_summary ?? {};
    const preds   = result.predictions   ?? [];
    const pct     = cs.claim_denial_probability_pct ?? 0;
    const topPred = [...preds].sort((a, b) => b.denial_probability - a.denial_probability)[0];

    const top3 = contributions.slice(0, 3).map(c => {
      const sign = c.value >= 0 ? '+' : '';
      const dir  = c.value >= 0 ? '↑ denial' : '↓ approval';
      return `**${c.name}** (${sign}${c.value.toFixed(3)}, ${dir})`;
    }).join(', ');

    const deniedCount = preds.filter(p => p.predicted_denial).length;
    const actSummary  = preds.map((p, i) =>
      `#${i + 1} \`${p.activity_code}\` **${(p.denial_probability * 100).toFixed(1)}%** ` +
      `${p.predicted_denial ? '⚠' : '✓'}`
    ).join(' · ');

    // Show secondary dx count in greeting if any were submitted
    const secDx   = _secondaryDxList();
    const secLine = secDx.length
      ? `Secondary diagnoses submitted: **${secDx.join(', ')}**.\n\n`
      : '';

    const greeting =
      `Claim analysed — **${pct.toFixed(1)}% denial score** (${cs.claim_risk_level ?? '?'}).\n\n` +
      `**${preds.length} activit${preds.length === 1 ? 'y' : 'ies'} evaluated** — ` +
      `**${deniedCount} predicted denied**:\n${actSummary}\n\n` +
      (topPred
        ? `Highest-risk: \`${topPred.activity_code}\` at ` +
          `**${(topPred.denial_probability * 100).toFixed(1)}%**.\n\n`
        : '') +
      secLine +
      `Top SHAP drivers (claim-level): ${top3 || '—'}.\n\n` +
      `Ask me about any specific activity, secondary diagnosis, feature, or remediation step.`;

    _appendBubble('assistant', _wrap(greeting));
    _initLLM(payload, result, contributions);
  }

  async function _initLLM(payload, result, contributions) {
    const sr = await AnalystAPI.createSession(_buildCI());
    if (sr.ok) { _sessionId = sr.session_id; _mode = 'LLM'; }
    else { _mode = 'RULE_BASED'; console.warn('[Chat] AI Analyst unavailable:', sr.detail); }
    _updateBadge();
  }

  async function send(userMessage) {
    if (!userMessage.trim() || !_result) return;
    const inp = _input(), btn = _btn();
    if (inp) inp.disabled = true;
    if (btn) btn.disabled = true;

    _appendBubble('user', _wrap(userMessage));
    const thinking = _appendThinking();
    const { text, source } = await _respond(userMessage);

    if (thinking && thinking.parentNode) {
      thinking.className = 'chat-bubble assistant';
      let html = _wrap(text);
      if (source === 'fallback' && _mode === 'LLM') {
        html += `<p class="chat-fallback-notice">⚡ Local response — AI Analyst temporarily unavailable.</p>`;
      }
      thinking.innerHTML = html;
      _msgs().scrollTop = _msgs().scrollHeight;
    }
    if (inp) { inp.disabled = false; inp.focus(); }
    if (btn) btn.disabled = false;
  }

  function clear() {
    if (_sessionId) AnalystAPI.deleteSession(_sessionId);
    _payload = null; _result = null; _contributions = [];
    _sessionId = null; _mode = 'INACTIVE';
    const el = _msgs();
    if (el) el.innerHTML = '';
    _idle()?.classList.remove('hidden');
    const inp = _input(), btn = _btn();
    if (inp) { inp.disabled = true; inp.placeholder = 'Submit a claim to activate…'; }
    if (btn) btn.disabled = true;
    _updateBadge();
  }

  return { init, send, clear };
})();
