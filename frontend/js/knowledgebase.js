/**
 * js/knowledgebase.js
 * --------------------
 * Global RAG — Knowledge Base tab.
 *
 * COMPLETELY ISOLATED from the AI analyst session flow.
 * Does not read or write _payload, _result, _sessionId.
 * Does not call Chat, SHAP, Charts, or any other module.
 * Only calls AnalystAPI.globalRag.search() and .browse().
 */

const KnowledgeBase = (() => {

  let _page        = 1;
  let _hasMore     = true;
  let _searchMode  = false;
  let _lastQuery   = '';
  let _debounce    = null;

  const SEVERITY_COLORS = {
    CRITICAL: { bg: 'var(--color-high-bg)',   border: 'var(--color-high-border)',   text: 'var(--color-high)'   },
    SEVERE:   { bg: 'var(--color-high-bg)',   border: 'var(--color-high-border)',   text: 'var(--color-high)'   },
    WARNING:  { bg: 'var(--color-medium-bg)', border: 'var(--color-medium-border)', text: 'var(--color-medium)' },
    INFO:     { bg: 'var(--color-low-bg)',    border: 'var(--color-low-border)',    text: 'var(--color-low)'    },
  };

  // ── Render a single rule card ──────────────────────────────────────
  function _renderCard(rule) {
    const sev     = (rule.severity || 'INFO').toUpperCase();
    const sevCol  = SEVERITY_COLORS[sev] || SEVERITY_COLORS.INFO;
    const sim     = rule.similarity != null
      ? `<span style="font-size:.68rem;color:var(--color-text-muted);margin-left:.5rem;">
           ${(rule.similarity * 100).toFixed(0)}% match
         </span>`
      : '';

    const longMsg = (rule.long_message || '').trim();
    const hasLong = longMsg && longMsg !== rule.short_message;
    const cardId  = `kb-card-${rule.rule_id.replace(/[^a-zA-Z0-9]/g, '_')}`;

    return `
      <div class="chart-card" style="padding:1rem 1.25rem;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
          <div style="flex:1;min-width:0;">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem;flex-wrap:wrap;">
              <code style="font-size:.78rem;font-weight:700;color:var(--color-text-primary);">${rule.rule_id}</code>
              ${sim}
            </div>
            <div style="font-size:.88rem;font-weight:600;color:var(--color-text-primary);margin-bottom:.3rem;">
              ${rule.short_message || '—'}
            </div>
            <div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:${hasLong ? '.6rem' : '0'};">
              <span style="
                font-size:.68rem;font-weight:600;padding:.12rem .45rem;border-radius:4px;
                background:${sevCol.bg};color:${sevCol.text};border:1px solid ${sevCol.border};">
                ${sev}
              </span>
              <span style="font-size:.68rem;color:var(--color-text-muted);padding:.12rem .45rem;
                           background:var(--color-surface-alt);border:1px solid var(--color-border);border-radius:4px;">
                ${rule.package_name || '—'}
              </span>
              <span style="font-size:.68rem;color:var(--color-text-muted);padding:.12rem .45rem;
                           background:var(--color-surface-alt);border:1px solid var(--color-border);border-radius:4px;">
                ${rule.object_type || '—'}
              </span>
            </div>
            ${hasLong ? `
              <div id="${cardId}-long" style="display:none;font-size:.8rem;color:var(--color-text-secondary);
                           line-height:1.6;padding:.6rem .75rem;background:var(--color-chart-bg);
                           border-radius:var(--radius-sm);border:1px solid var(--color-border-light);">
                ${longMsg}
              </div>
              <button onclick="KnowledgeBase.toggleDetail('${cardId}')"
                style="margin-top:.5rem;background:none;border:none;cursor:pointer;
                       font-size:.75rem;color:var(--color-accent);font-family:var(--font);padding:0;"
                id="${cardId}-btn">
                ▼ View full rule
              </button>
            ` : ''}
          </div>
        </div>
      </div>`;
  }

  // ── Toggle long message ───────────────────────────────────────────
  function toggleDetail(cardId) {
    const el  = document.getElementById(`${cardId}-long`);
    const btn = document.getElementById(`${cardId}-btn`);
    if (!el) return;
    const open = el.style.display === 'block';
    el.style.display  = open ? 'none' : 'block';
    if (btn) btn.textContent = open ? '▼ View full rule' : '▲ Hide rule';
  }

  // ── Render results list ───────────────────────────────────────────
  function _renderResults(rules, append = false) {
    const container = document.getElementById('kb-results');
    if (!container) return;

    if (!rules || rules.length === 0) {
      if (!append) {
        container.innerHTML = `<div style="text-align:center;padding:2.5rem;color:var(--color-text-muted);font-size:.88rem;">
          No rules found matching your query.
        </div>`;
      }
      return;
    }

    const html = rules.map(_renderCard).join('');
    if (append) {
      container.insertAdjacentHTML('beforeend', html);
    } else {
      container.innerHTML = html;
    }
  }

  // ── Show/hide load more button ────────────────────────────────────
  function _setLoadMore(show) {
    const btn = document.getElementById('kb-load-more');
    if (btn) btn.style.display = show ? 'inline-block' : 'none';
  }

  // ── Loading state ─────────────────────────────────────────────────
  function _setLoading(loading) {
    const container = document.getElementById('kb-results');
    if (!container || !loading) return;
    container.innerHTML = `
      <div style="text-align:center;padding:3rem;color:var(--color-text-muted);">
        <div style="margin-bottom:.5rem;">Searching knowledge base…</div>
      </div>`;
  }

  // ── Update stats label ────────────────────────────────────────────
  function _updateLabel(text) {
    const el = document.getElementById('kb-stats-label');
    if (el) el.textContent = text;
  }

  // ── Search ───────────────────────────────────────────────────────
  async function search(query) {
    const region = window.regionManager?.getRegionCode() || 'uae';
    _searchMode = true;
    _lastQuery  = query;
    _setLoadMore(false);
    _setLoading(true);

    const data = await AnalystAPI.globalRag.search(query, 10, region);

    if (!data) {
      _renderResults([]);
      _updateLabel('Search failed — is the AI Analyst service running?');
      return;
    }

    _updateLabel(
      `${data.results.length} result${data.results.length !== 1 ? 's' : ''} ` +
      `for "${query}" — ${data.total_indexed} rules indexed (${region.toUpperCase()})`
    );

    _renderResults(data.results);
  }

  // ── Browse ───────────────────────────────────────────────────────
  async function browse(append = false) {
    const region   = window.regionManager?.getRegionCode() || 'uae';
    const severity = document.getElementById('kb-severity-filter')?.value || '';
    _searchMode    = false;

    if (!append) {
      _page = 1;
      _setLoading(true);
    }

    const data = await AnalystAPI.globalRag.browse(region, severity, _page, 15);

    if (!data) {
      _renderResults([]);
      _updateLabel('Could not load rules — is the AI Analyst service running?');
      _setLoadMore(false);
      return;
    }

    const totalLoaded = (_page - 1) * 15 + data.rules.length;
    _hasMore = totalLoaded < data.total;

    _updateLabel(
      `${data.total} rules indexed (${region.toUpperCase()})` +
      (severity ? ` · Filtered: ${severity}` : '')
    );

    _renderResults(data.rules, append);
    _setLoadMore(_hasMore);
  }

  // ── Init — called when tab is first opened ───────────────────────
  async function init() {
    // Load all rules on first open
    await browse(false);

    // Search box — debounced
    const input = document.getElementById('kb-search-input');
    if (input) {
      input.addEventListener('input', () => {
        clearTimeout(_debounce);
        const q = input.value.trim();
        if (q.length === 0) {
          // Clear search → back to browse mode
          _searchMode = false;
          browse(false);
          return;
        }
        if (q.length < 3) return;
        _debounce = setTimeout(() => search(q), 400);
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          clearTimeout(_debounce);
          const q = input.value.trim();
          if (q) search(q);
        }
      });
    }

    // Severity filter
    const filter = document.getElementById('kb-severity-filter');
    if (filter) {
      filter.addEventListener('change', () => {
        const input = document.getElementById('kb-search-input');
        if (input && input.value.trim()) {
          search(input.value.trim());
        } else {
          browse(false);
        }
      });
    }

    // Load more
    const loadMoreBtn = document.getElementById('kb-load-more');
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener('click', () => {
        _page++;
        browse(true);
      });
    }
  }

  let _initialised = false;

  function maybeInit() {
    if (!_initialised) {
      _initialised = true;
      init();
    }
  }

  return { maybeInit, toggleDetail };
})();
