from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalystSessionRequest,
    AnalystSessionResponse,
    AnalystChatRequest,
    AnalystChatResponse,
)

from .session_manager import SessionManager
from .analyst_service import AnalystService
from .llm_reasoner import LLMReasoner
from .db.database import init_db


# ============================================================
# GLOBAL SERVICES
# ============================================================

session_manager = SessionManager()
llm_reasoner    = None
analyst_service = None
retriever       = None      # RAG retriever — None until startup succeeds


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global llm_reasoner, analyst_service, retriever

    print("\n" + "=" * 80)
    print("STARTING AI ANALYST")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. PostgreSQL — core tables
    # --------------------------------------------------------

    try:
        init_db()
        print(" PostgreSQL initialized successfully.")
    except Exception as exc:
        print("❌ PostgreSQL initialization failed.")
        print(str(exc))
        raise

    # --------------------------------------------------------
    # 2. ChromaDB RAG store — thynk_rules collection
    #    Non-fatal: if chromadb is not installed the
    #    analyst still works without RAG.
    # --------------------------------------------------------

    try:
        from .rag.rule_store import init_rule_store, count_rules
        init_rule_store("uae")
        init_rule_store("ksa")
        uae_count = count_rules("uae")
        ksa_count = count_rules("ksa")
        rule_count = uae_count + ksa_count
        print(f" RAG rule store ready — UAE: {uae_count} rules, KSA: {ksa_count} rules.")
    except Exception as exc:
        print(f"⚠️  RAG rule store init failed (RAG disabled): {exc}")
        rule_count = 0

    # --------------------------------------------------------
    # 3. RAG Retriever
    #    Only initialised when there are rules in the store.
    #    If the table is empty, RAG stays disabled until
    #    POST /analyst/ingest-rules is called.
    # --------------------------------------------------------

    if rule_count > 0:
        try:
            from .rag.retriever import Retriever
            retriever = Retriever(region="uae")
            print(" RAG Retriever initialised.")
        except Exception as exc:
            print(f"⚠️  RAG Retriever init failed (RAG disabled): {exc}")
            retriever = None
    else:
        print(
            "ℹ️  RAG rule store is empty — RAG disabled. "
            "Call POST /analyst/ingest-rules to index rules."
        )

    # --------------------------------------------------------
    # 4. LLM Reasoner + Analyst Service
    # --------------------------------------------------------

    try:
        llm_reasoner = LLMReasoner()

        analyst_service = AnalystService(
            session_manager=session_manager,
            llm_reasoner=llm_reasoner,
            retriever=retriever,        # None → RAG silently skipped per turn
        )

        rag_status = "enabled" if retriever is not None else "disabled"
        print(f" AI Analyst initialised successfully (RAG {rag_status}).")

    except Exception as exc:
        print("\n❌ AI Analyst initialization failed:")
        print(str(exc))
        raise

    print("=" * 80 + "\n")

    yield


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Healthcare Claim AI Analyst",
    description=(
        "Conversational AI Analyst for explaining "
        "XGBoost claim denial predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LIVE STATS — KPI STRIP
# ============================================================
#
# Aggregates across analyst_claim_intelligence.evidence JSON.
# All queries are read-only and use raw SQL for performance.
# Returns null-safe results so the frontend can always render.
# ============================================================

@app.get("/stats/summary")
def stats_summary():
    """
    Returns 4 KPI cards for the live dashboard strip:
      1. total_claims     — total claims processed (+ today count)
      2. avg_denial_pct   — average denial probability across all claims
      3. top_risk_factor  — most frequently occurring top SHAP feature
      4. riskiest_payer   — payer with highest avg denial probability
    """
    from .db.database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        # ── 1. Total claims + today count ─────────────────────
        total_row = db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE created_at >= CURRENT_DATE
                ) AS today
            FROM analyst_claim_intelligence
        """)).fetchone()

        total_claims = int(total_row.total or 0)
        today_claims = int(total_row.today or 0)

        # ── 2. Average denial probability ─────────────────────
        avg_row = db.execute(text("""
            SELECT
                ROUND(AVG(
                    (evidence->'prediction'->>'claim_denial_probability_pct')::numeric
                ), 1) AS avg_pct,
                ROUND(AVG(
                    (evidence->'prediction'->>'claim_denial_probability_pct')::numeric
                ) FILTER (
                    WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                    AND created_at < CURRENT_DATE
                ), 1) AS last_week_pct
            FROM analyst_claim_intelligence
            WHERE evidence->'prediction'->>'claim_denial_probability_pct' IS NOT NULL
        """)).fetchone()

        avg_denial_pct     = float(avg_row.avg_pct or 0)
        last_week_avg      = float(avg_row.last_week_pct or 0) if avg_row.last_week_pct else None
        avg_delta          = round(avg_denial_pct - last_week_avg, 1) if last_week_avg else None

        # ── 3. Top risk factor (most frequent top SHAP feature) ─
        factor_row = db.execute(text("""
            SELECT
                shap_elem->>'feature' AS feature,
                COUNT(*) AS frequency
            FROM analyst_claim_intelligence,
                 jsonb_array_elements((evidence->'top_shap_contributions')::jsonb) AS shap_elem
            WHERE (shap_elem->>'shap_value')::numeric > 0
            GROUP BY feature
            ORDER BY frequency DESC
            LIMIT 1
        """)).fetchone()

        top_factor       = factor_row.feature if factor_row else "—"
        top_factor_count = int(factor_row.frequency) if factor_row else 0

        # ── 4. Riskiest payer (highest avg denial score) ──────
        payer_row = db.execute(text("""
            SELECT
                COALESCE(
                    NULLIF(evidence->'claim'->>'payer_name', ''),
                    NULLIF(evidence->'claim'->>'payer_classification', ''),
                    'Unknown'
                ) AS payer_name,
                ROUND(AVG(
                    (evidence->'prediction'->>'claim_denial_probability_pct')::numeric
                ), 1) AS avg_denial_pct,
                COUNT(*) AS claim_count
            FROM analyst_claim_intelligence
            WHERE evidence->'prediction'->>'claim_denial_probability_pct' IS NOT NULL
            GROUP BY payer_name
            HAVING COUNT(*) >= 1
            ORDER BY avg_denial_pct DESC
            LIMIT 1
        """)).fetchone()

        riskiest_payer       = payer_row.payer_name if payer_row else "—"
        riskiest_payer_pct   = float(payer_row.avg_denial_pct) if payer_row else 0.0
        riskiest_payer_count = int(payer_row.claim_count) if payer_row else 0

        return {
            "total_claims":          total_claims,
            "today_claims":          today_claims,
            "avg_denial_pct":        avg_denial_pct,
            "avg_denial_delta":      avg_delta,
            "top_risk_factor":       top_factor,
            "top_risk_factor_count": top_factor_count,
            "riskiest_payer":        riskiest_payer,
            "riskiest_payer_pct":    riskiest_payer_pct,
            "riskiest_payer_count":  riskiest_payer_count,
            "trend":                 [],   # populated separately via /stats/trend
        }

    except Exception as exc:
        print(f"[Stats] Query failed: {exc}")
        return {
            "total_claims": 0, "today_claims": 0,
            "avg_denial_pct": 0.0, "avg_denial_delta": None,
            "top_risk_factor": "—", "top_risk_factor_count": 0,
            "riskiest_payer": "—", "riskiest_payer_pct": 0.0,
            "riskiest_payer_count": 0, "trend": [],
        }
    finally:
        db.close()


@app.get("/stats/trend")
def stats_trend():
    """
    Returns 14-day daily average denial probability for the sparkline.
    """
    from .db.database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                DATE(created_at)::text AS day,
                ROUND(AVG(
                    (evidence->'prediction'->>'claim_denial_probability_pct')::numeric
                ), 1) AS avg_pct
            FROM analyst_claim_intelligence
            WHERE created_at >= CURRENT_DATE - INTERVAL '14 days'
              AND evidence->'prediction'->>'claim_denial_probability_pct' IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """)).fetchall()

        return [{"day": r.day, "avg_pct": float(r.avg_pct)} for r in rows]

    except Exception as exc:
        print(f"[Stats/Trend] Query failed: {exc}")
        return []
    finally:
        db.close()


@app.get("/stats/model-metrics")
def stats_model_metrics():
    """
    Returns model performance metrics from the metrics.json file.
    Region-aware: reads from models/uae or models/ksa based on query param.
    """
    import json
    from fastapi import Query
    from pathlib import Path

    region = "uae"  # default — frontend passes ?region=ksa for KSA
    base   = Path(__file__).resolve().parent.parent

    # Try new region-aware path first, fall back to legacy
    paths_to_try = [
        base / "models" / region / "prediction_score" / "metrics.json",
        base / "models" / "xgboost" / "metrics.json",
    ]

    for path in paths_to_try:
        if path.exists():
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
                return {
                    "region":    region,
                    "roc_auc":   metrics.get("roc_auc"),
                    "pr_auc":    metrics.get("pr_auc"),
                    "accuracy":  metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall":    metrics.get("recall"),
                    "f1":        metrics.get("f1"),
                    "train_rows": metrics.get("train_rows"),
                    "test_rows":  metrics.get("test_rows"),
                }
            except Exception:
                pass

    return {"region": region, "roc_auc": None, "pr_auc": None,
            "accuracy": None, "precision": None, "recall": None, "f1": None}


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    from .rag.rule_store import count_rules

    try:
        rule_count = count_rules()
    except Exception:
        rule_count = -1      # -1 signals ChromaDB unavailable

    return {
        "status":                "healthy",
        "analyst_service_loaded": analyst_service is not None,
        "llm_loaded":            llm_reasoner is not None,
        "rag_enabled":           retriever is not None,
        "rag_rules_indexed":     rule_count,
    }


# ============================================================
# CREATE ANALYST SESSION
# ============================================================

@app.post(
    "/analyst/session",
    response_model=AnalystSessionResponse,
)
def create_analyst_session(
    request: AnalystSessionRequest,
    x_region: str = Header(default="uae", alias="X-Region"),
):
    """
    Create a new analyst session with auto-generated initial summary.
    
    Headers:
        X-Region: Region code ('uae' or 'ksa') - defaults to 'uae'
    """
    region = x_region.lower()
    if region not in ("uae", "ksa"):
        region = "uae"

    if analyst_service is None:
        raise HTTPException(
            status_code=500,
            detail="AI Analyst service is not initialized.",
        )

    try:
        print("\n" + "=" * 80)
        print("CREATE ANALYST SESSION")
        print("=" * 80)
        print(f"Region: {region.upper()}")
        print("Received claim intelligence:")
        print(request.claim_intelligence)
        print("=" * 80)

        session_id, initial_summary = analyst_service.create_session(
            claim_intelligence=request.claim_intelligence,
            region=region,
            auto_summarize=True,  # Enable auto-summary
        )

        return AnalystSessionResponse(
            session_id=session_id,
            message="Analyst session created successfully.",
            initial_summary=initial_summary,
            rag_used=initial_summary is not None,
        )

    except Exception as exc:

        print("\n❌ SESSION CREATION ERROR")
        print(str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Failed to create analyst session: {str(exc)}",
        )


# ============================================================
# ANALYST CHAT
# ============================================================

@app.post(
    "/analyst/chat",
    response_model=AnalystChatResponse,
)
def analyst_chat(
    request: AnalystChatRequest,
    x_region: str = Header(default="uae", alias="X-Region"),
):
    """
    Chat with the AI analyst about a claim.
    
    Headers:
        X-Region: Region code ('uae' or 'ksa') - defaults to 'uae'
    """
    region = x_region.lower()
    if region not in ("uae", "ksa"):
        region = "uae"  # Fallback to UAE if invalid region

    if analyst_service is None:
        raise HTTPException(
            status_code=500,
            detail="AI Analyst service is not initialized.",
        )

    try:
        answer, rag_used = analyst_service.chat(
            session_id=request.session_id,
            user_message=request.message,
            use_rag=request.use_rag,
            region=region,  # Pass region to analyst service
        )

        return AnalystChatResponse(
            session_id=request.session_id,
            answer=answer,
            rag_used=rag_used,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        print("\n❌ ANALYST CHAT ERROR")
        print(str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Analyst processing failed: {str(exc)}",
        )


# ============================================================
# INGEST THYNK RULES INTO RAG STORE
# ============================================================
# POST /analyst/ingest-rules
#
# Triggers the full ingest pipeline:
#   1. Reads Thynk/thynk-claimlookup.json
#   2. Deduplicates rules by ruleId
#   3. Embeds each rule chunk with Gemini text-embedding-004
#   4. Upserts into ChromaDB persistent store
#   5. Re-initialises the Retriever so new rules are live
#
# Safe to call multiple times — ChromaDB upsert is idempotent.
# Expected runtime: ~5–15 seconds depending on rule count.
# ============================================================

@app.post("/analyst/ingest-rules")
def ingest_rules(
    region: str = "uae",
    x_region: str = Header(default=None, alias="X-Region"),
):

    global retriever, analyst_service

    target_region = (x_region or region).lower()
    if target_region not in ("uae", "ksa"):
        target_region = "uae"

    try:

        print("\n" + "=" * 80)
        print(f"INGEST RULES → RAG STORE ({target_region.upper()})")
        print("=" * 80)

        from .rag.ingest import run_ingest
        result = run_ingest(region=target_region)

        if result["status"] != "ok":
            raise HTTPException(
                status_code=500,
                detail=f"Ingest failed: {result.get('detail', 'unknown error')}",
            )

        # --------------------------------------------------------
        # Re-initialise Retriever with freshly indexed rules
        # and inject it into the live AnalystService instance.
        # --------------------------------------------------------

        from .rag.retriever import Retriever

        retriever = Retriever(region=target_region)

        if analyst_service is not None:
            analyst_service.retriever = retriever

        print(
            f" Ingest complete for {target_region.upper()} — {result['ingested']} rules upserted, "
            f"{result['total_in_store']} total in store. "
            f"Retriever re-initialised."
        )
        print("=" * 80 + "\n")

        return {
            "status":          "ok",
            "region":          target_region,
            "ingested":        result["ingested"],
            "total_in_store":  result["total_in_store"],
            "rag_enabled":     retriever is not None,
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(f"\n❌ INGEST ERROR: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {str(exc)}",
        )


# ============================================================
# GLOBAL RAG — SEARCH
# ============================================================
#
# Standalone endpoint for the Knowledge Base tab.
# Completely isolated from the AI analyst session flow.
# Uses only the existing Retriever — no changes to anything.
# ============================================================

@app.post("/analyst/global-rag/search")
def global_rag_search(
    request: dict,
    x_region: str = Header(default="uae", alias="X-Region"),
):
    """
    Search the rule knowledge base directly.
    Used by the Knowledge Base tab — independent of analyst sessions.

    Body: { "query": "medical necessity iron panel", "top_k": 10 }
    """
    query  = (request.get("query") or "").strip()
    top_k  = min(int(request.get("top_k", 10)), 20)
    region = x_region.lower()
    if region not in ("uae", "ksa"):
        region = "uae"

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        from .rag.retriever import Retriever
        from .rag.rule_store import count_rules

        r = Retriever(region=region)

        # Lower threshold for browse-style search
        raw_results = r.query(query_text=query, top_k=top_k, min_similarity=0.25)

        # Return structured results (not formatted string)
        from .rag.embedder import Embedder
        from .rag.rule_store import similarity_search

        embedder  = Embedder()
        query_vec = embedder.embed(query)
        results   = similarity_search(query_vec, region=region, top_k=top_k)

        return {
            "query":        query,
            "region":       region,
            "total_indexed": count_rules(region=region),
            "results": [
                {
                    "rule_id":       r["rule_id"],
                    "package_name":  r["package_name"],
                    "severity":      r["severity"],
                    "object_type":   r["object_type"],
                    "short_message": r["short_message"],
                    "long_message":  r["long_message"],
                    "similarity":    r["similarity"],
                }
                for r in results
                if r["similarity"] >= 0.25
            ],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analyst/global-rag/rules")
def global_rag_browse(
    x_region:  str = Header(default="uae", alias="X-Region"),
    severity:  str = "",
    package:   str = "",
    page:      int = 1,
    limit:     int = 20,
):
    """
    Browse all indexed rules with optional severity/package filter.
    Used by the Knowledge Base tab — independent of analyst sessions.
    """
    region = x_region.lower()
    if region not in ("uae", "ksa"):
        region = "uae"

    try:
        from .rag.rule_store import _get_collection

        col    = _get_collection(region)
        total  = col.count()
        offset = (page - 1) * limit

        # Fetch all with filters via ChromaDB where clause
        where = {}
        if severity:
            where["severity"] = severity.upper()
        if package:
            where["package_name"] = package

        fetch_kwargs = {
            "limit":   min(limit, 50),
            "offset":  offset,
            "include": ["metadatas", "documents"],
        }
        if where:
            fetch_kwargs["where"] = where

        results = col.get(**fetch_kwargs)

        rules = []
        for rule_id, meta, doc in zip(
            results["ids"], results["metadatas"], results["documents"]
        ):
            rules.append({
                "rule_id":       rule_id,
                "package_name":  meta.get("package_name", ""),
                "severity":      meta.get("severity", ""),
                "object_type":   meta.get("object_type", ""),
                "short_message": meta.get("short_message", ""),
                "long_message":  meta.get("long_message", ""),
            })

        return {
            "region":       region,
            "total":        total,
            "page":         page,
            "limit":        limit,
            "rules":        rules,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# DELETE SESSION
# ============================================================

@app.delete("/analyst/session/{session_id}")
def delete_session(
    session_id: str,
):

    try:

        deleted = session_manager.delete_session(session_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        return {
            "status":     "success",
            "message":    "Session deleted.",
            "session_id": session_id,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
