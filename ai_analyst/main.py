from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
        print("✅ PostgreSQL initialized successfully.")
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
        init_rule_store()
        rule_count = count_rules()
        print(f"✅ RAG rule store ready — {rule_count} rules indexed.")
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
            retriever = Retriever()
            print("✅ RAG Retriever initialised.")
        except Exception as exc:
            print(f"⚠️  RAG Retriever init failed (RAG disabled): {exc}")
            retriever = None
    else:
        print(
            "ℹ️  RAG rule store is empty — RAG disabled. "
            "Call POST /analyst/ingest-rules to index Thynk rules."
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
        print(f"✅ AI Analyst initialised successfully (RAG {rag_status}).")

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
):

    if analyst_service is None:
        raise HTTPException(
            status_code=500,
            detail="AI Analyst service is not initialized.",
        )

    try:

        print("\n" + "=" * 80)
        print("CREATE ANALYST SESSION")
        print("=" * 80)
        print("Received claim intelligence:")
        print(request.claim_intelligence)
        print("=" * 80)

        session_id = analyst_service.create_session(
            claim_intelligence=request.claim_intelligence
        )

        return AnalystSessionResponse(
            session_id=session_id,
            message="Analyst session created successfully.",
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
):

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
def ingest_rules():

    global retriever, analyst_service

    try:

        print("\n" + "=" * 80)
        print("INGEST THYNK RULES → RAG STORE")
        print("=" * 80)

        from .rag.ingest import run_ingest
        result = run_ingest()

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

        retriever = Retriever()

        if analyst_service is not None:
            analyst_service.retriever = retriever

        print(
            f"✅ Ingest complete — {result['ingested']} rules upserted, "
            f"{result['total_in_store']} total in store. "
            f"Retriever re-initialised."
        )
        print("=" * 80 + "\n")

        return {
            "status":          "ok",
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
