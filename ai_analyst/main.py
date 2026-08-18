from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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
llm_reasoner = None
analyst_service = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global llm_reasoner
    global analyst_service

    print("\n" + "=" * 80)
    print("STARTING AI ANALYST")
    print("=" * 80)

    # --------------------------------------------------------
    # Initialize PostgreSQL
    # --------------------------------------------------------

    try:

        init_db()

        print("✅ PostgreSQL initialized successfully.")

    except Exception as exc:

        print("❌ PostgreSQL initialization failed.")
        print(str(exc))

        raise


    try:

        llm_reasoner = LLMReasoner()

        analyst_service = AnalystService(
            session_manager=session_manager,
            llm_reasoner=llm_reasoner,
        )

        print("✅ AI Analyst initialized successfully.")

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
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "analyst_service_loaded": analyst_service is not None,
        "llm_loaded": llm_reasoner is not None,
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

        answer = analyst_service.chat(
            session_id=request.session_id,
            user_message=request.message,
        )

        return AnalystChatResponse(
            session_id=request.session_id,
            answer=answer,
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
# DELETE SESSION
# ============================================================

@app.delete("/analyst/session/{session_id}")
def delete_session(
    session_id: str,
):

    try:

        deleted = session_manager.delete_session(
            session_id
        )

        if not deleted:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        return {
            "status": "success",
            "message": "Session deleted.",
            "session_id": session_id,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )