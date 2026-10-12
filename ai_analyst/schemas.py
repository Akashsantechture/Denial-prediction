from typing import Any, Dict

from pydantic import BaseModel


class AnalystSessionRequest(BaseModel):
    """
    Starts a conversational Analyst session.

    The prediction service's complete claim intelligence
    is accepted without restructuring.
    """

    claim_intelligence: Dict[str, Any]


class AnalystSessionResponse(BaseModel):
    session_id: str
    message: str


class AnalystChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True   # frontend toggle — False skips RAG retrieval for this turn


class AnalystChatResponse(BaseModel):
    session_id: str
    answer: str
    rag_used: bool = False   # True when RAG retrieved rules for this response