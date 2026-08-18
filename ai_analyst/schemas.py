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


class AnalystChatResponse(BaseModel):
    session_id: str
    answer: str