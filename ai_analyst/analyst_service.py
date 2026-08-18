import json
from typing import Any, Dict

from .session_manager import SessionManager
from .llm_reasoner import LLMReasoner


class AnalystService:

    def __init__(
        self,
        session_manager: SessionManager,
        llm_reasoner: LLMReasoner,
    ):

        self.session_manager = session_manager
        self.llm_reasoner = llm_reasoner

    # ============================================================
    # CREATE SESSION
    # ============================================================

    def create_session(
        self,
        claim_intelligence: Dict[str, Any],
    ) -> str:

        return self.session_manager.create_session(
            claim_intelligence
        )

    # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> str:

        # --------------------------------------------------------
        # Validate session and retrieve claim intelligence
        # --------------------------------------------------------

        claim_intelligence = (
            self.session_manager
            .get_claim_intelligence(session_id)
        )

        # --------------------------------------------------------
        # Store current user message
        # --------------------------------------------------------

        self.session_manager.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        # --------------------------------------------------------
        # Retrieve conversation AFTER adding current message
        # --------------------------------------------------------

        conversation = (
            self.session_manager
            .get_messages(session_id)
        )

        # --------------------------------------------------------
        # DEBUG
        # --------------------------------------------------------

        print("\n" + "=" * 80)
        print("ANALYST SERVICE → LLM")
        print("=" * 80)

        print("\nSESSION:")
        print(session_id)

        print("\nUSER QUESTION:")
        print(user_message)

        print("\nCLAIM INTELLIGENCE:")
        print(
            json.dumps(
                claim_intelligence,
                indent=2,
                default=str,
            )
        )

        print("\nCONVERSATION:")
        print(
            json.dumps(
                conversation,
                indent=2,
                default=str,
            )
        )

        print("=" * 80 + "\n")

        # --------------------------------------------------------
        # Ask Gemini
        # --------------------------------------------------------

        answer = self.llm_reasoner.generate_response(
            claim_intelligence=claim_intelligence,
            conversation=conversation,
            user_message=user_message,
        )

        # --------------------------------------------------------
        # Store assistant response
        # --------------------------------------------------------

        self.session_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return answer