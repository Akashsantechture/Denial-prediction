import json
from typing import Any, Dict, Optional

from .session_manager import SessionManager
from .llm_reasoner import LLMReasoner
from . import thynk_lookup


class AnalystService:

    def __init__(
        self,
        session_manager: SessionManager,
        llm_reasoner: LLMReasoner,
        retriever=None,          # rag.retriever.Retriever — optional
    ):

        self.session_manager = session_manager
        self.llm_reasoner    = llm_reasoner
        self.retriever       = retriever   # None when RAG is not available

    # ============================================================
    # CREATE SESSION
    # ============================================================

    def create_session(
        self,
        claim_intelligence: Dict[str, Any],
    ) -> str:

        # --------------------------------------------------------
        # Enrich with Thynk validation data (backend-only lookup)
        # --------------------------------------------------------

        claim_id = (
            claim_intelligence.get("claim", {}).get("claim_id")
            or claim_intelligence.get("claim_id")
        )

        thynk_data = thynk_lookup.find(str(claim_id)) if claim_id else None

        if thynk_data:
            print(
                f"[ThynkLookup] Enriching session for claim {claim_id} — "
                f"{thynk_data['summary']['total']} validation items "
                f"(CRITICAL={thynk_data['summary']['critical']} "
                f"SEVERE={thynk_data['summary']['severe']} "
                f"WARNING={thynk_data['summary']['warning']} "
                f"INFO={thynk_data['summary']['info']})"
            )
            claim_intelligence = dict(claim_intelligence)
            claim_intelligence["thynk_validation"] = thynk_data
        else:
            print(
                f"[ThynkLookup] No Thynk data for claim {claim_id} — "
                "session will use model evidence only."
            )
            claim_intelligence = dict(claim_intelligence)
            claim_intelligence["thynk_validation"] = None

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
        use_rag: bool = True,
    ) -> str:

        # --------------------------------------------------------
        # Retrieve claim intelligence for this session
        # --------------------------------------------------------

        claim_intelligence = (
            self.session_manager
            .get_claim_intelligence(session_id)
        )

        # --------------------------------------------------------
        # RAG — retrieve relevant Thynk rules from ChromaDB
        # Skipped when use_rag=False (toggle OFF) or retriever
        # is not initialised.  Failures never block the response.
        # --------------------------------------------------------

        retrieved_rules: Optional[str] = None
        rag_used = False

        if use_rag and self.retriever is not None:
            try:
                query_text = self.retriever.build_query(
                    user_message=user_message,
                    claim_intelligence=claim_intelligence,
                )
                retrieved_rules = self.retriever.query(query_text, top_k=4)

                if retrieved_rules:
                    rag_used = True
                    print(
                        f"[RAG] Retrieved relevant rulebook context "
                        f"for session {session_id}."
                    )
                else:
                    print(
                        f"[RAG] No rules above similarity threshold "
                        f"for session {session_id}."
                    )
            except Exception as exc:
                print(f"[RAG] Retrieval failed silently: {exc}")
                retrieved_rules = None
        elif not use_rag:
            print(f"[RAG] Skipped — toggle OFF for session {session_id}.")

        # --------------------------------------------------------
        # Store user message
        # --------------------------------------------------------

        self.session_manager.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        # --------------------------------------------------------
        # Retrieve full conversation (includes message just added)
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
        print("\nRAG RULES RETRIEVED:", "YES" if retrieved_rules else "NO")
        print("\nCLAIM INTELLIGENCE:")
        print(json.dumps(claim_intelligence, indent=2, default=str))
        print("\nCONVERSATION:")
        print(json.dumps(conversation, indent=2, default=str))
        print("=" * 80 + "\n")

        # --------------------------------------------------------
        # Generate LLM response
        # --------------------------------------------------------

        answer = self.llm_reasoner.generate_response(
            claim_intelligence=claim_intelligence,
            conversation=conversation,
            user_message=user_message,
            retrieved_rules=retrieved_rules,
        )

        # --------------------------------------------------------
        # Store assistant response
        # --------------------------------------------------------

        self.session_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return answer, rag_used
