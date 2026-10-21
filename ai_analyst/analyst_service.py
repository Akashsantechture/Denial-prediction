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
        region: str = "uae",
        auto_summarize: bool = True,
    ) -> tuple[str, str | None]:
        """
        Create a new analyst session with optional auto-summary.
        
        Args:
            claim_intelligence: Claim data + predictions
            region: 'uae' or 'ksa'
            auto_summarize: Generate initial summary automatically
            
        Returns:
            tuple of (session_id, initial_summary or None)
        """
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

        # Create session
        session_id = self.session_manager.create_session(claim_intelligence)
        
        # --------------------------------------------------------
        # Generate initial summary if requested
        # --------------------------------------------------------
        
        if not auto_summarize:
            return session_id, None
        
        initial_summary = None
        rag_used = False
        
        try:
            # RAG lookup for relevant rules
            retrieved_rules = None
            if self.retriever is not None:
                from .rag.retriever import Retriever
                region_retriever = Retriever(region=region)
                
                # Build query from claim context
                query_parts = []
                
                # Add denied activities
                activities = claim_intelligence.get("prediction", {}).get("activity_predictions", [])
                denied = [a["activity_code"] for a in activities if a.get("predicted_denial")]
                if denied:
                    query_parts.append(f"Denied activities: {', '.join(denied)}")
                
                # Add primary diagnosis
                claim = claim_intelligence.get("claim", {})
                dx = claim.get("primary_diagnosis_code", "")
                if dx:
                    query_parts.append(f"Primary diagnosis: {dx}")
                
                # Add Thynk severity flags
                if thynk_data and thynk_data.get("found"):
                    summary = thynk_data.get("summary", {})
                    if summary.get("critical", 0):
                        query_parts.append("CRITICAL compliance issues")
                    if summary.get("severe", 0):
                        query_parts.append("SEVERE medical necessity issues")
                    # Add short messages from top rules
                    for item in thynk_data.get("all", [])[:3]:
                        msg = item.get("shortMessage", "")
                        if msg:
                            query_parts.append(msg)
                
                query_text = " | ".join(query_parts) if query_parts else "Claim overview and compliance"
                retrieved_rules = region_retriever.query(query_text, top_k=5)
                rag_used = retrieved_rules is not None
                
                if rag_used:
                    print(f"[InitialSummary] RAG retrieved rules for session {session_id} (region: {region.upper()})")
            
            # Generate summary using LLM
            initial_summary = self.llm_reasoner.generate_initial_summary(
                claim_intelligence=claim_intelligence,
                retrieved_rules=retrieved_rules,
                region=region
            )
            
            # Store as first message in session
            self.session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=initial_summary,
            )
            
            print(f"[InitialSummary] Generated for session {session_id} (RAG: {rag_used})")
            
        except Exception as exc:
            print(f"[InitialSummary] Generation failed: {exc}")
            import traceback
            traceback.print_exc()
            initial_summary = None
        
        return session_id, initial_summary

    # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        session_id: str,
        user_message: str,
        use_rag: bool = True,   # kept for API compat, always True internally
        region: str = "uae",
    ) -> tuple[str, bool]:
        region = region.lower()
        if region not in ("uae", "ksa"):
            region = "uae"

        claim_intelligence = self.session_manager.get_claim_intelligence(session_id)

        # ── Always retrieve relevant rules from knowledge base ──
        retrieved_rules: Optional[str] = None
        rag_used = False

        if self.retriever is not None:
            try:
                from .rag.retriever import Retriever
                region_retriever = Retriever(region=region)
                query_text = region_retriever.build_query(
                    user_message=user_message,
                    claim_intelligence=claim_intelligence,
                )
                retrieved_rules = region_retriever.query(query_text, top_k=4)
                if retrieved_rules:
                    rag_used = True
                    print(f"[RAG] Rules retrieved for session {session_id} (region: {region.upper()}).")
                else:
                    print(f"[RAG] No rules above threshold for session {session_id}.")
            except Exception as exc:
                print(f"[RAG] Retrieval failed silently: {exc}")
                retrieved_rules = None

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
