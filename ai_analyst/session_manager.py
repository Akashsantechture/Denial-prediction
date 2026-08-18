from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.database import SessionLocal
from .db.models import (
    AnalystSession,
    AnalystMessage,
    AnalystClaimIntelligence,
)


class SessionManager:

    # ============================================================
    # CREATE SESSION
    # ============================================================

    def create_session(
        self,
        claim_intelligence: Dict[str, Any],
        model_version: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> str:

        import uuid

        session_id = str(uuid.uuid4())

        db: Session = SessionLocal()

        try:

            # ----------------------------------------------------
            # Create session
            # ----------------------------------------------------

            session = AnalystSession(
                session_id=session_id,
                model_version=model_version,
                llm_model=llm_model,
                status="active",
            )

            db.add(session)

            # ----------------------------------------------------
            # Persist Claim Intelligence
            # ----------------------------------------------------

            evidence = AnalystClaimIntelligence(
                session_id=session_id,
                evidence=claim_intelligence,
            )

            db.add(evidence)

            # ----------------------------------------------------
            # Commit both records together
            # ----------------------------------------------------

            db.commit()

            print("\n" + "=" * 80)
            print("POSTGRES SESSION CREATED")
            print("=" * 80)
            print(f"Session ID : {session_id}")
            print(f"Model      : {model_version}")
            print(f"LLM        : {llm_model}")
            print("Claim Intelligence : STORED")
            print("=" * 80 + "\n")

            return session_id

        except Exception:

            db.rollback()

            raise

        finally:

            db.close()

    # ============================================================
    # GET SESSION
    # ============================================================

    def get_session(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:

        db: Session = SessionLocal()

        try:

            stmt = select(
                AnalystSession
            ).where(
                AnalystSession.session_id == session_id
            )

            session = db.execute(
                stmt
            ).scalar_one_or_none()

            if session is None:
                return None

            return {
                "session_id": session.session_id,
                "model_version": session.model_version,
                "llm_model": session.llm_model,
                "status": session.status,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }

        finally:

            db.close()

    # ============================================================
    # ADD MESSAGE
    # ============================================================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:

        db: Session = SessionLocal()

        try:

            # ----------------------------------------------------
            # Check session
            # ----------------------------------------------------

            session_stmt = select(
                AnalystSession
            ).where(
                AnalystSession.session_id == session_id
            )

            session = db.execute(
                session_stmt
            ).scalar_one_or_none()

            if session is None:

                raise ValueError(
                    f"Session '{session_id}' does not exist."
                )

            # ----------------------------------------------------
            # Create message
            # ----------------------------------------------------

            message = AnalystMessage(
                session_id=session_id,
                role=role,
                content=content,
            )

            db.add(message)

            # ----------------------------------------------------
            # Update session timestamp
            # ----------------------------------------------------

            from datetime import datetime, timezone

            session.updated_at = datetime.now(
                timezone.utc
            )

            db.commit()

            print(
                f"[POSTGRES] Message stored | "
                f"session={session_id} | "
                f"role={role}"
            )

        except Exception:

            db.rollback()

            raise

        finally:

            db.close()

    # ============================================================
    # GET MESSAGES
    # ============================================================

    def get_messages(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:

        db: Session = SessionLocal()

        try:

            # ----------------------------------------------------
            # Verify session
            # ----------------------------------------------------

            session_stmt = select(
                AnalystSession
            ).where(
                AnalystSession.session_id == session_id
            )

            session = db.execute(
                session_stmt
            ).scalar_one_or_none()

            if session is None:

                raise ValueError(
                    f"Session '{session_id}' does not exist."
                )

            # ----------------------------------------------------
            # Get messages
            # ----------------------------------------------------

            message_stmt = (
                select(AnalystMessage)
                .where(
                    AnalystMessage.session_id == session_id
                )
                .order_by(
                    AnalystMessage.created_at.asc()
                )
            )

            messages = db.execute(
                message_stmt
            ).scalars().all()

            return [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ]

        finally:

            db.close()

    # ============================================================
    # GET CLAIM INTELLIGENCE
    # ============================================================

    def get_claim_intelligence(
        self,
        session_id: str,
    ) -> Dict[str, Any]:

        db: Session = SessionLocal()

        try:

            stmt = select(
                AnalystClaimIntelligence
            ).where(
                AnalystClaimIntelligence.session_id
                == session_id
            )

            evidence = db.execute(
                stmt
            ).scalar_one_or_none()

            if evidence is None:

                raise ValueError(
                    f"Claim intelligence for session "
                    f"'{session_id}' does not exist."
                )

            return evidence.evidence

        finally:

            db.close()

    # ============================================================
    # UPDATE CLAIM INTELLIGENCE
    # ============================================================

    def update_claim_intelligence(
        self,
        session_id: str,
        claim_intelligence: Dict[str, Any],
    ) -> None:

        db: Session = SessionLocal()

        try:

            stmt = select(
                AnalystClaimIntelligence
            ).where(
                AnalystClaimIntelligence.session_id
                == session_id
            )

            evidence = db.execute(
                stmt
            ).scalar_one_or_none()

            if evidence is None:

                raise ValueError(
                    f"Claim intelligence for session "
                    f"'{session_id}' does not exist."
                )

            evidence.evidence = claim_intelligence

            db.commit()

            print(
                f"[POSTGRES] Claim intelligence updated | "
                f"session={session_id}"
            )

        except Exception:

            db.rollback()

            raise

        finally:

            db.close()

    # ============================================================
    # DELETE SESSION
    # ============================================================

    def delete_session(
        self,
        session_id: str,
    ) -> bool:

        db: Session = SessionLocal()

        try:

            stmt = select(
                AnalystSession
            ).where(
                AnalystSession.session_id == session_id
            )

            session = db.execute(
                stmt
            ).scalar_one_or_none()

            if session is None:
                return False

            # ----------------------------------------------------
            # Because AnalystMessage and
            # AnalystClaimIntelligence have CASCADE,
            # deleting the session also deletes its children.
            # ----------------------------------------------------

            db.delete(session)

            db.commit()

            print(
                f"[POSTGRES] Session deleted: "
                f"{session_id}"
            )

            return True

        except Exception:

            db.rollback()

            raise

        finally:

            db.close()