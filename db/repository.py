"""
Repository layer for the traceability DB. Every function opens its own
session and returns plain dicts/ints (never ORM objects), so callers never
hit DetachedInstanceError once the session that produced the data is closed.
"""
from typing import Dict, List, Optional

from db.models import Conclusion, Contradiction, Finding, ResearchQuestion, Source
from db.session import get_session


def create_question(topic: str) -> int:
    with get_session() as session:
        q = ResearchQuestion(topic=topic, sub_questions=[])
        session.add(q)
        session.flush()
        return q.id


def set_sub_questions(question_id: int, sub_questions: List[str]) -> None:
    with get_session() as session:
        q = session.get(ResearchQuestion, question_id)
        q.sub_questions = sub_questions


def get_question(question_id: int) -> Optional[dict]:
    with get_session() as session:
        q = session.get(ResearchQuestion, question_id)
        if q is None:
            return None
        return {
            "id": q.id,
            "topic": q.topic,
            "sub_questions": q.sub_questions or [],
            "created_at": q.created_at.isoformat(),
        }


def get_or_create_source(question_id: int, url: str, title: str = "", raw_content: str = "") -> int:
    url = (url or "").strip() or "none"
    with get_session() as session:
        existing = (
            session.query(Source)
            .filter(Source.question_id == question_id, Source.url == url)
            .first()
        )
        if existing:
            return existing.id
        source = Source(question_id=question_id, url=url, title=title, raw_content=raw_content)
        session.add(source)
        session.flush()
        return source.id


def add_finding(question_id: int, source_id: int, text: str, sub_question: str = "", category: Optional[str] = None) -> int:
    with get_session() as session:
        finding = Finding(
            question_id=question_id,
            source_id=source_id,
            text=text,
            sub_question=sub_question,
            category=category,
        )
        session.add(finding)
        session.flush()
        return finding.id


def set_finding_category(finding_id: int, category: str) -> None:
    with get_session() as session:
        finding = session.get(Finding, finding_id)
        if finding is not None:
            finding.category = category


def add_contradiction(finding_a_id: int, finding_b_id: int, explanation: str) -> int:
    with get_session() as session:
        contradiction = Contradiction(
            finding_a_id=finding_a_id, finding_b_id=finding_b_id, explanation=explanation
        )
        session.add(contradiction)
        session.flush()
        return contradiction.id


def add_conclusion(question_id: int, text: str, finding_ids: List[int]) -> int:
    with get_session() as session:
        conclusion = Conclusion(question_id=question_id, text=text, finding_ids=finding_ids)
        session.add(conclusion)
        session.flush()
        return conclusion.id


def _finding_to_dict(f: Finding) -> dict:
    return {
        "id": f.id,
        "question_id": f.question_id,
        "source_id": f.source_id,
        "text": f.text,
        "category": f.category,
        "sub_question": f.sub_question,
        "created_at": f.created_at.isoformat(),
        "source_url": f.source.url if f.source else None,
        "source_title": f.source.title if f.source else None,
    }


def get_findings_list(question_id: int) -> List[dict]:
    with get_session() as session:
        findings = (
            session.query(Finding)
            .filter(Finding.question_id == question_id)
            .order_by(Finding.id)
            .all()
        )
        return [_finding_to_dict(f) for f in findings]


def get_findings_grouped(question_id: int) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for f in get_findings_list(question_id):
        key = f["sub_question"] or "General"
        grouped.setdefault(key, []).append(f)
    return grouped


def get_finding_by_id(finding_id: int) -> Optional[dict]:
    with get_session() as session:
        f = session.get(Finding, finding_id)
        return _finding_to_dict(f) if f else None


def get_contradictions(question_id: int) -> List[dict]:
    with get_session() as session:
        rows = (
            session.query(Contradiction)
            .join(Finding, Contradiction.finding_a_id == Finding.id)
            .filter(Finding.question_id == question_id)
            .all()
        )
        out = []
        for c in rows:
            fa = session.get(Finding, c.finding_a_id)
            fb = session.get(Finding, c.finding_b_id)
            out.append(
                {
                    "id": c.id,
                    "finding_a_id": c.finding_a_id,
                    "finding_b_id": c.finding_b_id,
                    "explanation": c.explanation,
                    "finding_a_text": fa.text if fa else None,
                    "finding_b_text": fb.text if fb else None,
                    "sub_question": fa.sub_question if fa else None,
                    "created_at": c.created_at.isoformat(),
                }
            )
        return out


def get_conclusions(question_id: int) -> List[dict]:
    with get_session() as session:
        rows = (
            session.query(Conclusion)
            .filter(Conclusion.question_id == question_id)
            .order_by(Conclusion.id)
            .all()
        )
        return [
            {
                "id": c.id,
                "question_id": c.question_id,
                "text": c.text,
                "finding_ids": c.finding_ids or [],
                "created_at": c.created_at.isoformat(),
            }
            for c in rows
        ]


def get_conclusion_by_id(conclusion_id: int) -> Optional[dict]:
    with get_session() as session:
        c = session.get(Conclusion, conclusion_id)
        return (
            {
                "id": c.id,
                "question_id": c.question_id,
                "text": c.text,
                "finding_ids": c.finding_ids or [],
                "created_at": c.created_at.isoformat(),
            }
            if c
            else None
        )


def get_trace(question_id: int) -> Optional[dict]:
    """
    Full traceability chain for a question: conclusions -> the findings each
    was derived from -> the source each finding came from.
    """
    question = get_question(question_id)
    if question is None:
        return None

    findings_by_id = {f["id"]: f for f in get_findings_list(question_id)}
    conclusions = get_conclusions(question_id)

    trace = []
    for c in conclusions:
        chain = [findings_by_id[fid] for fid in c["finding_ids"] if fid in findings_by_id]
        trace.append({"conclusion": c, "findings": chain})

    return {
        "question": question,
        "conclusions": trace,
        "contradictions": get_contradictions(question_id),
    }


def list_questions(limit: int = 50) -> List[dict]:
    with get_session() as session:
        rows = (
            session.query(ResearchQuestion)
            .order_by(ResearchQuestion.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": q.id,
                "topic": q.topic,
                "sub_questions": q.sub_questions or [],
                "created_at": q.created_at.isoformat(),
            }
            for q in rows
        ]
