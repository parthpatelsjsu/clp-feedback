"""
extract_improvement_pairs.py
----------------------------
Populate `improvement_pairs` with (submission_A → submission_B, delta score)
for every user × assignment where the score increases.

• Full back-fill by default; --since ISO-timestamp for incremental runs
• Reads DB creds from ~/.my.cnf    (no passwords in code)
• Uses MySQL INSERT IGNORE → duplicate pairs are skipped automatically

RUN ORDER: 1
LAST RUN: 2025-07-14, inserted 204 pairs in improvement_pairs (total submissions = 548, 
for assignment 55  and 56 which is CMPE255 Fall 2023 (class_id = 34))
"""

from __future__ import annotations
import argparse
import datetime as dt
from collections import defaultdict

from sqlalchemy import (
    create_engine, Column, Float, DateTime, ForeignKey, Index, func
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.mysql import INTEGER, BIGINT, insert as mysql_insert

# ---------------------------------------------------------------------------
# 1. DATABASE CONNECTION
# ---------------------------------------------------------------------------
ENGINE_URL = (
    "mysql+pymysql:///clp_feedback"
    "?read_default_file=~/.my.cnf"
    "&charset=latin1"
)

engine = create_engine(
    ENGINE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600            # recycle idle conns
)
Session = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# ---------------------------------------------------------------------------
# 2. ORM MODELS  (only required columns)
# ---------------------------------------------------------------------------
class Submission(Base):
    __tablename__ = "submissions"

    id            = Column(INTEGER(unsigned=True), primary_key=True)
    assignment_id = Column(INTEGER(unsigned=True), nullable=False)
    user_id       = Column(BIGINT(unsigned=True),  nullable=False)
    timestamp     = Column(DateTime, nullable=False)   # col name is 'timestamp'
    measure       = Column(Float)


class ImprovementPair(Base):
    __tablename__ = "improvement_pairs"

    id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)

    submission_id_a = Column(
        INTEGER(unsigned=True),
        ForeignKey("submissions.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    submission_id_b = Column(
        INTEGER(unsigned=True),
        ForeignKey("submissions.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    delta_score = Column(Float, nullable=False)
    created_at  = Column(DateTime, default=func.now(), index=True)

Index(
    "uniq_pair",
    ImprovementPair.submission_id_a,
    ImprovementPair.submission_id_b,
    unique=True
)

# ---------------------------------------------------------------------------
# 3. EXTRACTOR CORE
# ---------------------------------------------------------------------------
BATCH_SIZE = 10_000   # tune per memory/latency


def extract_pairs(since: dt.datetime | None = None, assignment_id: int | None = None) -> int:
    """
    Insert (A→B, Δscore) rows where B.score > A.score.
    If *since* is None, scan the entire table.
    Returns count of *attempted* inserts (duplicates ignored).
    """
    sess = Session()
    total_added = 0

    q = sess.query(
        Submission.user_id,
        Submission.assignment_id,
        Submission.id,
        Submission.measure.label("score"),
        Submission.timestamp,
    )
    if assignment_id:
        q = q.filter(Submission.assignment_id == assignment_id)

    if since:
        q = q.filter(Submission.timestamp > since)

    q = q.order_by(
        Submission.user_id,
        Submission.assignment_id,
        Submission.timestamp,
    )

    groups: dict[tuple[int, int], list] = defaultdict(list)

    for row in q.yield_per(BATCH_SIZE):
        key = (row.user_id, row.assignment_id)
        groups[key].append(row)

        if len(groups[key]) >= BATCH_SIZE:
            total_added += _flush_group(sess, groups[key])
            groups[key].clear()

    for subs in groups.values():
        total_added += _flush_group(sess, subs)

    sess.commit()
    sess.close()
    return total_added


def _flush_group(sess, subs: list) -> int:
    """
    Given submissions for one user × assignment (time-sorted),
    build INSERT IGNORE rows for every upward score jump.
    """
    vals = [
        {
            "submission_id_a": prev.id,
            "submission_id_b": cur.id,
            "delta_score":     cur.score - prev.score,
        }
        for prev, cur in zip(subs, subs[1:])
        if prev.score is not None and cur.score is not None and cur.score > prev.score
    ]

    if vals:
        stmt = mysql_insert(ImprovementPair).values(vals).prefix_with("IGNORE")
        sess.execute(stmt)
    return len(vals)

# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back-fill or incrementally populate improvement_pairs."
    )
    parser.add_argument(
        "--since",
        help="ISO-8601 timestamp (UTC) for incremental mode, "
             "e.g. 2025-07-01T00:00. If omitted, scans ALL submissions."
    )
    parser.add_argument(
    "--assignment", type=int,
    help="Only process submissions from this assignment_id"
)

    args = parser.parse_args()

    since_dt = dt.datetime.fromisoformat(args.since) if args.since else None
    inserted = extract_pairs(since_dt, args.assignment)
    span = args.since or "ALL TIME"
    print(f"[extract_improvement_pairs] attempted {inserted} inserts  (span: {span})")


