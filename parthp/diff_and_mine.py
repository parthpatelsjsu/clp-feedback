#!/usr/bin/env python3
"""
diff_and_mine.py
────────────────────────────────────────
For every (A → B) row in `improvement_pairs`, load the JSON‑encoded *sorted*
AST‑token lists from `ast_store`, compute the **added** tokens, and up‑sert one
row per added token into `motif_stats`.

`motif_stats` schema (final):
    pattern_id    BINARY(32) PK
    assignment_id INT        PK
    support       INT        NOT NULL DEFAULT 0
    delta_score   DOUBLE     NOT NULL               -- latest Δ
    created_at    TIMESTAMP  NOT NULL DEFAULT NOW() -- first seen
    token_text    TEXT
    median_gain   DOUBLE  NULL                      -- optional
    mean_gain     DOUBLE  NULL
    last_seen     TIMESTAMP NULL

On every duplicate key we **increment support** and keep a running mean gain;
`last_seen` is overwritten with the current timestamp, and the latest
`delta_score` is recorded.

CLI:
  --assignment 55        # mine all pairs in assignment 55
  --pair 13104 13106     # mine exactly one pair

RUN ORDER: 3 (after build_ast_store)
  EX:
    submission id 13152 and 13113 have a significant improvement.
    13152: 914b131204404b1db8a91ee5c60e3e62_output (1).dat
    13113: c43c8fcf11b743bfb30706a0b7dbca07_output (1).dat

    Test: 13288, 13289
"""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from typing import Optional, List, Set

from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, Float, DateTime, ForeignKey,
    insert as mysql_insert, Index, text as sql_text, Text
)

from sqlalchemy.dialects.mysql import insert as mysql_insert, BINARY, LONGTEXT
from sqlalchemy.orm import declarative_base, sessionmaker, aliased

# ───────────────────────── DB CONNECTION ─────────────────────────
ENGINE_URL = (
    "mysql+pymysql:///clp_feedback"
    "?read_default_file=~/.my.cnf"
    "&charset=latin1"
)
engine = create_engine(
    ENGINE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
)
Session = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# ───────────────────────── ORM MODELS ────────────────────────────
class Submission(Base):
    __tablename__ = "submissions"

    id            = Column(Integer, primary_key=True)
    assignment_id = Column(Integer)

class ImprovementPair(Base):
    __tablename__ = "improvement_pairs"

    submission_id_a = Column(Integer, primary_key=True)
    submission_id_b = Column(Integer, primary_key=True)
    delta_score     = Column(Float)

class ASTStore(Base):
    __tablename__ = "ast_store"

    submission_id = Column(Integer, primary_key=True)
    ast_json      = Column(LONGTEXT)

class MotifStat(Base):
    __tablename__ = "motif_stats"

    pattern_id    = Column(BINARY(32), primary_key=True)
    assignment_id = Column(Integer, primary_key=True)

    support       = Column(Integer, nullable=False, default=0)
    delta_score   = Column(Float,   nullable=False)

    created_at    = Column(DateTime, default=dt.datetime.utcnow)
    token_text    = Column(Text, nullable=False)
    median_gain   = Column(Float)
    mean_gain     = Column(Float)
    last_seen     = Column(DateTime)

Index("idx_motif_created", MotifStat.created_at)

# ───────────────────────── HELPERS ───────────────────────────────

def sha256_id(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()

def load_tokens(ast_json: str) -> Set[str]:
    """Return *set* of tokens from JSON‑encoded list."""
    return set(json.loads(ast_json))

# ───────────────────────── CORE LOGIC ────────────────────────────

def process_pair(sess, pair_row, assignment_id: int) -> int:
    """Insert / up‑date one motif_stats row per added token.
    Returns number of tokens processed."""
    ast_a = sess.get(ASTStore, pair_row.submission_id_a)
    ast_b = sess.get(ASTStore, pair_row.submission_id_b)
    if not ast_a or not ast_b:
        print(f"[skip] pair ({pair_row.submission_id_a},{pair_row.submission_id_b}) – missing AST")
        return 0

    added = load_tokens(ast_b.ast_json) - load_tokens(ast_a.ast_json)
    if not added:
        return 0

    now = dt.datetime.utcnow()
    rows = [
        {
            "pattern_id":    sha256_id(tok),
            "assignment_id": assignment_id,
            "support":       1,              # first occurrence
            "delta_score":   pair_row.delta_score,
            "token_text":    tok,
            "created_at":    now,
            "mean_gain":     pair_row.delta_score, # replacing this with delta_score for now
            "median_gain":   None,           # optional – filled later
            "last_seen":     now,
        }
        for tok in added
    ]

    stmt = mysql_insert(MotifStat).values(rows)

    # ── ON DUPLICATE KEY: bump support & update stats ────────────
    stmt = stmt.on_duplicate_key_update(
        support      = MotifStat.support + 1,
        # running mean:  (old*old_n + new) / (old_n+1)
        mean_gain    = (
            (MotifStat.mean_gain * MotifStat.support + stmt.inserted.delta_score)
            / (MotifStat.support + 1)
        ),
        delta_score  = stmt.inserted.delta_score,
        last_seen    = stmt.inserted.created_at,
        # token_text stays unchanged (identical)
    )

    sess.execute(stmt)
    return len(rows)

# ───────────────────────── RUNNERS ───────────────────────────────

def run(assignment_id: Optional[int], pair_ids: Optional[List[int]]) -> None:
    sess = Session()
    total = 0

    if pair_ids:  # mine a single explicit pair
        a_id, b_id = pair_ids
        pair = sess.get(ImprovementPair, {
            "submission_id_a": a_id,
            "submission_id_b": b_id,
        })
        if not pair:
            print("[warn] improvement pair not found")
            sess.close()
            return
        # derive assignment from submission A
        aid = sess.get(Submission, a_id).assignment_id
        total += process_pair(sess, pair, aid)

    else:  # full or assignment‑filtered run
        Sub = aliased(Submission)
        q = (
            sess.query(ImprovementPair, Sub.assignment_id)
            .join(Sub, Sub.id == ImprovementPair.submission_id_a)
        )
        if assignment_id:
            q = q.filter(Sub.assignment_id == assignment_id)

        for pair_row, aid in q.yield_per(1000):
            total += process_pair(sess, pair_row, aid)

    sess.commit()
    sess.close()
    print(f"[diff_and_mine] processed tokens: {total}")

# ───────────────────────── CLI ───────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assignment", type=int, help="limit to one assignment_id")
    p.add_argument("--pair", nargs=2, type=int, metavar=("A_ID", "B_ID"),
                   help="mine exactly one improvement pair")
    args = p.parse_args()

    run(args.assignment, args.pair)
