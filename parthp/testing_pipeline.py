# testing_submissions.py
# ------------------------------------------------------------
# Run a 90-10 split test per assignment to evaluate hint quality
# Uses existing ast_store, regenerates improvement_pairs + motif_stats
# Outputs top 5 hints per test submission into testing_submissions.csv

import csv
import json
import random
from collections import defaultdict

from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, DateTime, BINARY, ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.mysql import insert as mysql_insert

# --- DB Setup ---
ENGINE_URL = (
    "mysql+pymysql:///clp_feedback"
    "?read_default_file=~/.my.cnf"
    "&charset=latin1"
)
engine = create_engine(ENGINE_URL, pool_size=10, max_overflow=20, pool_recycle=3600)
Session = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# --- Models ---
class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, nullable=False)
    user_id = Column(Integer)
    timestamp = Column(DateTime)
    submission = Column(Text)
    measure = Column(Float)

class AstStore(Base):
    __tablename__ = "ast_store"
    submission_id = Column(Integer, primary_key=True)
    ast_json = Column(Text, nullable=False)

class ImprovementPair(Base):
    __tablename__ = "improvement_pairs"
    submission_id_a = Column(Integer, primary_key=True)
    submission_id_b = Column(Integer, primary_key=True)
    delta_score = Column(Float)

class MotifStat(Base):
    __tablename__ = "motif_stats"
    pattern_id = Column(BINARY(32), primary_key=True)
    assignment_id = Column(Integer, primary_key=True)
    support = Column(Integer, nullable=False, default=0)
    delta_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    token_text = Column(Text, nullable=False)
    mean_gain = Column(Float)
    composite_score = Column(Float, default=0)
    last_seen = Column(DateTime)

class Assignment(Base):
    __tablename__ = "assignments"
    id = Column(Integer, primary_key=True)
    class_id = Column(Integer, ForeignKey("classes.id"))
    dir = Column(Text)
    measure = Column(Text)

class Class(Base):
    __tablename__ = "classes"
    id = Column(Integer, primary_key=True)
    code = Column(Text)
    number = Column(Integer)
    section = Column(Integer)
    semester = Column(Text)

class Measure(Base):
    __tablename__ = "measure"
    id = Column(Integer, primary_key=True)
    short_name = Column(Text)
    file_name = Column(Text)
    data_type = Column(Text)
    order = Column(Integer)  # 0: descending is better, 1: ascending is better

# --- Utility ---
def generate_hint(token: str) -> str:
    if token.startswith("CALL:"):
        return f"Consider using `{token[5:]}()`"
    elif token.startswith("HP:"):
        parts = token.split(":")
        if len(parts) == 3:
            return f"Try tuning `{parts[2]}` in `{parts[1]}`"
        return f"Try adjusting hyperparameter `{parts[-1]}`"
    return f"Consider incorporating `{token}` — a common high-gain pattern."

# --- Main Pipeline ---
def run_split_test():
    sess = Session()
    assignment_ids = [r[0] for r in sess.query(Submission.assignment_id).distinct()]
    output_rows = []

    for aid in assignment_ids:
        subs = sess.query(Submission).filter(Submission.assignment_id == aid).all()
        if len(subs) < 10:
            print(f"[skip] assignment {aid}: too few submissions")
            continue

        # Determine sort direction from assignment.measure → measure.order
        assignment = sess.get(Assignment, aid)
        measure_order = 0  # Default: descending is better
        if assignment and assignment.measure:
            m = sess.query(Measure).filter(Measure.short_name == assignment.measure).first()
            if m:
                measure_order = m.order  # 0: descending (gain), 1: ascending (loss)

        random.shuffle(subs)
        split = int(0.9 * len(subs))
        train_subs, test_subs = subs[:split], subs[split:]
        train_ids = {s.id for s in train_subs}
        test_ids = {s.id for s in test_subs}

        # --- Clear & Rebuild improvement_pairs ---
        sess.query(ImprovementPair).delete()
        sess.flush()
        user_groups = defaultdict(list)
        for s in sorted(train_subs, key=lambda x: (x.user_id, x.timestamp)):
            user_groups[(s.user_id, s.assignment_id)].append(s)
        for group in user_groups.values():
            for a, b in zip(group, group[1:]):
                if a.measure is not None and b.measure is not None:
                    delta = b.measure - a.measure
                    if (measure_order == 0 and delta > 0) or (measure_order == 1 and delta < 0):
                        sess.add(ImprovementPair(
                            submission_id_a=a.id,
                            submission_id_b=b.id,
                            delta_score=abs(delta)
                        ))
        sess.flush()

        # --- Clear & Rebuild motif_stats ---
        sess.query(MotifStat).filter(MotifStat.assignment_id == aid).delete()
        sess.flush()
        pairs = sess.query(ImprovementPair).all()
        for p in pairs:
            a_ast = sess.get(AstStore, p.submission_id_a)
            b_ast = sess.get(AstStore, p.submission_id_b)
            if not a_ast or not b_ast:
                continue
            tokens_a = set(json.loads(a_ast.ast_json))
            tokens_b = set(json.loads(b_ast.ast_json))
            added = tokens_b - tokens_a
            for tok in added:
                row = {
                    "pattern_id": hashlib.sha256(tok.encode()).digest(),
                    "assignment_id": aid,
                    "support": 1,
                    "delta_score": p.delta_score,
                    "token_text": tok,
                    "mean_gain": p.delta_score,
                    "last_seen": func.now(),
                }
                stmt = mysql_insert(MotifStat).values([row]).on_duplicate_key_update(
                    support=MotifStat.support + 1,
                    mean_gain=(MotifStat.mean_gain * MotifStat.support + row["delta_score"]) / (MotifStat.support + 1),
                    delta_score=row["delta_score"],
                    last_seen=func.now(),
                )
                sess.execute(stmt)
        sess.flush()

        # --- Normalize support and mean_gain for composite score ---
        motif_rows = sess.query(MotifStat).filter(MotifStat.assignment_id == aid).all()
        support_max = max((m.support for m in motif_rows), default=1)
        gain_max = max((m.mean_gain for m in motif_rows), default=1.0)
        for m in motif_rows:
            norm_support = m.support / support_max
            norm_gain = m.mean_gain / gain_max
            m.composite_score = round(0.5 * norm_support + 0.5 * norm_gain, 4)
        sess.flush()

        # --- Suggest hints for test set ---
        for s in test_subs:
            ast_row = sess.get(AstStore, s.id)
            if not ast_row:
                continue
            tokens = set(json.loads(ast_row.ast_json))
            motifs = [m for m in motif_rows if m.token_text not in tokens and not m.token_text.startswith("IMPORT:")]
            motifs.sort(key=lambda m: m.composite_score, reverse=True)
            top_motifs = motifs[:5]
            hints = [generate_hint(m.token_text) + f" (Score: {m.composite_score:.2f})" for m in top_motifs]
            hints += [""] * (5 - len(hints))

            assign = sess.get(Assignment, s.assignment_id)
            cls = sess.get(Class, assign.class_id) if assign else None

            output_rows.append([
                s.id, s.assignment_id,
                cls.code if cls else "", cls.number if cls else "", cls.section if cls else "", cls.semester if cls else "",
                assign.dir if assign else ""
            ] + hints)

    # --- Write to CSV ---
    with open("testing_submissions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "submission_id", "assignment_id", "class", "number", "section", "semester", "assignment_name",
            "hint_1", "hint_2", "hint_3", "hint_4", "hint_5"
        ])
        writer.writerows(output_rows)

    sess.commit()
    sess.close()
    print("[done] testing_submissions.csv written")

if __name__ == "__main__":
    import hashlib
    run_split_test()


"""
Assignment 33 manual check
High score: /home/parthp/clp-submissions-copy/CMPE255-02-sp21/prhw1/dff4a3712e3f4fee8a71acc931a8755f_CMPE255_Bisecting_K_Means.ipynb
Lower score: /home/parthp/clp-submissions-copy/CMPE255-02-sp21/prhw1/03ce5998064b4fc2b49c8d70fb52a9c8_HW1 - LDA.py

RMSE assignments: 12, 43, 49, 50, 52


"""