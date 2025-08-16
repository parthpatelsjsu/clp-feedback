#!/usr/bin/env python3
"""
build_ast_store.py  
Parse all code files and cache canonical ASTs, store them into ast_store table.
RAN: 2025-07-14 on CMPE255-01-fa23 hw1 and hw2 (assignment id = 55 and 56)

USAGE:
 Argument --assignment selects ALL submissions under that assignment, not just the ones that have improved

RUN ORDER: 2
"""

from __future__ import annotations
import argparse, datetime as dt, hashlib, io, subprocess, tokenize, ast, os, json
from pathlib import Path
from typing import Optional
from sqlalchemy import (
    create_engine, Column, DateTime, Text, ForeignKey, func, Index, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.mysql import INTEGER, LONGTEXT
import ast

# ── Configuration ─────────────────────────────────────────────
ROOT_DEFAULT = "/home/parthp/clp-submissions-copy"
VALID_EXTS   = {".py", ".ipynb"}
SEM_MAP      = {"Fall": "fa", "Spring": "sp", "Summer": "su"}

# ── DB setup ───────────────────────────────────────────────────
ENGINE_URL = (
    "mysql+pymysql:///clp_feedback"
    "?read_default_file=~/.my.cnf"
    "&charset=latin1"
)
engine = create_engine(ENGINE_URL, pool_size=10, max_overflow=20, pool_recycle=3600)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# ── Models ─────────────────────────────────────────────────────
class Class(Base):
    __tablename__ = "classes"
    id       = Column(INTEGER(unsigned=True), primary_key=True)
    code     = Column(Text)
    number   = Column(INTEGER(unsigned=True))
    section  = Column(INTEGER(unsigned=True))
    semester = Column(Text)

class Assignment(Base):
    __tablename__ = "assignments"
    id       = Column(INTEGER(unsigned=True), primary_key=True)
    dir      = Column(Text)
    class_id = Column(INTEGER(unsigned=True), ForeignKey("classes.id"))

class Submission(Base):
    __tablename__ = "submissions"
    id            = Column(INTEGER(unsigned=True), primary_key=True)
    assignment_id = Column(INTEGER(unsigned=True), ForeignKey("assignments.id"))
    timestamp     = Column(DateTime, nullable=False)
    submission    = Column(Text)

class ASTStore(Base):
    __tablename__ = "ast_store"
    submission_id = Column(INTEGER(unsigned=True), ForeignKey("submissions.id"),
                           primary_key=True)
    ast_json   = Column(LONGTEXT, nullable=False)
    code_hash  = Column(Text(64), nullable=False)
    created_at = Column(DateTime, default=func.now())

Index("code_hash_idx", ASTStore.code_hash)

# ── AST helpers ────────────────────────────────────────────────
class SkipFile(Exception): pass

def semester_slug(sem: str) -> str:
    season, year = sem.split()
    return SEM_MAP.get(season, "xx") + year[-2:]

def build_folder(cls: Class) -> str:
    return f"{cls.code}{cls.number}-{str(cls.section).zfill(2)}-{semester_slug(cls.semester)}"

def strip_comments(code: str) -> str:
    io_obj, out = io.StringIO(code), []
    prev_type, last_lineno, last_col = tokenize.INDENT, -1, 0
    try:
        for tok_type, tok_str, (srow, scol), _, _ in tokenize.generate_tokens(io_obj.readline):
            if tok_type == tokenize.COMMENT: continue
            if tok_type == tokenize.STRING and prev_type == tokenize.INDENT: continue
            if srow > last_lineno: last_col = 0
            if scol > last_col: out.append(" " * (scol - last_col))
            out.append(tok_str)
            prev_type, last_lineno, last_col = tok_type, srow, scol + len(tok_str)
    except tokenize.TokenError as e:
        raise SyntaxError(f"tokenize error: {e}") from e
    return "".join(out)

class MaskLiterals(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (str, bytes)):
            return ast.copy_location(ast.Constant("<STR>"), node)
        return node
    def visit_List(self, node):
        return ast.copy_location(ast.Constant("<SEQ>"), node) if len(node.elts) >= 5 else self.generic_visit(node)
    def visit_Tuple(self, node):
        return ast.copy_location(ast.Constant("<SEQ>"), node) if len(node.elts) >= 5 else self.generic_visit(node)


def canonicalise(code_raw: str) -> tuple[str, str]:
    code_nc = strip_comments(code_raw)
    tree    = ast.parse(code_nc)
    tree    = MaskLiterals().visit(tree)
    ast.fix_missing_locations(tree)

    tokens  = sorted(tokens_from_ast(tree))          
    return (hashlib.sha256("".join(tokens).encode()).hexdigest(),
            json.dumps(tokens))



def load_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Detect notebook JSON saved with .py extension
    if path.suffix == ".py" and text.strip().startswith('{') and '"cells": [' in text:
        print(f"[convert] treating {path.name} as notebook JSON despite .py extension")
        # Write temp file and use nbconvert
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", tmp.name]
            result = subprocess.check_output(cmd, text=True)
            os.unlink(tmp.name)
            return result

    elif path.suffix == ".ipynb":
        cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(path)]
        return subprocess.check_output(cmd, text=True)

    elif path.suffix == ".py":
        return text

    raise SkipFile(f"Unknown file type: {path}")


# ── Main function ───────────────────────────────────────────────
def build(root: Path, after: Optional[dt.datetime], assignment_id: Optional[int]):
    sess = Session()

    q = (
        sess.query(Submission)
        .outerjoin(ASTStore, Submission.id == ASTStore.submission_id)
        .filter(ASTStore.submission_id.is_(None))
    )
    if after:
        q = q.filter(Submission.timestamp > after)
    if assignment_id:
        q = q.filter(Submission.assignment_id == assignment_id)

    submissions = q.all()
    print(f"[debug] total submissions to process: {len(submissions)}")

    processed, batch = 0, 0
    for sub in submissions:
        asn = sess.get(Assignment, sub.assignment_id)
        if not asn:
            print(f"[skip] sub {sub.id}: assignment {sub.assignment_id} not found")
            continue

        cls = sess.get(Class, asn.class_id)
        if not cls:
            print(f"[skip] sub {sub.id}: class {asn.class_id} not found")
            continue

        folder = build_folder(cls)
        assign_dir = asn.dir
        sub_dir = root / folder / assign_dir

        if not sub_dir.exists():
            print(f"[not-found] {sub.id} → missing folder {sub_dir}")
            continue

        prefix = sub.submission[:32] if sub.submission and len(sub.submission) >= 32 else "<no-prefix>"
        print(f"[debug] sub_id={sub.id}  prefix={prefix}  path={sub_dir}")

        code_file = None

        # 1️ Hash prefix match (glob)
        if prefix != "<no-prefix>":
            matches = list(sub_dir.glob(f"{prefix}*.py")) + list(sub_dir.glob(f"{prefix}*.ipynb"))
            if matches:
                code_file = matches[0]
                print(f"[match] prefix match → {code_file.name}")

        # 2️ Fallback: flat match
        if not code_file:
            matches = list(sub_dir.glob("*.py")) + list(sub_dir.glob("*.ipynb"))
            if matches:
                code_file = matches[0]
                print(f"[fallback-flat] sub {sub.id}: used {code_file.name}")

        # 3️ Fallback: recursive match
        if not code_file:
            for file in sub_dir.rglob("*"):
                if file.suffix in VALID_EXTS and file.name.startswith(prefix):
                    code_file = file
                    print(f"[fallback-rglob] sub {sub.id}: matched {file.name}")
                    break

        if not code_file:
            print(f"[fail] sub {sub.id}: no code file matched prefix {prefix} in {sub_dir}")
            with open("no_match_ids.txt", "a") as f:
                f.write(f"{sub.id}\t{sub.submission or 'NULL'}\t{prefix}\t{sub_dir}\n")
            continue

        try:
            code_raw = load_code(code_file)
        except Exception as e:
            print(f"[read-fail] sub {sub.id}: {e}")
            continue

        try:
            code_hash, ast_json = canonicalise(code_raw)
        except (SyntaxError, ValueError) as exc:
            code_hash = hashlib.sha256(code_raw.encode()).hexdigest()
            ast_json = json.dumps({"syntax_error": True})
            print(f"[placeholder] sub {sub.id}: {exc}")

        sess.add(ASTStore(
            submission_id=sub.id,
            ast_json=ast_json,
            code_hash=code_hash
        ))
        processed += 1
        batch += 1
        if batch >= 500:
            sess.commit()
            batch = 0

    sess.commit()
    real = sess.execute(text("SELECT COUNT(*) FROM ast_store")).scalar_one()
    sess.close()

    span = after.isoformat() if after else "ALL"
    asn_txt = assignment_id if assignment_id else "ALL"
    print(f"[ast_store] +{processed} rows (after={span}, assignment={asn_txt})")
    print(f"[ast_store] total rows now in DB: {real}")

def tokens_from_ast(tree: ast.AST) -> set[str]:
    """
    Walk the AST and emit structural tokens that are robust across
    submissions: imports, calls, and hyper-parameter keywords.
    """

    tokens: set[str] = set()

    class TVisitor(ast.NodeVisitor):
        # imports  -------------------------------------------------
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                tokens.add(f"IMPORT:{alias.name.split('.')[0]}")
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom):
            mod = (node.module or "").split('.')[0]
            for alias in node.names:
                tokens.add(f"IMPORT:{alias.name or mod}")
            self.generic_visit(node)

        # calls  ---------------------------------------------------
        def visit_Call(self, node: ast.Call):
            # Function name (simple cases)
            fname = ""
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname:
                tokens.add(f"CALL:{fname.lower()}")

            # keyword hyper-parameters
            for kw in node.keywords or []:
                if kw.arg:                       # ignore **kwargs
                    tokens.add(f"HP:{fname}:{kw.arg}")
            self.generic_visit(node)

    TVisitor().visit(tree)
    return tokens
# ── CLI entry ───────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT_DEFAULT)
    p.add_argument("--after")
    p.add_argument("--assignment", type=int)
    args = p.parse_args()

    after_dt = dt.datetime.fromisoformat(args.after) if args.after else None
    build(Path(args.root), after_dt, args.assignment)
