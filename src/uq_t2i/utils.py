import hashlib
import time
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id VARCHAR PRIMARY KEY,
    text      VARCHAR NOT NULL,
    grp       VARCHAR
);
CREATE TABLE IF NOT EXISTS prompt_sets (
    set_id    VARCHAR,
    prompt_id VARCHAR,
    kind      VARCHAR NOT NULL,
    role      VARCHAR,
    PRIMARY KEY (set_id, prompt_id)
);
CREATE TABLE IF NOT EXISTS images (
    image_id  VARCHAR PRIMARY KEY,
    prompt_id VARCHAR NOT NULL,
    seed      BIGINT NOT NULL,
    run       VARCHAR NOT NULL,
    model     VARCHAR,
    steps     INTEGER,
    guidance  DOUBLE,
    path      VARCHAR
);
CREATE TABLE IF NOT EXISTS captions (
    image_id    VARCHAR,
    captioner   VARCHAR,
    instruction VARCHAR,
    caption     VARCHAR NOT NULL,
    PRIMARY KEY (image_id, captioner, instruction)
);
CREATE TABLE IF NOT EXISTS metrics (
    image_id VARCHAR,
    name   VARCHAR,
    metric   VARCHAR,
    value    DOUBLE,
    PRIMARY KEY (image_id, metric, name)
);
"""

COLUMNS = {
    "prompts": (["prompt_id"], ["text"], ["grp"]),
    "prompt_sets": (["set_id", "prompt_id"], ["kind"], ["role"]),
    "images": (
        ["image_id"],
        ["prompt_id", "seed", "run"],
        ["model", "steps", "guidance", "path"],
    ),
    "captions": (["image_id", "captioner", "instruction"], ["caption"], []),
    "metrics": (["image_id", "metric", "name"], ["value"], []),
}
SET_KINDS = ("ambiguity", "specification", "paraphrase", "ladder")


def connect(path=None, read_only=False, wait=60):
    """Opens the database and creates missing tables. If another process has the file open, tries again for up to `wait` seconds."""
    if path is None:
        raise ValueError("path must be specified")
    path = Path(path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait
    while True:
        try:
            con = duckdb.connect(str(path), read_only=read_only)
            break
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or time.monotonic() > deadline:
                raise
            time.sleep(0.5)
    if not read_only:
        con.execute(SCHEMA)
    return con


def prompt_id(text: str) -> str:
    """Stable id from the prompt text: the same on every machine and in every run,
    and unchanged when prompts.py is reordered."""
    return "p" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def image_id(pid: str, seed: int, run: str = "main") -> str:
    return f"{run}_{pid}_s{seed}"


def _upsert(con, table, rows):
    """Within one call, later rows win. Optional columns can be left out (stored as
    NULL); columns the table does not have are ignored."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return 0
    key, required, optional = COLUMNS[table]
    missing = [c for c in key + required if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")
    df = df.reindex(columns=key + required + optional).drop_duplicates(
        subset=key, keep="last"
    )
    cols = ", ".join(df.columns)
    cur = con.cursor()
    try:
        cur.register("_rows", df)
        cur.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _rows")
    finally:
        cur.close()
    return len(df)


def add_prompts(con, rows):
    return _upsert(con, "prompts", rows)


def add_prompt_sets(con, rows):
    return _upsert(con, "prompt_sets", rows)


def add_images(con, rows):
    return _upsert(con, "images", rows)


def add_captions(con, rows):
    return _upsert(con, "captions", rows)


def add_metrics(con, rows):
    return _upsert(con, "metrics", rows)


def register_prompts(con):
    """Generate and register the prompts and prompt sets from prompts.py."""
    try:
        from .prompts import (
            ambiguous_prompts,
            generate_prompts_list,
            paraphrases,
            specificity_ladders,
            underspecified_prompts,
        )
    except ImportError:
        from prompts import (
            ambiguous_prompts,
            generate_prompts_list,
            paraphrases,
            specificity_ladders,
            underspecified_prompts,
        )

    add_prompts(
        con,
        [
            {"prompt_id": prompt_id(t), "text": t, "grp": g}
            for g, t in generate_prompts_list()
        ],
    )

    sets = []

    def add_set(sid, kind, members):
        sets.extend(
            {"set_id": sid, "kind": kind, "prompt_id": prompt_id(p), "role": r}
            for p, r in members
        )

    for kind, groups in (
        ("ambiguity", ambiguous_prompts),
        ("specification", underspecified_prompts),
    ):
        for base, variants in groups.items():
            add_set(
                f"{kind[:4]}_{prompt_id(base)}",
                kind,
                [(base, "base")] + [(v, "variant") for v in variants],
            )
    for group in paraphrases:
        add_set(
            f"para_{prompt_id(group[0])}", "paraphrase", [(p, "variant") for p in group]
        )
    for ladder in specificity_ladders:
        add_set(
            f"ladd_{prompt_id(ladder[-1])}",
            "ladder",
            [(p, f"step_{i}") for i, p in enumerate(ladder)],
        )

    kinds = ", ".join(f"'{k}'" for k in SET_KINDS)
    con.execute(f"DELETE FROM prompt_sets WHERE kind IN ({kinds})")
    add_prompt_sets(con, sets)


def prompts_without_images(con, run="main"):
    """Prompts that don't have any generated images in the specified run."""
    return con.execute(
        """
        SELECT p.* FROM prompts p
        WHERE NOT EXISTS (SELECT 1 FROM images i
                          WHERE i.prompt_id = p.prompt_id AND i.run = ?)
        ORDER BY p.prompt_id""",
        [run],
    ).df()


def images_without_caption(con, captioner, instruction):
    """Images that still need a caption from this captioner with this instruction."""
    return con.execute(
        """
        SELECT i.image_id, i.path FROM images i
        WHERE NOT EXISTS (SELECT 1 FROM captions c
                          WHERE c.image_id = i.image_id AND c.captioner = ? AND c.instruction = ?)
        ORDER BY i.image_id""",
        [captioner, instruction],
    ).df()


def get_image_path(con, image_id):
    """Returns the path of an image given its image_id."""
    row = con.execute("SELECT path FROM images WHERE image_id = ?", [image_id]).df()
    if row.empty:
        raise KeyError(image_id)
    return row.iloc[0]["path"]


def caption_without_punc(con, captioner):
    """Captions that still needs PUNC from this captioner."""
    return con.execute(
        """
        SELECT c.image_id, p.text, c.caption FROM captions c
        JOIN prompts p ON c.image_id = p.image_id
        WHERE NOT EXISTS (SELECT 1 FROM metrics m
                          WHERE m.image_id = c.image_id AND m.name = 'punc') AND c.captioner = ?
        ORDER BY c.image_id""",
        [captioner],
    ).df()


def get_prompt_from_image_id(con, image_id):
    """Returns the prompt text associated with a given image_id."""
    row = con.execute(
        """
        SELECT p.text FROM prompts p
        JOIN images i ON p.prompt_id = i.prompt_id
        WHERE i.image_id = ?""",
        [image_id],
    ).df()
    if row.empty:
        raise KeyError(image_id)
    return row.iloc[0]["text"]


def images_without_vqascore(con):
    """Images that still need a VQA score from this model."""
    return con.execute("""
        SELECT i.image_id, p.text, i.path FROM images i
        JOIN prompts p ON i.prompt_id = p.prompt_id
        WHERE NOT EXISTS (SELECT 1 FROM metrics m
                          WHERE m.image_id = i.image_id AND m.name = 'vqa')
        ORDER BY i.image_id""").df()
