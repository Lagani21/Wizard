# .WIZ Schema Design

## The problem with a relational schema here

The obvious first design for a file that stores blinks, transcripts, emotions, and scene summaries looks like this:

```
atom ──── speaker       (atom_id FK)
      ├── sentiment     (atom_id FK)
      ├── context_summary (atom_id FK)
      └── No_cut        (atom_id FK, is_blink, is_breath)
```

Seven tables. Every query that wants to ask "show me where PERSON_001 was speaking confidently" has to `JOIN` across at least three of them. That means the database engine scans rows, builds intermediate result sets in memory, and resolves foreign keys — all before returning a single result.

This is fine for a traditional database server that's running continuously and has a query planner, caching, and a buffer pool. It is not fine for a `.wiz` file that is opened, queried, and closed in a single editing session on a laptop. The overhead compounds with every new detection type added.

The deeper problem is that a relational schema like this is closed. If the product adds face identity next month, you add a `face_identity` table. Add shot composition: another table. Each addition extends the JOIN chain for any query that touches multiple features. The schema fights you every time it grows.

---

## The atom as analogy

Think of a physical atom: it has a fixed identity (position in time) and a set of properties — charge, mass, spin — attached to it. You do not describe an atom by saying "go to the charge table and look up charge_id 472". The charge is part of the atom.

A **WizAtom** works the same way. It is a time-ranged segment of video:

```
atom_id:     "3f9a..."
frame_start: 450
frame_end:   495
```

That is the entire fixed structure. Everything the pipeline knows about that window — who was speaking, what they said, whether there was a blink, what the emotional tone was — is attached to the atom as **tags**. The atom is the key; its tags are the values:

```
"3f9a..." → {
    speaker:    "PERSON_001"
    transcript: "we need to talk about Q3 revenue"
    topic:      "Q3 revenue"
    emotion:    "concerned"
    blink:      "true"
}
```

There are no foreign keys. No joining. The atom *is* the object. Its properties travel with it.

---

## Glossary

**Atom** — A temporal unit. Represents one contiguous time window in a video, defined by a start frame and end frame. Every piece of intelligence attaches to an atom.

**Tag** — A key-value pair attached to an atom. Has a `tag_type` (e.g. `speaker`), a `tag_value` (e.g. `PERSON_001`), and an optional `confidence` score between 0 and 1.

**Tag type** — The category of information a tag represents. The current taxonomy:

| tag_type    | tag_value example          | what it means                              |
|-------------|----------------------------|--------------------------------------------|
| `speaker`   | `PERSON_001`               | this speaker is talking in this window     |
| `transcript`| `"we need to review this"` | full speech text for this window           |
| `topic`     | `"Q3 revenue"`             | keyword or phrase extracted from speech    |
| `emotion`   | `"concerned"`              | emotional tone detected in this window     |
| `blink`     | `"true"`                   | a blink event occurs in this window        |
| `breath`    | `"true"`                   | a breath event occurs in this window       |
| `safe_cut`  | `"true"`                   | no blink, no breath — clean edit point     |
| `safe_cut`  | `"pause"`                  | natural silence gap between turns          |
| `caption`   | `"two people at a desk"`   | visual description from VideoMAE           |
| `summary`   | `"CEO discusses margins"`  | LLM scene narrative                        |

**Schema** — The structure of the tables in the database. In the .wiz format this is intentionally minimal: two tables and one metadata table.

**DDL** (Data Definition Language) — The SQL commands that create the table structure (`CREATE TABLE`, `CREATE INDEX`).

**WAL** (Write-Ahead Logging) — A SQLite journal mode that makes concurrent reads safe while a write is in progress. Used by .wiz so the pipeline can write results while the editor is reading them.

**Covering index** — A database index that contains all the columns a query needs, so the query can be answered entirely from the index without touching the main table rows. The `idx_tags_type_value` index on `(tag_type, tag_value)` is the covering index that makes graph search fast.

**WizGraph** — An in-memory index built from a .wiz file at load time. Maps `(tag_type, tag_value)` → set of atom IDs. Queries are Python set intersections. No SQL involved once the graph is built.

---

## The actual schema

```sql
-- Video-level metadata (fps, duration, language, resolution)
CREATE TABLE wiz_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per temporal segment
CREATE TABLE atoms (
    atom_id     TEXT    PRIMARY KEY,
    frame_start INTEGER NOT NULL,
    frame_end   INTEGER NOT NULL
);

-- One row per tag attached to an atom
CREATE TABLE atom_tags (
    tag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id    TEXT    NOT NULL REFERENCES atoms(atom_id) ON DELETE CASCADE,
    tag_type   TEXT    NOT NULL,
    tag_value  TEXT    NOT NULL,
    confidence REAL    NOT NULL DEFAULT 1.0,
    metadata   TEXT    NOT NULL DEFAULT '{}'   -- JSON blob for optional extras
);

-- The entire search layer lives on this index
CREATE INDEX idx_tags_type_value ON atom_tags(tag_type, tag_value);
CREATE INDEX idx_tags_atom_id    ON atom_tags(atom_id);
CREATE INDEX idx_atoms_frames    ON atoms(frame_start, frame_end);
```

That is the complete file format. Three tables, three indexes.

`wiz_meta` stores file-level facts that do not belong to any time window (fps, total frames, video resolution, detected language).

`atoms` stores nothing but frame coordinates. A row in this table means "something happened between frame X and frame Y". What happened is entirely in `atom_tags`.

`atom_tags` is the OO layer. Each row is one property of one atom. An atom with five tags has five rows here. The `metadata` column is a JSON blob reserved for data that does not fit neatly into a single string value — it is not queried directly, just returned alongside results.

---

## Why flat is correct here

A relational schema adds depth: each new feature adds a table, and each query that touches multiple features adds a JOIN. Depth is a cost you pay every single time a query runs.

A flat tag schema adds width: each new feature adds new tag rows to the existing `atom_tags` table. The structure does not change. Existing queries do not get slower. The covering index `(tag_type, tag_value)` works identically for any tag type, whether it was defined at launch or added two years later.

Concretely: adding face identity to a relational schema means writing a new `face_identity` table, a migration script, and updating every query that might want to combine it with existing data. Adding face identity to the .wiz format means the pipeline starts writing `face_id:"FACE_003"` tags. The schema, the search engine, and every existing query require zero changes.

---

## How a multi-dimensional query works

**Relational approach** — "find moments where PERSON_001 talks about revenue and is not mid-blink":

```sql
SELECT a.frame_start, a.frame_end
FROM   atoms a
JOIN   speaker s        ON s.atom_id    = a.atom_id
JOIN   topic t          ON t.atom_id    = a.atom_id
LEFT JOIN no_cut nc     ON nc.atom_id   = a.atom_id
WHERE  s.speaker_id    = 'PERSON_001'
AND    t.keyword       LIKE '%revenue%'
AND    (nc.is_blink IS NULL OR nc.is_blink = FALSE);
```

Three tables joined. SQLite must scan the `speaker` table, resolve every matching atom_id, then look them up in `topic`, then again in `no_cut`. Query cost scales with row count in each table.

**WizGraph approach** — the same query:

```python
speakers   = graph["speaker"]["PERSON_001"]    # set of atom IDs
revenue    = graph["topic"]["revenue"]          # set of atom IDs
blinks     = graph["blink"]["true"]             # set of atom IDs

results = speakers & revenue - blinks           # pure set arithmetic
```

Three dictionary lookups and two set operations. No SQL. No table scans. No JOIN resolution. The result is the answer. Query time is proportional to the size of the result, not the size of the dataset.

---

## Extending the schema

To add a new detection type — face identity, shot composition, camera motion, anything — the process is:

1. The pipeline writes atoms tagged with the new type: `face_id:"FACE_003"`, `shot:"wide"`, `motion:"handheld"`.
2. WizGraph picks up the new tag type automatically at load time.
3. `SearchEngine` gains one new method if you want a named query for it.

No table migrations. No schema versioning. No changes to existing queries. The format grows by adding data, not structure.