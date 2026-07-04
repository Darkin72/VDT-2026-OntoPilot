import argparse
import json
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def default_repo_dir():
    container_repo = Path('/app')
    if (container_repo / 'Ontology').exists():
        return container_repo
    return Path(__file__).resolve().parents[2]


REPO_DIR = default_repo_dir()
DEFAULT_DOCUMENTS = REPO_DIR / 'Ontology' / 'normalized' / 'embedding_documents.jsonl'
DEFAULT_DB = REPO_DIR / 'Ontology' / 'normalized' / 'embedding_lookup.sqlite'


def normalize_term(value):
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    return ' '.join(value.replace('_', ' ').casefold().split())


def candidate_terms(document):
    terms = set()
    for key in ('label', 'curie', 'uri', 'id', 'parent_id'):
        value = document.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        terms.add(value)
        terms.add(value.rstrip('/#').rsplit('/', 1)[-1].rsplit('#', 1)[-1])
        if ':' in value:
            terms.add(value.split(':', 1)[1])

    payload = document.get('payload')
    labels_json = payload.get('labels_json') if isinstance(payload, dict) else None
    if isinstance(labels_json, str):
        try:
            labels = json.loads(labels_json)
        except json.JSONDecodeError:
            labels = []
        terms.update(label for label in labels if isinstance(label, str))

    return {normalize_term(term) for term in terms if normalize_term(term)}


def setup_db(conn, recreate):
    if recreate:
        conn.executescript('DROP TABLE IF EXISTS terms; DROP TABLE IF EXISTS documents;')
    conn.executescript(
        '''
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            line_number INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL,
            kind TEXT,
            label TEXT,
            curie TEXT,
            uri TEXT
        );
        CREATE TABLE IF NOT EXISTS terms (
            term TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            PRIMARY KEY (term, doc_id)
        );
        '''
    )


def insert_batch(conn, documents, terms):
    conn.executemany('INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)', documents)
    conn.executemany('INSERT OR IGNORE INTO terms VALUES (?, ?)', terms)


def make_progress(documents_path, limit):
    if tqdm is None:
        return None
    total = None if limit else documents_path.stat().st_size
    return tqdm(
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc='Building lookup DB',
        dynamic_ncols=True,
        mininterval=1,
        file=sys.stdout,
    )


def build(documents_path, db_path, batch_size, recreate, limit):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    setup_db(conn, recreate)
    document_rows = []
    term_rows = []
    indexed = 0
    started = time.monotonic()
    progress = make_progress(documents_path, limit)
    try:
        with documents_path.open('rb') as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if progress is not None:
                    progress.update(len(raw_line))
                if not raw_line.strip():
                    continue
                indexed += 1
                document = json.loads(raw_line)
                doc_id = str(document.get('id') or '').strip()
                if not doc_id:
                    raise ValueError(f'Line {indexed} is missing id')
                document_rows.append((
                    doc_id,
                    indexed,
                    offset,
                    str(document.get('kind') or ''),
                    str(document.get('label') or ''),
                    str(document.get('curie') or ''),
                    str(document.get('uri') or ''),
                ))
                term_rows.extend((term, doc_id) for term in candidate_terms(document))
                if len(document_rows) >= batch_size:
                    insert_batch(conn, document_rows, term_rows)
                    conn.commit()
                    document_rows.clear()
                    term_rows.clear()
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = indexed / elapsed
                    if progress is not None:
                        progress.set_postfix_str(f'docs={indexed:,} docs/s={rate:,.0f}')
                    else:
                        print(f'Indexed {indexed:,} docs at {rate:,.0f} docs/s', flush=True)
                if limit and indexed >= limit:
                    break
        if document_rows:
            insert_batch(conn, document_rows, term_rows)
        conn.executescript('''
        CREATE INDEX IF NOT EXISTS idx_documents_label ON documents(label);
        CREATE INDEX IF NOT EXISTS idx_documents_curie ON documents(curie);
        CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind);
        CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri);
        ''')
        conn.commit()
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.execute('PRAGMA journal_mode = DELETE')
        conn.commit()
    finally:
        if progress is not None:
            elapsed = max(time.monotonic() - started, 0.001)
            progress.set_postfix_str(f'docs={indexed:,} docs/s={indexed / elapsed:,.0f}')
            progress.close()
        conn.close()
    print(f'Done. Indexed {indexed:,} docs into {db_path} in {time.monotonic() - started:.1f}s')


def query(db_path, documents_path, text, limit):
    term = normalize_term(text)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        '''
        SELECT d.* FROM terms t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE t.term = ? ORDER BY d.line_number LIMIT ?
        ''',
        (term, limit),
    ).fetchall()
    conn.close()
    if not rows:
        print(f'No match for {text!r} normalized as {term!r}')
        return
    with documents_path.open('rb') as handle:
        for row in rows:
            handle.seek(row['byte_offset'])
            document = json.loads(handle.readline())
            print(json.dumps({'line_number': row['line_number'], 'byte_offset': row['byte_offset'], 'document': document}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='Build/query a SQLite lookup DB for embedding_documents.jsonl.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    build_parser = subparsers.add_parser('build')
    build_parser.add_argument('--documents', type=Path, default=DEFAULT_DOCUMENTS)
    build_parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    build_parser.add_argument('--batch-size', type=int, default=50000)
    build_parser.add_argument('--limit', type=int, default=0)
    build_parser.add_argument('--recreate', action='store_true')
    query_parser = subparsers.add_parser('query')
    query_parser.add_argument('text')
    query_parser.add_argument('--documents', type=Path, default=DEFAULT_DOCUMENTS)
    query_parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    query_parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.documents, args.db, max(1, args.batch_size), args.recreate, max(0, args.limit))
    else:
        query(args.db, args.documents, args.text, max(1, args.limit))


if __name__ == '__main__':
    main()
