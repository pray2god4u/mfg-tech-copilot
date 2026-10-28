#!/usr/bin/env python3
"""Offline, lossless raw-offset chunk index. Never writes to the source/legacy DB.

Public APIs: clean_text(text), embedding_input(ds_name, title, label, raw_text),
load_model(device='mps'), chunk_section(raw, ds_name, title, label, tokenizer,
max_tokens=512, overlap_chars=160), validate_index(path, source, tokenizer),
build_index(source=SOURCE_DB, output=OUTPUT_DB, result_path=RESULT_PATH, ...).
Offsets are Python Unicode character offsets, not bytes. Query callers should
use clean_text(query), the pinned model, and normalized float32 embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
SOURCE_DB = BASE / 'mfg_corpus_20261026_expanded_2.db'
OUTPUT_DB = BASE / 'mfg_chunks_20261028_v2.db'
RESULT_PATH = BASE.parent.parent / 'tmp' / 'chunk_build_result.json'
MODEL = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
VERSION = 'raw-offset-chunks-v2.0'
DIM = 1024
MAX_TOKENS = 512
OVERLAP_CHARS = 160
# Unknown angle-bracket text (including code and Korean table instructions) is
# retained literally. Only recognized HTML markup is stripped.
HTML_TAGS = set('a abbr b blockquote body br caption center code col colgroup dd del div dl dt em font h1 h2 h3 h4 h5 h6 head hr html i img li link meta ol p pre s small span strong sub sup table tbody td th thead title tr u ul'.split())
BLOCK_TAGS = set('br hr p div tr td th li h1 h2 h3 h4 h5 h6 pre blockquote table caption'.split())
IMAGE_MARKER = re.compile(r'!?\[\s*이미지[^\]]*\](?:\([^\n)]*\))?', re.I)
MARKDOWN_IMAGE = re.compile(r'!\[[^\]]*\]\([^\n)]*\)')


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag not in HTML_TAGS:
            self.parts.append(self.get_starttag_text() or '')
        elif tag in BLOCK_TAGS:
            self.parts.append(' ')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag not in HTML_TAGS:
            self.parts.append(f'</{tag}>')
        elif tag in BLOCK_TAGS:
            self.parts.append(' ')

    def handle_data(self, data):
        self.parts.append(data)

    def handle_comment(self, data):
        # Comments can contain relevant code examples; do not discard content.
        self.parts.append(data)


def clean_text(text: str | None) -> str:
    """HTML-to-text with table cells retained; JSON/code content never removed.

    Images and image placeholders have no text content. Markdown punctuation
    and pipe characters are deliberately retained, not interpreted as JSON.
    Raw source remains unmodified in chunks.raw_text.
    """
    raw = MARKDOWN_IMAGE.sub(' ', IMAGE_MARKER.sub(' ', text or ''))
    parser = _TextParser()
    parser.feed(raw)
    parser.close()
    return re.sub(r'\s+', ' ', ''.join(parser.parts)).strip()


def embedding_input(ds_name: str, title: str, label: str | None, raw_text: str) -> str:
    """Deterministic context+body used for both hashing and token counting."""
    return f'{clean_text(ds_name)} — {clean_text(title)}\n{clean_text(label)}\n{clean_text(raw_text)}'


def token_count(text: str, tokenizer) -> int:
    # Includes special tokens; never truncates silently.
    return len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_model(device: str = 'mps'):
    """Resolve only the exact cached revision, with network explicitly disabled."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    local = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = SentenceTransformer(local, revision=REVISION, device=device,
                                local_files_only=True)
    model.max_seq_length = MAX_TOKENS
    model.eval()
    return model, local


def _safe_boundary(raw: str, start: int, end: int) -> int:
    """Prefer paragraph/word boundary; never split a complete HTML tag."""
    if end == len(raw):
        return end
    floor = start + max(1, int((end - start) * .8))
    for separator in ('\n', ' '):
        pos = raw.rfind(separator, floor, end)
        if pos >= floor:
            end = pos + 1
            break
    opening = raw.rfind('<', start, end)
    closing = raw.rfind('>', start, end)
    if opening > closing and raw.find('>', end) >= 0 and opening > start:
        end = opening
    return max(start + 1, end)


def chunk_section(raw: str, ds_name: str, title: str, label: str | None,
                  tokenizer, max_tokens: int = MAX_TOKENS,
                  overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
    """Cover the complete raw section with overlapping token-budgeted slices.

    Empty/image-only sections are excluded without a heading fallback. Context
    is included in the measured budget. Large context fails rather than being
    silently truncated. Content/JSON is never limited by a character cutoff.
    """
    if not re.search(r'\w', clean_text(raw), re.UNICODE):
        return []
    if token_count(embedding_input(ds_name, title, label, ''), tokenizer) >= max_tokens - 32:
        raise ValueError(f'Context leaves insufficient token budget: {title!r} {label!r}')
    result = []
    start, n = 0, len(raw)
    while start < n:
        def count(end):
            return token_count(embedding_input(ds_name, title, label, raw[start:end]), tokenizer)
        if count(n) <= max_tokens:
            end = n
        else:
            lo, hi = start + 1, n
            if count(lo) > max_tokens:
                raise ValueError('A single raw character exceeds the context token budget')
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count(mid) <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            end = _safe_boundary(raw, start, lo)
            # Tokenization is not mathematically monotone; exact final check.
            while end > start and count(end) > max_tokens:
                end -= 1
            if end <= start:
                raise ValueError('Unable to make chunk progress')
        piece = raw[start:end]
        text = embedding_input(ds_name, title, label, piece)
        if not clean_text(piece):
            # An empty trailing span has no embedding meaning, but preserve its
            # raw offsets by absorbing it into the previous meaningful chunk.
            if not result or end != n:
                raise ValueError('Unexpected empty intermediate chunk')
            previous = result[-1]
            previous['end_char'] = n
            previous['raw_text'] = raw[previous['start_char']:n]
            previous['input_text'] = embedding_input(ds_name, title, label, previous['raw_text'])
            previous['token_count'] = token_count(previous['input_text'], tokenizer)
            previous['input_hash'] = hashlib.sha256(previous['input_text'].encode()).hexdigest()
            if previous['token_count'] > max_tokens:
                raise ValueError('Empty tail absorption exceeded token budget')
            break
        result.append(dict(start_char=start, end_char=end, raw_text=piece,
                           input_text=text, input_hash=hashlib.sha256(text.encode()).hexdigest(),
                           token_count=token_count(text, tokenizer)))
        if end == n:
            break
        # Keep strict forward progress even for unusually dense tokenization.
        following = max(start + 1, end - min(overlap_chars, (end - start) // 4))
        # Start outside markup so HTMLParser receives complete table cells/tags.
        opening, closing = raw.rfind('<', end - min(overlap_chars, end), following), raw.rfind('>', 0, following)
        if opening > closing:
            tag_end = raw.find('>', following, end)
            if tag_end >= 0:
                following = tag_end + 1
        start = min(following, end)
    return result


def read_sections(source: Path):
    con = sqlite3.connect(f'{source.resolve().as_uri()}?mode=ro', uri=True)
    try:
        return con.execute('''SELECT s.doc_id, s.sec_idx, s.label, s.tx, d.title, d.ds_name
          FROM sections s JOIN docs d ON d.id=s.doc_id ORDER BY s.doc_id,s.sec_idx''').fetchall()
    finally:
        con.close()


def _reserve_output(requested: Path) -> Path:
    requested.parent.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = requested if i == 1 else requested.with_name(f'{requested.stem}_{i}{requested.suffix}')
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            return candidate
        except FileExistsError:
            i += 1


SCHEMA = '''
CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE chunks(
 chunk_id INTEGER PRIMARY KEY,
 doc_id TEXT NOT NULL,
 sec_idx INTEGER NOT NULL,
 label TEXT,
 start_char INTEGER NOT NULL,
 end_char INTEGER NOT NULL,
 raw_text TEXT NOT NULL,
 embedding BLOB NOT NULL,
 input_hash TEXT NOT NULL,
 token_count INTEGER NOT NULL,
 CHECK(start_char>=0 AND end_char>start_char),
 CHECK(length(embedding)=4096),
 CHECK(token_count<=512 AND token_count>0)
);
CREATE INDEX chunks_section ON chunks(doc_id,sec_idx,start_char);
CREATE INDEX chunks_input_hash ON chunks(input_hash);
'''


def validate_index(path: Path, source: Path, tokenizer, expected_source_sha: str | None = None) -> dict:
    """Read-only integrity/raw coverage/input hash/token/float32 norm audit."""
    import numpy as np
    con = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        meta = dict(con.execute('SELECT key,value FROM meta'))
        actual_sha = sha256_file(source)
        assert actual_sha == meta['source_sha256']
        if expected_source_sha:
            assert actual_sha == expected_source_sha
        assert meta['model'] == MODEL and meta['revision'] == REVISION and int(meta['dim']) == DIM
        rows = read_sections(source)
        by_section = {}
        min_norm, max_norm, max_count = float('inf'), 0., 0
        chunks_n = 0
        for row in con.execute('SELECT chunk_id,doc_id,sec_idx,label,start_char,end_char,raw_text,embedding,input_hash,token_count FROM chunks ORDER BY doc_id,sec_idx,start_char'):
            _, doc, sec, label, start, end, raw, blob, digest, count = row
            vec = np.frombuffer(blob, dtype=np.float32)
            assert vec.shape == (DIM,) and np.isfinite(vec).all()
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.) < 1e-5
            min_norm, max_norm, max_count = min(min_norm, norm), max(max_norm, norm), max(max_count, count)
            by_section.setdefault((doc, sec), []).append((start, end, raw, label, digest, count))
            chunks_n += 1
        included, excluded, raw_chars, coverage = 0, 0, 0, {}
        known = set()
        for doc, sec, label, raw, title, ds in rows:
            raw = raw or ''
            key = (doc, sec)
            known.add(key)
            chunks = by_section.get(key, [])
            meaningful = bool(re.search(r'\w', clean_text(raw), re.UNICODE))
            if not meaningful:
                assert not chunks, f'Empty/image-only section included: {key}'
                excluded += 1
                continue
            assert chunks, f'Section omitted: {key}'
            assert chunks[0][0] == 0 and chunks[-1][1] == len(raw), f'Tail omitted: {key}'
            covered_end = 0
            offsets = []
            for start, end, stored_raw, stored_label, digest, count in chunks:
                assert start <= covered_end and end > covered_end, f'Gap/no progress: {key}'
                assert 0 <= start < end <= len(raw) and stored_raw == raw[start:end]
                assert stored_label == label
                text = embedding_input(ds, title, label, stored_raw)
                assert clean_text(stored_raw), f'Empty chunk: {key}'
                assert hashlib.sha256(text.encode()).hexdigest() == digest
                assert token_count(text, tokenizer) == count and count <= int(meta['max_tokens'])
                covered_end = end
                offsets.append([start, end])
            included += 1
            raw_chars += len(raw)
            if doc.startswith('71802') and sec == 0:
                coverage[f'{doc}:{sec}'] = dict(raw_chars=len(raw), chunks=len(chunks), offsets=offsets,
                                                covered_to=covered_end, no_gaps=True)
        assert set(by_section) <= known
        return dict(integrity_check='ok', source_sha256=actual_sha, source_sections=len(rows),
                    included_sections=included, excluded_empty_or_image_sections=excluded,
                    chunks=chunks_n, covered_raw_chars=raw_chars, max_observed_tokens=max_count,
                    embedding_norm_min=min_norm, embedding_norm_max=max_norm,
                    finite_embeddings=True, raw_slice_exact=True, coverage_no_gaps=True,
                    coverage_71802_sec0=coverage)
    finally:
        con.close()


def build_index(source: Path = SOURCE_DB, output: Path = OUTPUT_DB,
                result_path: Path = RESULT_PATH, device: str = 'mps',
                batch_size: int = 16, max_tokens: int = MAX_TOKENS) -> dict:
    import numpy as np
    if not 64 <= max_tokens <= MAX_TOKENS:
        raise ValueError('max_tokens must be between 64 and 512')
    t0 = time.monotonic()
    source, output, result_path = Path(source), Path(output), Path(result_path)
    if source.resolve() == output.resolve():
        raise ValueError('Output must not be source')
    source_sha = sha256_file(source)
    rows = read_sections(source)
    model, local = load_model(device)
    model.max_seq_length = max_tokens
    tokenizer = model.tokenizer
    chunks = []
    for i, (doc, sec, label, raw, title, ds) in enumerate(rows):
        for chunk in chunk_section(raw or '', ds, title, label, tokenizer, max_tokens):
            chunk.update(doc_id=doc, sec_idx=sec, label=label)
            chunks.append(chunk)
        if (i + 1) % 500 == 0:
            print(f'chunked sections={i+1}/{len(rows)} chunks={len(chunks)}', flush=True)
    chunking_seconds = time.monotonic() - t0
    print(f'encoding chunks={len(chunks)} device={device} batch_size={batch_size}', flush=True)
    encode_start = time.monotonic()
    blocks = []
    for start in range(0, len(chunks), 256):
        texts = [c['input_text'] for c in chunks[start:start + 256]]
        cache_key = hashlib.sha256((MODEL + REVISION + source_sha + str(max_tokens) +
                    json.dumps([c['input_hash'] for c in chunks[start:start+256]])).encode()).hexdigest()
        cache_dir = BASE.parent.parent / 'tmp' / 'mfg_chunk_checkpoints'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (cache_key + '.npy')
        if cache_file.exists():
            emb = np.load(cache_file, allow_pickle=False)
            print(f'resumed checkpoint={start}', flush=True)
        else:
            emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True, device=device)
            emb = np.asarray(emb, dtype=np.float32)
            assert emb.shape == (len(texts), DIM) and np.isfinite(emb).all()
            assert np.all(np.abs(np.linalg.norm(emb, axis=1) - 1.) < 1e-5)
            with cache_file.open('xb') as cache_stream:
                np.save(cache_stream, emb, allow_pickle=False)
                cache_stream.flush()
                os.fsync(cache_stream.fileno())
        emb = np.asarray(emb, dtype=np.float32)
        assert emb.shape == (len(texts), DIM) and np.isfinite(emb).all()
        norms = np.linalg.norm(emb, axis=1)
        assert np.all(np.abs(norms - 1.) < 1e-5)
        blocks.append(emb)
        print(f'encoded={min(start+256,len(chunks))}/{len(chunks)} elapsed_seconds={time.monotonic()-encode_start:.1f}', flush=True)
    vectors = np.concatenate(blocks, axis=0)
    encode_seconds = time.monotonic() - encode_start
    assert sha256_file(source) == source_sha, 'Source changed during encoding'
    # Exclusive file creation after all embeddings pass validation. One explicit
    # transaction commits schema, rows, and metadata together, never a partial DB.
    actual_output = _reserve_output(output)
    con = sqlite3.connect(actual_output)
    built_at = datetime.now(timezone.utc).isoformat()
    meta = dict(model=MODEL, revision=REVISION, dim=str(DIM), source_sha256=source_sha,
                source_db=str(source.resolve()), version=VERSION, built_at=built_at,
                max_tokens=str(max_tokens), overlap_chars=str(OVERLAP_CHARS),
                device=device, batch_size=str(batch_size), dtype='float32', normalized='true',
                model_path=local, offset_unit='unicode_char', input_format='ds_name — title\\nlabel\\nclean_text(raw_text)',
                chunks=str(len(chunks)), source_sections=str(len(rows)))
    try:
        con.executescript('BEGIN IMMEDIATE;\n' + SCHEMA)
        con.executemany('INSERT INTO meta VALUES (?,?)', meta.items())
        con.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)',
                        ((i + 1, c['doc_id'], c['sec_idx'], c['label'], c['start_char'], c['end_char'],
                          c['raw_text'], vectors[i].tobytes(), c['input_hash'], c['token_count'])
                         for i, c in enumerate(chunks)))
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0] == len(chunks)
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()
    validation = validate_index(actual_output, source, tokenizer, source_sha)
    stats = dict(status='BUILD_OK', output_db=str(actual_output.resolve()),
                 output_bytes=actual_output.stat().st_size, source_db=str(source.resolve()),
                 model=MODEL, revision=REVISION, model_path=local, version=VERSION,
                 built_at=built_at, dim=DIM, device=device, batch_size=batch_size,
                 max_tokens=max_tokens, overlap_chars=OVERLAP_CHARS,
                 chunking_seconds=round(chunking_seconds, 3), encode_seconds=round(encode_seconds, 3),
                 total_seconds=round(time.monotonic() - t0, 3), **validation)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DB)
    parser.add_argument('--output', type=Path, default=OUTPUT_DB)
    parser.add_argument('--result', type=Path, default=RESULT_PATH)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    args = parser.parse_args()
    build_index(args.source, args.output, args.result, args.device, args.batch_size, args.max_tokens)


if __name__ == '__main__':
    main()
