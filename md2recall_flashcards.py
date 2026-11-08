#!/usr/bin/env python3
"""md2recall_flashcards.py - turn a folder of Markdown notes into flashcards with a local Ollama model.

Single file, Python 3.8+, standard library only.

  python3 md2recall_flashcards.py FOLDER                       # model: first one Ollama has installed
  python3 md2recall_flashcards.py FOLDER -m qwen3:14b -o cards
  python3 md2recall_flashcards.py FOLDER --formats jsonl,csv,md --max-cards 6 --resume

What it does
  1. Reads every *.md below FOLDER (the output of `pdf2md.py --split` works best).
  2. Cuts each note into chunks along its headings (about --chunk-words words each).
  3. Asks the model for atomic cards per chunk (structured JSON output).
  4. Checks every card against the source: the card must carry a verbatim quote that
     really occurs in the chunk, otherwise it is dropped (--keep-unverified keeps it).
  5. Writes cards.jsonl / cards.csv / one `<note>.cards.md` per note (Obsidian Spaced
     Repetition `front::back` lines).

Card fields (jsonl): front, back, quote, source (file), section (heading path), tags, verified.
Re-run with --resume to skip chunks that are already done.
"""
import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "http://localhost:11434"

SYSTEM = """You write flashcards for studying from a text excerpt.
Rules:
- Use ONLY facts stated in the excerpt. Never add outside knowledge.
- One fact per card (atomic). Prefer definitions, requirements, parameters, steps, causes,
  differences and numbers that are worth remembering. Skip filler, tables of contents,
  page headers, references and anything trivial.
- The question must make sense on its own: name the subject (standard, algorithm, system)
  instead of saying "it" or "the text".
- The answer is short (a phrase or 1-3 sentences). Keep identifiers, numbers and units exact.
- "quote" must be a short passage (at most 25 words) copied word for word from the excerpt
  that supports the card.
- Write the cards in {lang}.
- If the excerpt contains nothing worth a card, return an empty list."""

SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "front": {"type": "string"},
                    "back": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["front", "back", "quote"],
            },
        }
    },
    "required": ["cards"],
}

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE = re.compile(r"^(```|~~~)")
IMAGE = re.compile(r"!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]|!\[[^\]]*\]\(([^)]*)\)")
NOTE_COMMENT = re.compile(r"<!--\s*(?:Page \d+|Note: .*?)-->", re.S)
FORMULA = re.compile(r"<!--\s*formula:\s*(.*?)-->", re.S)


# --------------------------------------------------------------------------- reading

def split_chunks(text, chunk_words):
    """[(heading_path, text)] - cut at headings, then at paragraphs when too long."""
    path, buf, chunks, in_fence = [], [], [], False

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            chunks.append((" > ".join(h for _, h in path), body))

    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()

    out = []
    for head, body in chunks:
        paras, cur, n = re.split(r"\n\s*\n", body), [], 0
        for p in paras:
            w = len(p.split())
            if cur and n + w > chunk_words:
                out.append((head, "\n\n".join(cur)))
                cur, n = [], 0
            cur.append(p)
            n += w
        if cur:
            out.append((head, "\n\n".join(cur)))
    return out


def prepare(chunk):
    """Text sent to the model: images become [image], formula comments become text."""
    images = [m.group(1) or m.group(2) for m in IMAGE.finditer(chunk)]
    chunk = NOTE_COMMENT.sub("", chunk)
    chunk = FORMULA.sub(lambda m: "[formula: " + m.group(1).strip() + "]", chunk)
    chunk = IMAGE.sub("[image]", chunk)
    return re.sub(r"\n{3,}", "\n\n", chunk).strip(), images


# --------------------------------------------------------------------------- ollama

def api(host, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(host.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_model(host):
    models = [m["name"] for m in api(host, "/api/tags", timeout=15).get("models", [])]
    if not models:
        sys.exit("Ollama has no models installed. Pull one first, e.g. `ollama pull qwen3:14b`.")
    return models[0]


def ask(host, model, system, user, num_ctx, temperature, timeout, retries=3):
    payload = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"num_ctx": num_ctx, "temperature": temperature},
        "think": False,
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            msg = api(host, "/api/chat", payload, timeout)["message"]["content"]
            cards = json.loads(msg).get("cards", [])
            return [c for c in cards if isinstance(c, dict)]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"model call failed after {retries} tries: {last}")


# --------------------------------------------------------------------------- checks

def norm(s):
    s = s.casefold().replace("­", "")
    return re.sub(r"[\W_]+", " ", s).strip()


def quote_ok(quote, source):
    q, s = norm(quote), norm(source)
    if len(q) < 8:
        return False
    if q in s:
        return True
    # tolerate small differences: best matching window of the same length
    n = len(q)
    best = 0.0
    step = max(1, n // 4)
    for i in range(0, max(1, len(s) - n + 1), step):
        best = max(best, difflib.SequenceMatcher(None, q, s[i:i + n]).ratio())
        if best >= 0.9:
            return True
    return False


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- output

def write_outputs(cards, out_dir, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "jsonl" in formats:
        with open(out_dir / "cards.jsonl", "w", encoding="utf-8") as f:
            for c in cards:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if "csv" in formats:
        with open(out_dir / "cards.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["front", "back", "source", "section", "tags"])
            for c in cards:
                w.writerow([c["front"], c["back"], c["source"], c["section"], " ".join(c["tags"])])
    if "md" in formats:
        by_note = {}
        for c in cards:
            by_note.setdefault(c["source"], []).append(c)
        for source, items in by_note.items():
            name = re.sub(r"[\\/:*?\"<>|]+", "_", Path(source).stem) + ".cards.md"
            lines = [f"# Cards: {Path(source).stem}", "", "#flashcards", ""]
            section = None
            for c in items:
                if c["section"] != section:
                    section = c["section"]
                    lines += ["", f"## {section}" if section else "## (no heading)", ""]
                lines.append(f"{c['front'].replace('::', ':')}::{c['back'].replace('::', ':')}")
                lines.append("")
            (out_dir / name).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Create flashcards from Markdown notes with Ollama.")
    ap.add_argument("folder", type=Path, help="folder with .md notes (searched recursively)")
    ap.add_argument("-o", "--output", type=Path, help="output folder (default: FOLDER/cards)")
    ap.add_argument("-m", "--model", help="Ollama model (default: first installed model)")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama URL (default {DEFAULT_HOST})")
    ap.add_argument("--lang", default="the same language as the excerpt",
                    help='language of the cards, e.g. "German" (default: same as the source)')
    ap.add_argument("--max-cards", type=int, default=6, help="at most this many cards per chunk")
    ap.add_argument("--chunk-words", type=int, default=800, help="target words per chunk")
    ap.add_argument("--min-words", type=int, default=40, help="skip chunks shorter than this")
    ap.add_argument("--num-ctx", type=int, default=8192, help="context window for the model")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per model call")
    ap.add_argument("--formats", default="jsonl,csv,md", help="comma list of jsonl, csv, md")
    ap.add_argument("--keep-unverified", action="store_true",
                    help="keep cards whose quote is not found in the source")
    ap.add_argument("--tag", action="append", default=[], help="tag added to every card (repeatable)")
    ap.add_argument("--resume", action="store_true", help="skip chunks that were already processed")
    ap.add_argument("--limit", type=int, help="process only the first N chunks (for a test run)")
    args = ap.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"Not a folder: {args.folder}")
    out_dir = args.output or args.folder / "cards"
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    if not formats <= {"jsonl", "csv", "md"}:
        sys.exit("--formats: use jsonl, csv, md")
    try:
        model = args.model or first_model(args.host)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"Cannot reach Ollama at {args.host}: {exc}\nIs `ollama serve` running?")
    system = SYSTEM.format(lang=args.lang)

    notes = sorted(p for p in args.folder.rglob("*.md")
                   if out_dir not in p.parents and not p.name.endswith(".cards.md")
                   and p.stem.lower() not in ("000 index", "index"))
    jobs = []
    for note in notes:
        text = note.read_text(encoding="utf-8", errors="replace")
        for head, chunk in split_chunks(text, args.chunk_words):
            body, images = prepare(chunk)
            if len(body.split()) >= args.min_words:
                jobs.append((note, head, body, images))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        sys.exit("No text chunks found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    state = out_dir / ".chunks.jsonl"  # one line per processed chunk, with its cards
    done, cards = {}, []
    if args.resume and state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec["cards"]
            except (ValueError, KeyError):
                pass
    elif state.exists():
        state.unlink()

    print(f"{len(notes)} notes, {len(jobs)} chunks, model {model}")
    seen, dropped, started = set(), 0, time.time()
    for n, (note, head, body, images) in enumerate(jobs, 1):
        cid = hashlib.sha1(f"{model}|{args.max_cards}|{args.lang}|{note.name}|{head}|{body}"
                           .encode("utf-8")).hexdigest()[:16]
        label = f"[{n}/{len(jobs)}] {note.name[:40]} | {head[-40:]}"
        if cid in done:
            made = done[cid]
            print(f"{label}: cached, {len(made)} cards")
        else:
            user = (f"Source: {note.stem}\nSection: {head or '(none)'}\n"
                    f"Create at most {args.max_cards} cards.\n\nExcerpt:\n{body}")
            try:
                raw = ask(args.host, model, system, user, args.num_ctx, args.temperature, args.timeout)
            except RuntimeError as exc:
                print(f"{label}: ERROR {exc}", file=sys.stderr)
                continue
            made = []
            for c in raw[:args.max_cards]:
                front, back, quote = clean(c.get("front")), clean(c.get("back")), clean(c.get("quote"))
                if not front or not back:
                    continue
                made.append({"front": front, "back": back, "quote": quote,
                             "verified": quote_ok(quote, body)})
            with open(state, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": cid, "cards": made}, ensure_ascii=False) + "\n")
            print(f"{label}: {len(made)} cards")
        for c in made:
            key = norm(c["front"])
            if key in seen:
                continue
            if not c["verified"] and not args.keep_unverified:
                dropped += 1
                continue
            seen.add(key)
            cards.append({**c, "source": str(note.relative_to(args.folder)), "section": head,
                          "tags": list(args.tag),
                          "images": images})

    write_outputs(cards, out_dir, formats)
    print(f"\n{len(cards)} cards written to {out_dir} "
          f"({dropped} dropped: quote not found in the source) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
