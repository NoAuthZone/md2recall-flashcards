# md2recall_flashcards

Turns a folder of Markdown notes into flashcards with a **local** [Ollama](https://ollama.com) model. One Python file, standard library only (Python 3.8+), runs on Windows, Linux and macOS. Nothing leaves your machine.

It is the second step after [pdf2md](README.md): convert the PDF to Markdown notes first, then create cards from them.

## Features

- **Chunked by headings**: each note is cut along its headings (about 800 words per chunk), so the model sees one topic at a time and the card keeps its section path.
- **Structured output**: the model must answer in a JSON schema (`front`, `back`, `quote`), no free-text parsing.
- **Source check**: every card carries a verbatim quote. If the quote cannot be found in the chunk, the card is dropped. This removes most invented answers.
- **Atomic cards**: one fact per card, the question names its subject, short answers, exact numbers and identifiers.
- **Resumable**: with `--resume` finished chunks are skipped, so an interrupted run on a big folder continues where it stopped.
- **Formats**: `cards.jsonl`, `cards.csv` and one `<note>.cards.md` per note (`front::back` lines for the Obsidian Spaced Repetition plugin).
- **Works with pdf2md output**: formulas (`<!-- formula: ... -->`) are passed to the model as text, images as `[image]`; page and note comments are removed.

## Requirements

- Python 3.8 or newer. No packages to install.
- A running Ollama (`ollama serve`) with at least one model, e.g. `ollama pull qwen3:14b`.

## Usage

```bash
# 1. PDF -> notes (one note per chapter)
python3 pdf2md.py book.pdf -f md --split 2

# 2. notes -> cards
python3 md2recall_flashcards.py book -m qwen3:14b --lang German --tag book

# short test run first: only the first 5 chunks
python3 md2recall_flashcards.py book --limit 5

# continue an interrupted run
python3 md2recall_flashcards.py book -m qwen3:14b --resume
```

Without `-m` the first model that Ollama lists is used. Output goes to `<folder>/cards/`.

## Options

| Option | Meaning |
|---|---|
| `folder` | Folder with `.md` notes, searched recursively (`000 Index.md` and `*.cards.md` are skipped) |
| `-o`, `--output DIR` | Output folder (default `<folder>/cards`) |
| `-m`, `--model NAME` | Ollama model (default: first installed model) |
| `--host URL` | Ollama address (default `http://localhost:11434`) |
| `--lang TEXT` | Language of the cards, e.g. `German` (default: same as the excerpt) |
| `--max-cards N` | At most N cards per chunk (default 6) |
| `--chunk-words N` | Target words per chunk (default 800) |
| `--min-words N` | Skip chunks shorter than this (default 40) |
| `--num-ctx N` | Context window handed to the model (default 8192) |
| `--temperature X` | Sampling temperature (default 0.2) |
| `--timeout S` | Seconds per model call (default 600) |
| `--formats LIST` | Comma list of `jsonl`, `csv`, `md` (default all three) |
| `--keep-unverified` | Keep cards whose quote is not found in the source |
| `--tag TAG` | Tag added to every card (repeatable) |
| `--resume` | Skip chunks that were already processed |
| `--limit N` | Process only the first N chunks (test run) |

## Output

`cards.jsonl` has one card per line (illustrative example):

```json
{"front": "Which key length does BSI TR-02102-1 recommend for RSA?",
 "back": "At least 3000 bits.",
 "quote": "a key length of at least 3000 bits is recommended",
 "verified": true,
 "source": "004 Asymmetric encryption.md",
 "section": "Asymmetric encryption > RSA",
 "tags": ["book"],
 "images": []}
```

- `source` is the note, `section` its heading path, `images` the image files that appear in the chunk (for adding the picture to a card later).
- `cards.csv` uses `;` as separator and has the columns `front`, `back`, `source`, `section`, `tags`.
- `<note>.cards.md` groups the cards by section, one `front::back` line each, with the tag `#flashcards` at the top.
- `.chunks.jsonl` in the output folder is the progress file used by `--resume`. Without `--resume` a run starts from scratch and replaces it.

If your flashcard tool needs another import format, only the function `write_outputs` has to change.

## Tips

- **Chunk size**: 500 to 1500 words per chunk works for most models. Smaller local models make shallow cards from long chunks; raise `--num-ctx` (e.g. 16384) together with `--chunk-words` when you use a larger model.
- **Model**: 8-14B models are enough for plain text. Tables and German technical text profit from a larger model; try two models on the same chapter before running a whole folder.
- **Quality control**: look at `verified` and `quote` when you review cards. A card is only as good as the note it comes from, so check the Markdown (pdf2md `--review`) first.
- **Language**: set `--lang German` if the notes are English but you want German cards (the quote stays in the source language).
- **Images**: a text model cannot see them. Captions are part of the Markdown text; diagrams need a vision model or manual cards.

## Limits

- The quote check proves that the quote exists in the chunk, not that the answer is correct. Read the cards before you learn from them.
- The result depends on the model. Small models sometimes skip content or write vague questions.
- Duplicates are removed by identical question text only.
- `think` is switched off for models that support it, to keep answers short and fast; older Ollama versions ignore that setting.

## Note

Independent project, not affiliated with or endorsed by Ollama or Obsidian. The names are used only to describe compatibility.
