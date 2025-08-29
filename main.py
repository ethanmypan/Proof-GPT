"""
extract_questions_with_llm.py
-----------------------------
Parse a JSON file of {"question": ..., "answer": ...} objects, pass each pair to an
LLM (OpenAI Chat Completions API by default), and collect ONLY the questions that the LLM
outputs after "understanding" both the question and the correct answer.

Usage:
  pip install --upgrade openai tqdm python-dotenv
  # Create a .env file in this directory with:
  # OPENAI_API_KEY="sk-..."

  python main.py \
    --input /path/to/homework0_qa.json \
    --output questions.json

Notes:
  - Input format: a JSON LIST of objects, each with keys "question" and "answer".
  - Output format: JSON with {"questions": [ "...", "...", ... ]}.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError(
        "Required libraries not found. Please run: pip install --upgrade openai python-dotenv"
    ) from e

# Load environment variables from the .env file in the current directory.
# This line is now correctly un-indented and will always run.
load_dotenv()

try:
    from tqdm import tqdm  # progress bar (optional)
except ImportError:
    # If tqdm is not installed, create a dummy function that just returns the iterator
    tqdm = lambda x, **kwargs: x


# OpenAI client helpers
def _get_openai_client() -> OpenAI:
    """Initializes and returns the OpenAI client, ensuring the SDK is installed."""
    # The OpenAI client automatically reads the OPENAI_API_KEY from the environment,
    # which was loaded by load_dotenv().
    return OpenAI()


def call_llm(
    client: OpenAI, model: str, system: str, prompt: str, temperature: float = 0.0
) -> Dict[str, str]:
    """
    Call the OpenAI Chat Completions API and return {"text": str, "request_id": str|None}.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    if not resp.choices:
        raise ValueError("LLM response was empty or malformed.")

    text = resp.choices[0].message.content
    if text is None:
        text = ""  # Ensure we handle cases where content might be null

    req_id = resp.id
    return {"text": text.strip(), "request_id": req_id}


# -----------------------------
# Core logic
# -----------------------------
DEFAULT_SYSTEM = "You are a careful, deterministic parsing assistant."
DEFAULT_TASK_BASE = (
    "You will receive a question and its correct answer. Read both so you fully understand the item.\n"
    "Your job: OUTPUT ONLY THE QUESTION TEXT, with no extra words, lists, quotes, or formatting.\n"
    "If the question text is already clear, return it verbatim.\n"
)
DEFAULT_TASK_NO_REWRITE = DEFAULT_TASK_BASE + "Do NOT rewrite or paraphrase."
DEFAULT_TASK_ALLOW_REWRITE = (
    DEFAULT_TASK_BASE
    + "If wording is unclear, minimally rewrite for clarity but preserve semantics."
)


def build_prompt(question: str, answer: str, allow_rewrite: bool) -> str:
    """Constructs the final prompt to be sent to the LLM."""
    header = DEFAULT_TASK_ALLOW_REWRITE if allow_rewrite else DEFAULT_TASK_NO_REWRITE
    return (
        f"{header}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== CORRECT ANSWER ===\n{answer}\n\n"
        f"Return ONLY the final question text."
    )


def iter_qa(data: Any) -> Iterable[Dict[str, str]]:
    """
    Safely iterates through the input data, yielding valid Q&A pairs.
    """
    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must be a LIST of objects with keys 'question' and 'answer'."
        )
    for item in data:
        if not isinstance(item, dict):
            continue
        q, a = item.get("question"), item.get("answer")
        if q is not None and a is not None:
            yield {"question": str(q), "answer": str(a)}


def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(
        description="Parse {question, answer} JSON, send to an LLM, and output all questions."
    )
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to JSON file (list of {question, answer}).",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("questions.json"),
        help="Path to write JSON with {'questions': [...]} (default: questions.json).",
    )
    ap.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model id (default: gpt-4o-mini).",
    )
    ap.add_argument(
        "--system",
        "-s",
        type=str,
        default=DEFAULT_SYSTEM,
        help="System/instructions string for the LLM.",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)."
    )
    ap.add_argument(
        "--rewrite",
        action="store_true",
        help="Allow minimal rewrites for clarity (default: verbatim).",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print prompts instead of calling the API."
    )
    ap.add_argument(
        "--dedupe", action="store_true", help="Deduplicate identical question outputs."
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[error] Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[error] Failed to read JSON from {args.input}: {e}", file=sys.stderr)
        sys.exit(2)

    # Initialize the OpenAI client once.
    client = None
    if not args.dry_run:
        try:
            client = _get_openai_client()
        except Exception as e:
            print(f"[error] Failed to initialize OpenAI client: {e}", file=sys.stderr)
            sys.exit(1)

    questions: List[str] = []
    for idx, qa in enumerate(tqdm(list(iter_qa(data)), desc="Processing Q&A pairs")):
        prompt = build_prompt(qa["question"], qa["answer"], allow_rewrite=args.rewrite)

        if args.dry_run:
            print("\n--- DRY RUN (idx", idx, ") ---")
            print("System:", args.system)
            print("Prompt:", prompt)
            questions.append(qa["question"])  # Add original question in dry run
            continue

        try:
            result = call_llm(
                client=client,
                model=args.model,
                system=args.system,
                prompt=prompt,
                temperature=args.temperature,
            )
            q_text = result["text"].strip()
            if q_text:
                questions.append(q_text)
        except Exception as e:
            # Don't crash the entire run—log and continue
            print(f"\n[warn] idx={idx} LLM call failed: {e}", file=sys.stderr)
            continue

    # Optionally de-duplicate while preserving order
    if args.dedupe:
        seen = set()
        deduped = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        questions = deduped

    # Write output
    out_payload = {"questions": questions}
    try:
        args.output.write_text(
            json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[done] Wrote {len(questions)} question(s) to {args.output}")
    except Exception as e:
        print(f"\n[error] Failed to write output to {args.output}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()