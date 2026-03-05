import argparse
import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI


SYSTEM_PROMPT = (
    "You are a helpful teacher. Solve the multiple-choice question and provide "
    "a concise Chinese explanation. End with the final selected option letter."
)


class JsonArrayAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.has_items = False
        self._init_file()

    def _init_file(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.write_text("[\n]\n", encoding="utf-8")
            self.has_items = False
            return
        content = self.path.read_text(encoding="utf-8").strip()
        if not content:
            self.path.write_text("[\n]\n", encoding="utf-8")
            self.has_items = False
            return
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError(f"{self.path} must be a JSON array.")
        self.has_items = len(parsed) > 0

    def append(self, item: dict[str, str]) -> None:
        with self.path.open("r+", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            while pos >= 0:
                f.seek(pos)
                ch = f.read(1)
                if not ch.isspace():
                    break
                pos -= 1
            if pos < 0 or ch != "]":
                raise ValueError(f"{self.path} is not a valid JSON array.")

            f.seek(pos)
            f.truncate()
            if self.has_items:
                f.write(",\n")
            f.write(json.dumps(item, ensure_ascii=False, indent=2))
            f.write("\n]\n")
            f.flush()
            os.fsync(f.fileno())
            self.has_items = True


def _get_sample_id(sample: dict[str, Any], idx: int) -> str:
    for key in ("question_id", "id", "problem_id", "uuid"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(idx)


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS completed_samples (
            sample_id TEXT PRIMARY KEY,
            dataset_index INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _load_completed_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT sample_id FROM completed_samples").fetchall()
    return {row[0] for row in rows}


def _format_input(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    options = sample.get("options", [])
    if not isinstance(options, list):
        options = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [
        f"{letters[idx]}. {str(option).strip()}" for idx, option in enumerate(options)
    ]
    category = str(sample.get("category", "")).strip()
    category_line = f"Subject：{category}\n" if category else ""
    return (
        f"{category_line}Question：{question}\n\n"
        f"Options：\n" + "\n".join(option_lines) + "\n\nPlease solve the question step by step."
    )


def _build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    input_text = _format_input(sample)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    sample: dict[str, Any],
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    instruction = "You are a helpful assistant."
    input_text = _format_input(sample)
    messages = _build_messages(sample)

    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=2048,
        )
        output = (response.choices[0].message.content or "").strip()

    print(f"[gen {idx + 1}/{total}] done")
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


async def _writer(
    queue: "asyncio.Queue[tuple[str, int, dict[str, str] | None, Exception | None]]",
    appender: JsonArrayAppender,
    conn: sqlite3.Connection,
    total_pending: int,
) -> tuple[int, int]:
    success = 0
    failed = 0
    processed = 0
    while processed < total_pending:
        sample_id, idx, record, err = await queue.get()
        processed += 1
        if err is not None or record is None:
            failed += 1
            print(f"[write {idx + 1}] failed: {err}")
            continue
        appender.append(record)
        conn.execute(
            """
            INSERT OR IGNORE INTO completed_samples
            (sample_id, dataset_index, created_at, record_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                sample_id,
                idx,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(record, ensure_ascii=False),
            ),
        )
        conn.commit()
        success += 1
        print(f"[write {processed}/{total_pending}] appended sample_id={sample_id}")
    return success, failed


async def _worker(
    queue: "asyncio.Queue[tuple[str, int, dict[str, str] | None, Exception | None]]",
    client: AsyncOpenAI,
    model: str,
    sample: dict[str, Any],
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> None:
    sample_id = _get_sample_id(sample, idx)
    try:
        record = await _generate_one(client, model, sample, idx, total, semaphore)
        await queue.put((sample_id, idx, record, None))
    except Exception as e:
        await queue.put((sample_id, idx, None, e))


async def _run(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_cache_dir = base_dir / "dataset"
    output_path = base_dir / args.output
    db_path = base_dir / args.sqlite
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    model = args.model or os.getenv("MODEL_NAME")

    if not base_url:
        raise ValueError("Missing BASE_URL in .env")
    if not api_key:
        raise ValueError("Missing API_KEY in .env")
    if not model:
        raise ValueError("Missing model name. Set MODEL_NAME in .env or pass --model")

    print("Loading MMLU-Pro test split...")
    ds = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
        cache_dir=str(dataset_cache_dir),
    )

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    total = len(ds)
    conn = _init_db(db_path)
    completed_ids = _load_completed_ids(conn)
    pending_indices = [
        i for i in range(total) if _get_sample_id(ds[i], i) not in completed_ids
    ]

    print(f"Total samples: {total}")
    print(f"Completed in sqlite: {len(completed_ids)}")
    print(f"Pending to generate: {len(pending_indices)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_path}")
    print(f"SQLite: {db_path}")

    if not pending_indices:
        print("All samples already completed. Nothing to do.")
        conn.close()
        return

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    appender = JsonArrayAppender(output_path)
    queue: asyncio.Queue[tuple[str, int, dict[str, str] | None, Exception | None]] = (
        asyncio.Queue()
    )

    tasks = [
        asyncio.create_task(_worker(queue, client, model, ds[i], i, total, semaphore))
        for i in pending_indices
    ]
    writer_task = asyncio.create_task(_writer(queue, appender, conn, len(tasks)))

    await asyncio.gather(*tasks)
    success, failed = await writer_task
    conn.close()
    print(f"Finished. success={success}, failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct MMLU-Pro SFT dataset using OpenAI-compatible API."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sft.json",
        help="Output file name under speculative/ (default: sft.json)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Concurrent request count (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max sample count for debugging",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for OpenAI-compatible API (fallback: MODEL_NAME in .env)",
    )
    parser.add_argument(
        "--sqlite",
        type=str,
        default="sft_progress.db",
        help="SQLite file name under speculative/ for resume (default: sft_progress.db)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
