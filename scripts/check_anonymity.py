#!/usr/bin/env python3
"""Fail when a release tree contains identifying or nonportable material."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {"", ".cfg", ".csv", ".ini", ".json", ".jsonl", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
BINARY_SUFFIXES = {".bin", ".ckpt", ".dll", ".dylib", ".exe", ".gif", ".gz", ".jpeg", ".jpg", ".npy", ".npz", ".onnx", ".parquet", ".pdf", ".pickle", ".pkl", ".png", ".pt", ".pth", ".pyc", ".so", ".tar", ".zip"}
SKIP_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "artifacts", "cache", "checkpoints", "outputs"}


def joined(*parts: str) -> str:
    return "".join(parts)


DENIED_TERMS = [
    joined("si", "va"), joined("ra", "je", "sh"), joined("ka", "sa"),
    joined("sa", "nth", "osh"), joined("ka", "sa", "si", "va"), joined("ku", "ma", "r"),
    joined("sa", "nt", "ka", "sa"), joined("ka", "sa", "kh"), joined("am", "az", "on"),
    joined("e", "fs"), joined("ec2", "_rps"), joined("ec2", "-user"),
    joined("a", "2z"),
]

PATTERNS = {
    "private cloud identifier": re.compile(joined("arn", ":", "aws") + r"|\b\d{12}\b", re.IGNORECASE),
    "object-store URI": re.compile(joined("s", "3", "://"), re.IGNORECASE),
    "mailbox address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "secret assignment": re.compile(r"\b(?:password|api_key|secret_key|access_key)\b\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    "private host": re.compile(r"\bip-\d+-\d+-\d+-\d+(?:\.[A-Z0-9.-]+)?\b|\.(?:corp|internal)(?:\.|\b)", re.IGNORECASE),
    "internal domain": re.compile(joined("ama", "zon", r"\.(?:com|dev|work)") + "|" + joined("a", "2z", r"\.com"), re.IGNORECASE),
    "absolute path": re.compile(r"(?<![A-Za-z0-9_:./])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+"),
    "internal work label": re.compile(
        joined(r"\bIS", r"SUE-\d+\b") + "|" + joined(r"\bHand", r"off\b") + "|" + joined(r"\bMod", r"ule\s+\d+\b"),
        re.IGNORECASE,
    ),
    "internal date label": re.compile(r"\b20\d{2}[-_/]\d{2}[-_/]\d{2}\b"),
    "forbidden dash": re.compile("[\u2013\u2014]"),
}


def scan(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if any(part in SKIP_DIRECTORIES for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        lowered_name = relative.lower()
        for term in DENIED_TERMS:
            if term.lower() in lowered_name:
                failures.append(f"name denylist: {relative}")
        if path.is_dir():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            failures.append(f"binary extension: {relative}")
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            failures.append(f"binary content: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            failures.append(f"unapproved file type: {relative}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 content: {relative}")
            continue
        lowered = text.lower()
        for term in DENIED_TERMS:
            if term.lower() in lowered:
                failures.append(f"content denylist: {relative}: {term}")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                if label == "absolute path" and value == "/usr/bin/env":
                    continue
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {relative}:{line}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = scan(root)
    if failures:
        print("anonymity scan failed")
        for failure in failures:
            print(f"  {failure}")
        return 1
    files = sum(
        path.is_file()
        and not any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts)
        for path in root.rglob("*")
    )
    print(f"anonymity scan passed: {files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
