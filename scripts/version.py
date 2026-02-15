#!/usr/bin/env python3
"""Check that version numbers are consistent across its definitions."""

import re
import sys
from pathlib import Path


def normalize_version(version: str) -> str:
    """Normalize version strings to handle different pre-release conventions."""
    version = version.strip().strip('"').strip("'")
    version = re.sub(r"\+.*$", "", version)
    version = re.sub(r"-rc\.", "rc", version)
    version = re.sub(r"\.rc\.", "rc", version)
    version = re.sub(r"-pre\.", "rc", version)
    version = re.sub(r"\.pre\.", "rc", version)
    version = re.sub(r"-rc", "rc", version)
    version = re.sub(r"-pre", "rc", version)
    return version


def extract_version(path: Path, pattern: str) -> str:
    """Extract version from file."""
    content = path.read_text()
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {path}")
    return match.group(1)


def main() -> int:
    """Main entry point."""
    root = Path(__file__).parent.parent

    files = {
        "Cargo.toml": (root / "Cargo.toml", r'^version\s*=\s*["\']([^"\']+)["\']'),
        "pyproject.toml": (root / "pyproject.toml", r'^version\s*=\s*["\']([^"\']+)["\']'),
        "__init__.py": (root / "python" / "evalica" / "__init__.py", r'__version__\s*=\s*["\']([^"\']+)["\']'),
    }

    versions = {}
    for name, (path, pattern) in files.items():
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        versions[name] = normalize_version(extract_version(path, pattern))

    for name, version in versions.items():
        print(f"{name}: {version}")

    if len(set(versions.values())) != 1:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
