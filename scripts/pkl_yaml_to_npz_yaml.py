#!/usr/bin/env python3
"""Convert a pkl motion-library yaml into an npz motion-library yaml.

Supported source schema:

root_path: /path/to/pkl_root/
motions:
  - file: some/sub/path.pkl
    weight: 1.0
    description: xxx

Output keeps all non-file fields and only rewrites:
1) root_path/root_dir/root (unless --keep_root_path is set)
2) motions[*].file suffix: .pkl/.pickle -> .npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pkl motion yaml to npz motion yaml.")
    parser.add_argument("--input_yaml", type=str, required=True, help="Source yaml path.")
    parser.add_argument(
        "--output_yaml",
        type=str,
        default=None,
        help="Output yaml path. Default: <input_stem>_npz.yaml in the same directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Override output root path written to yaml. "
            "If unset and --keep_root_path is false, auto-derive by appending '_npz' to source root."
        ),
    )
    parser.add_argument(
        "--keep_root_path",
        action="store_true",
        help="Do not modify root_path/root_dir/root in yaml.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write result back to input yaml (overrides --output_yaml).",
    )
    parser.add_argument(
        "--allow_non_pkl",
        action="store_true",
        help="If set, motions with non-.pkl/.pickle suffix are kept unchanged instead of raising error.",
    )
    return parser.parse_args()


def derive_output_yaml_path(input_yaml: Path) -> Path:
    return input_yaml.with_name(f"{input_yaml.stem}_npz.yaml")


def with_suffix_preserve(path_text: str, suffix: str) -> str:
    path = Path(path_text)
    return str(path.with_suffix(suffix))


def derive_output_root(input_root: str) -> str:
    stripped = input_root.rstrip("/")
    if stripped.endswith("_npz"):
        return input_root
    derived = f"{stripped}_npz"
    if input_root.endswith("/"):
        derived = f"{derived}/"
    return derived


def pick_root_key(payload: dict[str, Any]) -> str | None:
    for key in ("root_path", "root_dir", "root"):
        if key in payload:
            return key
    return None


def main() -> None:
    args = parse_args()

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required. Please install with: pip install pyyaml") from error

    input_yaml = Path(args.input_yaml).expanduser().resolve()
    if not input_yaml.is_file():
        raise FileNotFoundError(f"input yaml not found: {input_yaml}")

    if args.inplace:
        output_yaml = input_yaml
    elif args.output_yaml is not None:
        output_yaml = Path(args.output_yaml).expanduser().resolve()
    else:
        output_yaml = derive_output_yaml_path(input_yaml)

    with input_yaml.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"yaml root must be a mapping, got {type(payload)!r}")
    motions = payload.get("motions", None)
    if not isinstance(motions, list):
        raise ValueError("yaml must contain a 'motions' list")

    converted = 0
    unchanged = 0
    for index, item in enumerate(motions):
        if isinstance(item, str):
            source_file = item
            suffix = Path(source_file).suffix.lower()
            if suffix in {".pkl", ".pickle"}:
                motions[index] = with_suffix_preserve(source_file, ".npz")
                converted += 1
            elif args.allow_non_pkl:
                unchanged += 1
            else:
                raise ValueError(
                    f"motions[{index}] has non-pkl suffix: {source_file}. "
                    "Use --allow_non_pkl to keep unchanged."
                )
            continue

        if not isinstance(item, dict):
            raise TypeError(f"motions[{index}] must be a mapping or string, got {type(item)!r}")
        source_file = item.get("file", None)
        if source_file is None:
            raise KeyError(f"motions[{index}] missing required key: file")
        source_file_text = str(source_file)
        suffix = Path(source_file_text).suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            item["file"] = with_suffix_preserve(source_file_text, ".npz")
            converted += 1
        elif args.allow_non_pkl:
            unchanged += 1
        else:
            raise ValueError(
                f"motions[{index}].file has non-pkl suffix: {source_file_text}. "
                "Use --allow_non_pkl to keep unchanged."
            )

    root_key = pick_root_key(payload)
    if not args.keep_root_path:
        if args.output_root is not None:
            if root_key is None:
                root_key = "root_path"
            payload[root_key] = args.output_root
        elif root_key is not None:
            payload[root_key] = derive_output_root(str(payload[root_key]))

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)

    print(f"[INFO] Input yaml   : {input_yaml}")
    print(f"[INFO] Output yaml  : {output_yaml}")
    if root_key is not None:
        print(f"[INFO] Root key     : {root_key} -> {payload[root_key]}")
    else:
        print("[INFO] Root key     : not found (kept absent)")
    print(f"[INFO] Motions total : {len(motions)}")
    print(f"[INFO] Converted     : {converted}")
    print(f"[INFO] Unchanged     : {unchanged}")


if __name__ == "__main__":
    main()
