#!/usr/bin/env python3
"""
Build script to package detector-train_deploy_db as a tar.gz module
for electron-app-clean-windows one-click import.

Usage:
    python build_package.py
"""

import hashlib
import json
import os
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VERSION_FILE = PROJECT_ROOT / "VERSION.txt"
OUTPUT_DIR = PROJECT_ROOT / "dist"

MODULE_ID = "detector-train"
MODULE_NAME = "目标检测训练部署平台"
MODULE_DESCRIPTION = "YOLOv7 目标检测模型训练与部署全栈系统，支持分布式训练、ONNX导出及Web可视化监控"

FILES_TO_PACKAGE = [
    "VERSION.txt",
    "requirements.txt",
    "setup.py",
    "MANIFEST.in",
    "detector",
    "detector_gui",
    "cfg",
    "data",
    "examples",
]

EXCLUDE_PATTERNS = {
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".gitignore",
    ".gitattributes",
    "*.egg-info",
    "dist",
    "build",
    "wandb",
    "runs",
    "*.log",
    "logs",
    "yolo_train_tool/logs",
    "yolo_train_tool/runs",
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def should_exclude(path: Path) -> bool:
    """Check if path matches any exclude pattern."""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    if path.name.startswith("."):
        return True
    return False


def collect_files(base_dir: Path) -> list[tuple[Path, str]]:
    """Collect all files to package with their archive paths."""
    files = []
    for item in FILES_TO_PACKAGE:
        item_path = base_dir / item
        if item_path.is_dir():
            for root, dirs, filenames in os.walk(item_path):
                dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
                for filename in filenames:
                    file_path = Path(root) / filename
                    if should_exclude(file_path):
                        continue
                    archive_path = file_path.relative_to(base_dir)
                    files.append((file_path, str(archive_path)))
        elif item_path.is_file():
            if not should_exclude(item_path):
                files.append((item_path, item))
    return files


def build_module_json(version: str, files: list) -> dict:
    """Build module.json metadata."""
    file_entries = []
    for file_path, archive_path in files:
        sha256 = compute_sha256(file_path)
        size = file_path.stat().st_size
        file_entries.append({
            "path": archive_path,
            "sha256": sha256,
            "size": size
        })

    return {
        "id": MODULE_ID,
        "name": MODULE_NAME,
        "version": version,
        "description": MODULE_DESCRIPTION,
        "repository": "",
        "runtime": {
            "type": "stdio",
            "mode": "http",
            "command": [
                "python",
                "-m",
                "detector_gui.cli",
                "--port",
                "{PORT}"
            ],
            "workingDir": ".",
            "env": {},
            "web": {
                "enabled": True,
                "port": "auto",
                "entry": "/"
            }
        },
        "autoRestart": True,
        "hotReload": False,
        "platforms": ["win", "linux", "mac"],
        "files": file_entries
    }


def create_package():
    """Create the tar.gz package."""
    version = VERSION_FILE.read_text().strip()
    print(f"Building package version: {version}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / f"{MODULE_ID}-v{version}.tar.gz"
    if output_file.exists():
        output_file.unlink()

    files = collect_files(PROJECT_ROOT)
    print(f"Collected {len(files)} files to package")

    module_json = build_module_json(version, files)

    with tarfile.open(output_file, "w:gz") as tar:
        module_json_bytes = json.dumps(module_json, ensure_ascii=False, indent=2).encode("utf-8")
        import io
        module_json_info = tarfile.Info(name="module.json", size=len(module_json_bytes))
        tar.addfile(module_json_info, io.BytesIO(module_json_bytes))

        for file_path, archive_path in files:
            tar.add(file_path, arcname=archive_path)
            print(f"  Added: {archive_path}")

    print(f"\nPackage created: {output_file}")
    print(f"Package size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    module_json_file = OUTPUT_DIR / f"{MODULE_ID}-v{version}-module.json"
    with open(module_json_file, "w", encoding="utf-8") as f:
        json.dump(module_json, f, ensure_ascii=False, indent=2)
    print(f"Module metadata: {module_json_file}")

    return output_file


if __name__ == "__main__":
    create_package()
