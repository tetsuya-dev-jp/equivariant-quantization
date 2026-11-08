"""Small output and runtime helpers for calibration commands."""

import json
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def prepare_output_files(
    output_dir: str | Path,
    filenames: list[str],
    overwrite: bool,
) -> Path:
    output_path = Path(output_dir)
    existing = [output_path / name for name in filenames]
    if any(path.exists() for path in existing) and not overwrite:
        raise FileExistsError(
            f"{output_path} already contains calibration output; pass overwrite=True"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink(missing_ok=True)
    return output_path


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; other platforms report bytes.
    if sys.platform.startswith("linux"):
        return int(usage * 1024)
    return int(usage)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def software_metadata() -> dict[str, Any]:
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        return {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {"device": str(device), "device_name": platform.processor()}
