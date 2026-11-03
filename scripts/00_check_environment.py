"""Step 0 — record the hardware/software environment (Reproducibility section of the paper).

    python scripts/00_check_environment.py
"""
from __future__ import annotations

import importlib
import platform
import sys

from _common import init, parser, results_path

from librag.config import write_json

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "torch", "sentence_transformers", "transformers",
            "requests", "yaml", "openpyxl", "matplotlib", "pymarc", "nltk", "rapidfuzz"]


def cpu_name() -> str:
    """Marketing name of the CPU (e.g. 'Intel(R) Core(TM) i7-...'), needed for the paper."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def os_name() -> str:
    if sys.platform == "win32":
        build = int(platform.version().split(".")[-1]) if platform.version().split(".")[-1].isdigit() else 0
        return f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    return platform.platform()


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": cpu_name(),
           "os": os_name()}
    versions = {}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "installed")
        except Exception:
            versions[name] = None
    env["packages"] = versions

    try:
        import torch  # type: ignore

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        env["cuda_available"] = False

    try:
        import psutil  # type: ignore

        env["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass

    from librag.generation.llm import OllamaClient

    g = cfg["generation"]
    try:
        installed = OllamaClient(g["ollama_url"], "x").available_models()
        env["ollama_models_installed"] = installed
        env["ollama_models_missing"] = [m for m in g["models"] if m not in installed and f"{m}:latest" not in installed]
        env["ollama_model_details"] = [OllamaClient(g["ollama_url"], m).model_info()
                                       for m in g["models"] if m in installed]
    except Exception as exc:
        env["ollama_error"] = f"Ollama not reachable at {g['ollama_url']}: {exc}"

    write_json(env, results_path(cfg, "environment.json"))
    for k, v in env.items():
        log.info(f"{k}: {v}")
    if env.get("ollama_models_missing"):
        log.info("Pull the missing models with:  " + "  &  ".join(f"ollama pull {m}" for m in env["ollama_models_missing"]))
    if not versions.get("sentence_transformers"):
        log.info("sentence-transformers is missing:  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
