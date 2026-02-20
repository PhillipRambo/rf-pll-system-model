from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()

    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"YAML is empty or invalid: {p}")

    return data
