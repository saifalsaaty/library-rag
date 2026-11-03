"""Source loaders: OAI-PMH (Dublin Core), MARC21 and CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..schema import Record


def load_source(src: dict, raw_dir: Path, log=print) -> Iterator[Record]:
    kind = src["type"]
    name = src["name"]
    if kind == "oai" and src.get("journals"):
        from .oai import OAIUnavailable, harvest

        failed, closed = [], []
        for code in src["journals"]:
            url = src["url_template"].format(code=code)
            log(f"Harvesting journal '{code}' from {url}")
            try:
                yield from harvest(
                    url, name, raw_dir / "oai", metadata_prefix=src.get("metadata_prefix", "oai_dc"),
                    from_date=src.get("from_date"), until_date=src.get("until_date"),
                    sleep_seconds=float(src.get("sleep_seconds", 1.0)), log=log,
                    page_dir=raw_dir / "oai" / name / code,
                )
            except OAIUnavailable as exc:
                closed.append(code)
                log(f"  !! journal '{code}': {exc} — it cannot be harvested and is excluded")
            except Exception as exc:          # one unreachable journal must not stop the others
                failed.append(code)
                log(f"  !! journal '{code}' failed: {exc} — re-run later with:  --only {name} --journals {code}")
        if closed:
            log(f"Journals without a public OAI service: {closed}")
        if failed:
            log(f"Journals that failed (server errors): {failed}")
        return
    if kind == "oai":
        from .oai import harvest

        sets = src.get("sets") or [None]
        for s in sets:
            log(f"Harvesting {name} set={s or 'ALL'} from {src['base_url']}")
            yield from harvest(
                src["base_url"], name, raw_dir / "oai",
                metadata_prefix=src.get("metadata_prefix", "oai_dc"), set_spec=s,
                from_date=src.get("from_date"), until_date=src.get("until_date"),
                sleep_seconds=float(src.get("sleep_seconds", 1.0)), log=log,
            )
    elif kind == "oai_local":
        from .oai import reparse_raw

        yield from reparse_raw(src["path"], name)
    elif kind == "marc":
        from .marc import read_marc

        yield from read_marc(src["path"], name)
    elif kind == "csv":
        from .csv_loader import read_table

        yield from read_table(src["path"], name, src["columns"], src.get("list_separator", ";"))
    else:
        raise ValueError(f"unknown source type: {kind}")
