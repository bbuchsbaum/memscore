from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": MAIN_NS, "r": REL_NS}


def _column_index(cell_ref: str) -> int:
    letters = []
    for char in cell_ref:
        if char.isalpha():
            letters.append(char.upper())
        else:
            break
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return max(value - 1, 0)


def _shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", NS):
        values.append("".join(node.text or "" for node in item.findall(".//a:t", NS)))
    return values


def _sheet_rows(workbook_path: Path) -> Iterable[list[str]]:
    with ZipFile(workbook_path) as archive:
        shared = _shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find("a:sheets", NS)
        if sheets is None or not list(sheets):
            raise ValueError(f"No worksheets found in {workbook_path}")

        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = list(sheets)[0]
        rel_id = first_sheet.attrib[f"{{{REL_NS}}}id"]
        sheet_xml = ET.fromstring(archive.read("xl/" + rel_map[rel_id]))

        for row in sheet_xml.findall(".//a:sheetData/a:row", NS):
            current: list[str] = []
            for cell in row.findall("a:c", NS):
                column = _column_index(cell.attrib.get("r", "A1"))
                while len(current) < column:
                    current.append("")
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//a:t", NS))
                else:
                    node = cell.find("a:v", NS)
                    raw = "" if node is None or node.text is None else node.text
                    if cell_type == "s" and raw:
                        value = shared[int(raw)]
                    else:
                        value = raw
                current.append(value)
            yield current


def build_manifest(
    workbook_path: Path,
    output_path: Path,
    *,
    score_column: str = "CR",
) -> tuple[Path, int]:
    workbook_path = workbook_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _sheet_rows(workbook_path)
    header = next(rows, None)
    if header is None:
        raise ValueError(f"No rows found in {workbook_path}")

    indices = {name: idx for idx, name in enumerate(header)}
    required = ["ID", "Title", "Artist", "Style", "Image ID", "URL", score_column]
    missing_columns = [name for name in required if name not in indices]
    if missing_columns:
        raise KeyError(f"Missing expected artwork columns: {missing_columns}")

    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "url",
                "score",
                "id",
                "artwork_id",
                "image_id",
                "title",
                "artist",
                "style",
                "public_domain",
                "source",
            ],
        )
        writer.writeheader()

        for row in rows:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))

            artwork_id = row[indices["ID"]].strip()
            image_id = row[indices["Image ID"]].strip()
            url = row[indices["URL"]].strip()
            score = row[indices[score_column]].strip()
            if not artwork_id or not image_id or not url or not score:
                continue

            writer.writerow(
                {
                    "path": f"images/{artwork_id}_{image_id}.jpg",
                    "url": url,
                    "score": score,
                    "id": artwork_id,
                    "artwork_id": artwork_id,
                    "image_id": image_id,
                    "title": row[indices["Title"]].strip(),
                    "artist": row[indices["Artist"]].strip(),
                    "style": row[indices["Style"]].strip(),
                    "public_domain": row[indices.get("Public Domain", -1)].strip() if "Public Domain" in indices else "",
                    "source": "AIC-artwork-memorability",
                }
            )
            count += 1

    return output_path, count


def download_images(
    manifest_path: Path,
    output_root: Path,
    *,
    delay_seconds: float = 1.0,
    limit: int | None = None,
    timeout: float = 60.0,
) -> tuple[int, int]:
    manifest_path = manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    downloaded = 0
    failed = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rel_path = row["path"].strip()
            url = row["url"].strip()
            destination = output_root / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                downloaded += 1
                continue
            try:
                with opener.open(url, timeout=timeout) as response, destination.open("wb") as out_handle:
                    out_handle.write(response.read())
                downloaded += 1
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                failed += 1
                print(f"download failed: {url} -> {destination} ({exc})")
            time.sleep(delay_seconds)
    return downloaded, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest CSV for the artwork memorability dataset.")
    parser.add_argument(
        "--workbook-path",
        default="data/artwork/Master List - Online Exp Paintings.xlsx",
        help="Path to the official artwork memorability workbook.",
    )
    parser.add_argument(
        "--output-path",
        default="data/artwork/artwork_memorability_manifest.csv",
        help="Where to write the manifest CSV.",
    )
    parser.add_argument(
        "--score-column",
        choices=["CR", "RESMEM", "HR"],
        default="CR",
        help="Workbook score column to export as the memorability target.",
    )
    parser.add_argument(
        "--download-root",
        default=None,
        help="Optional root directory where images should be downloaded sequentially.",
    )
    parser.add_argument(
        "--download-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between artwork image requests. The Art Institute API asks scrapers to throttle aggressively.",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Optional cap on the number of artwork images to download.",
    )
    args = parser.parse_args()

    output_path, count = build_manifest(
        Path(args.workbook_path),
        Path(args.output_path),
        score_column=args.score_column,
    )
    print(f"Wrote manifest to {output_path}")
    print(f"Rows written: {count}")

    if args.download_root:
        downloaded, failed = download_images(
            output_path,
            Path(args.download_root),
            delay_seconds=args.download_delay_seconds,
            limit=args.download_limit,
        )
        print(f"Downloaded: {downloaded}")
        print(f"Failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
