"""Parse ground-control-point files: total-station TXT, CSV and Excel.

The field convention follows the user's survey exports: ``ID E N [Z]`` per
line/row, where E/N are projected metric coordinates (the same numbers
AutoCAD shows). Realities this parser absorbs:

* decimal commas ("4795123,45") next to comma/semicolon separators,
* lines without an ID (``E N`` or ``E N Z`` — disambiguated from
  ``ID E N`` by whether the first token plausibly IS an id: a short
  integer, unlike a coordinate),
* Excel sheets with or without a header row (header names are matched in
  English and Croatian; X/Y headers use the geodetic convention Y=easting,
  X=northing, which the in-app auto-match corrects anyway if wrong),
* comment lines (#, //) and blank lines.

Every function returns points as ``[{"id": str, "e": float, "n": float}]``
— Z is parsed but dropped (the affine georeferencing is 2-D).
"""

from __future__ import annotations

import io
import re

from boneseg.logging_setup import get_logger

try:
    import openpyxl

    HAS_OPENPYXL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_OPENPYXL = False

logger = get_logger(__name__)

EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
POINT_FILE_EXTENSIONS = (".txt", ".csv", ".tsv", ".dat", ".asc") + EXCEL_EXTENSIONS

# Header synonyms for Excel/CSV column detection (lowercased, stripped).
_ID_NAMES = {"id", "pt", "point", "name", "label", "broj", "tocka", "točka", "br"}
_E_NAMES = {"e", "east", "easting", "y"}   # geodetic: Y = easting
_N_NAMES = {"n", "north", "northing", "x"}  # geodetic: X = northing
_Z_NAMES = {"z", "h", "elev", "elevation", "height", "visina", "kota"}

_DEC_COMMA = re.compile(r"^-?\d+,\d+$")


def _to_float(tok: str) -> float | None:
    """Parse a numeric token; Croatian decimal commas accepted."""
    tok = tok.strip()
    if _DEC_COMMA.match(tok):
        tok = tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def _looks_like_id(tok: str) -> bool:
    """A plausible point id: short integer (or any non-pure-number string).

    Coordinates are 6-7 digit values or carry decimals; survey point ids
    are small integers ("1", "101") or alphanumeric ("T5", "GCP_2").
    """
    if _to_float(tok) is None:
        return True
    return bool(re.fullmatch(r"-?\d{1,5}", tok.strip()))


def _classify(tokens: list[str]) -> dict | None:
    """Turn one row of string tokens into a point dict, or None."""
    tokens = [t for t in (t.strip() for t in tokens) if t]
    if not tokens:
        return None
    nums = [_to_float(t) for t in tokens]

    # Leading non-numeric token is always the id.
    if nums[0] is None:
        rest = [v for v in nums[1:] if v is not None]
        if len(rest) >= 2:
            return {"id": tokens[0], "e": rest[0], "n": rest[1]}
        return None

    numeric = [v for v in nums if v is not None]
    if len(numeric) < 2 or len(numeric) != len(nums):
        # Mixed junk beyond a leading id is not a point row.
        return None
    if len(numeric) == 2:
        return {"id": "", "e": numeric[0], "n": numeric[1]}
    if len(numeric) == 3:
        # "ID E N" vs "E N Z": decided by whether token 0 looks like an id.
        if _looks_like_id(tokens[0]):
            return {"id": tokens[0], "e": numeric[1], "n": numeric[2]}
        return {"id": "", "e": numeric[0], "n": numeric[1]}
    # 4+ numeric tokens: convention "ID E N Z ..." unless the first value
    # cannot be an id (then "E N Z ..." without one).
    if _looks_like_id(tokens[0]):
        return {"id": tokens[0], "e": numeric[1], "n": numeric[2]}
    return {"id": "", "e": numeric[0], "n": numeric[1]}


def _split_line(line: str) -> list[str]:
    """Split on tabs/semicolons first (safe with decimal commas), then
    whitespace; plain comma only as a last resort and only where commas
    cannot be decimal separators (an even mix is ambiguous — semicolon and
    whitespace exports dominate in practice)."""
    if "\t" in line:
        return line.split("\t")
    if ";" in line:
        return line.split(";")
    parts = line.split()
    if len(parts) > 1:
        return parts
    # Single blob with commas: "1,457123.45,4795321.98" (dot decimals) or
    # "457123,45" alone (decimal comma — NOT a separator).
    if "," in line and not _DEC_COMMA.match(line.strip()):
        return line.split(",")
    return parts


def parse_gcp_text(text: str) -> list[dict]:
    """Parse pasted/text-file point data, one point per line."""
    pts: list[dict] = []
    for raw in text.split("\n"):
        raw = raw.strip().rstrip("\r")
        if not raw or raw.startswith("#") or raw.startswith("//"):
            continue
        p = _classify(_split_line(raw))
        if p is not None:
            pts.append(p)
    # A leading header line ("ID;E;N") classifies as nothing and is skipped
    # naturally; a header like "Point E N" would too (no numerics).
    return pts


def _header_columns(row: list) -> dict[str, int] | None:
    """Detect a header row; returns column indices for e/n (+id) or None."""
    names = [str(c).strip().lower() if c is not None else "" for c in row]
    cols: dict[str, int] = {}
    for i, name in enumerate(names):
        if name in _E_NAMES and "e" not in cols:
            cols["e"] = i
        elif name in _N_NAMES and "n" not in cols:
            cols["n"] = i
        elif name in _ID_NAMES and "id" not in cols:
            cols["id"] = i
        elif name in _Z_NAMES and "z" not in cols:
            cols["z"] = i
    return cols if "e" in cols and "n" in cols else None


def parse_gcp_xlsx(data: bytes) -> list[dict]:
    """Parse the first sheet of an Excel workbook.

    With a recognized header row, columns are taken from it; otherwise each
    row's cells are classified with the same heuristic as text lines.
    """
    if not HAS_OPENPYXL:
        raise RuntimeError(
            "Excel support needs the 'openpyxl' package (pip install openpyxl) "
            "— or save the points as .txt/.csv instead.")
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()

    cols: dict[str, int] | None = None
    start = 0
    for i, row in enumerate(rows[:5]):  # header, if any, is near the top
        c = _header_columns(row)
        if c is not None:
            cols, start = c, i + 1
            break

    pts: list[dict] = []
    for row in rows[start:]:
        if row is None:
            continue
        if cols is not None:
            def cell(key: str):
                j = cols.get(key)
                return row[j] if j is not None and j < len(row) else None
            e = _to_float(str(cell("e"))) if cell("e") is not None else None
            n = _to_float(str(cell("n"))) if cell("n") is not None else None
            if e is None or n is None:
                continue
            rid = cell("id")
            pts.append({"id": "" if rid is None else str(rid).strip(),
                        "e": e, "n": n})
        else:
            tokens = [str(c) for c in row if c is not None and str(c).strip()]
            p = _classify(tokens)
            if p is not None:
                pts.append(p)
    return pts


def parse_gcp_file(filename: str, data: bytes) -> list[dict]:
    """Parse an uploaded point file by extension (Excel vs text/CSV)."""
    name = (filename or "").lower()
    if name.endswith(EXCEL_EXTENSIONS):
        pts = parse_gcp_xlsx(data)
    elif name.endswith(".xls"):
        raise RuntimeError(
            "Legacy .xls is not supported — save the sheet as .xlsx or .csv.")
    else:
        text = None
        for enc in ("utf-8-sig", "cp1250", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise RuntimeError("Could not decode the file as text.")
        pts = parse_gcp_text(text)
    logger.info("Parsed %d GCP point(s) from %s", len(pts), filename)
    return pts


def looks_like_lonlat(points: list[dict]) -> bool:
    """True when coordinates look like geographic degrees, not metres —
    georeferencing needs a projected CRS, so the UI warns about these."""
    if not points:
        return False
    return all(abs(p["e"]) <= 360 and abs(p["n"]) <= 90 for p in points)
