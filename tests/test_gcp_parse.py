"""Unit test: GCP point-file parsing (txt/CSV/Excel) + auto-matching.

Run:  python tests/test_gcp_parse.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boneseg.data.gcp_parse import (HAS_OPENPYXL, looks_like_lonlat,
                                    parse_gcp_file, parse_gcp_text)

# --- classic total-station lines: ID E N Z, whitespace ---
pts = parse_gcp_text("""# stanica 1
1  457123.45  4795321.98  152.30
2  457125.10  4795324.55  152.28
T5 457127.99  4795320.01  152.41
""")
assert [p["id"] for p in pts] == ["1", "2", "T5"], pts
assert pts[0]["e"] == 457123.45 and pts[0]["n"] == 4795321.98
print("ID E N Z whitespace OK")

# --- decimal commas + semicolons (Croatian CSV export) ---
pts = parse_gcp_text("101;457123,45;4795321,98\n102;457125,10;4795324,55")
assert pts[0]["e"] == 457123.45 and pts[1]["n"] == 4795324.55
print("decimal commas + semicolons OK")

# --- E N only (no id) ---
pts = parse_gcp_text("457123.45 4795321.98\n457125.10 4795324.55")
assert pts[0]["id"] == "" and pts[0]["e"] == 457123.45
print("E N without id OK")

# --- E N Z without id: first token is NOT a plausible id (6+ digits) ---
pts = parse_gcp_text("457123.45 4795321.98 152.30")
assert pts[0]["e"] == 457123.45 and pts[0]["n"] == 4795321.98, pts
print("E N Z (no id) heuristic OK")

# --- comma-separated with dot decimals ---
pts = parse_gcp_text("1,457123.45,4795321.98")
assert pts[0]["id"] == "1" and pts[0]["n"] == 4795321.98
print("comma-separated OK")

# --- header line skipped naturally ---
pts = parse_gcp_text("ID;E;N\n1;457123.45;4795321.98")
assert len(pts) == 1 and pts[0]["id"] == "1"
print("header skip OK")

# --- lat/lon detector ---
assert looks_like_lonlat([{"e": 16.44, "n": 43.51}])
assert not looks_like_lonlat([{"e": 457123.0, "n": 4795321.0}])
print("lonlat detector OK")

# --- text file bytes (cp1250 with BOM-less content) ---
data = "1\t457123,45\t4795321,98\r\n2\t457125,10\t4795324,55\r\n".encode("cp1250")
pts = parse_gcp_file("tocke.txt", data)
assert len(pts) == 2 and pts[1]["e"] == 457125.10
print("txt bytes decode OK")

# --- Excel: with header, geodetic Y/X convention, and without header ---
if HAS_OPENPYXL:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Point", "Y", "X", "H"])          # geodetic: Y=E, X=N
    ws.append([1, 457123.45, 4795321.98, 152.3])
    ws.append([2, 457125.10, 4795324.55, 152.28])
    buf = io.BytesIO()
    wb.save(buf)
    pts = parse_gcp_file("grob12.xlsx", buf.getvalue())
    assert len(pts) == 2 and pts[0]["id"] == "1"
    assert pts[0]["e"] == 457123.45 and pts[0]["n"] == 4795321.98
    print("xlsx with Y/X header OK")

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.append([1, 457123.45, 4795321.98])       # no header at all
    ws2.append([2, 457125.10, 4795324.55])
    ws2.append([None, None, None])               # trailing empty row
    buf2 = io.BytesIO()
    wb2.save(buf2)
    pts = parse_gcp_file("grob12.xlsx", buf2.getvalue())
    assert len(pts) == 2 and pts[1]["n"] == 4795324.55
    print("xlsx headerless OK")
else:
    print("openpyxl missing — xlsx tests skipped")

print("GCP PARSE TEST PASSED")
