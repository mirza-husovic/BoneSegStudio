# SESSION HANDOFF — 2026-07-08 (Fable 5 → Opus 4.8)

Za Opusa: ovo je predaja posla na **BoneSeg Studiju** (`C:\Users\mirza\BoneSegStudio`,
pokreće se s `python app.py`). Korisnik je Mirza — arheolog, ne programer;
komuniciraj na hrvatskom (engleski tehnički termini u zagradama su OK).
**Najvažnije pravilo: STABILNOST.** Aplikacija je za korisnika "gotov
proizvod"; svaka izmjena mora biti aditivna, verificirana i commitana u git.
Ako nešto pođe po zlu: `git log --oneline` + `git checkout <tag/commit> -- .`
— tag `baseline-pre-catalog` je stanje prije današnjih dodataka.

## Što je napravljeno danas (sve verificirano i commitano)

1. **Drag & drop batch** — dovuci folder/slike na batch panel →
   `/api/batch_upload` stagea u `uploads/batch/<timestamp>` i popuni
   "Input folder". (commit u prošloj sesiji od danas)
2. **Batch red → editor** — klik na ime datoteke u batch tablici otvori
   fotku s VEĆ izračunatim rezultatom (bez re-inferencea):
   `/api/batch_open` čita exportanu `<stem>_mask.(tif|png)` i seeda result
   kroz `pipeline.postprocess(prob=mask.astype(float32))`. Redovi u
   `process_folder` sada nose `path` + `out_dir`. Output dir se
   auto-postavi na podfolder itema (re-export zamjenjuje artefakte).
3. **Faza 1: kataloška tabla (PDF)** — `boneseg/data/plate.py`;
   checkbox "Plate (PDF)" u Export izborima (single + batch) + polje
   "Site name" (localStorage). A4, fotka + maska + crveni centerline
   vektori, **egzaktno standardno mjerilo** (1:10/1:20/... — panel se
   dimenzionira da papir×mjerilo = metri), crno-bijela mjerka, N strelica
   izračunata iz affine transformacije, CRS/datum/model/nota.
   Bez georeference: tabla bez mjerke ("scale unknown").
4. **Faza 2: GCP georeferenciranje u appu** — `boneseg/data/georef_fit.py`
   (least-squares affine iz 3+ točaka, rezidiuali u metrima) +
   `POST /api/georef` + GCP alat u toolbaru (tipka **G**): plutajući panel,
   klik na fotku = narančasti križić, upiši/zalijepi E/N (paste txt
   total-station formata `ID E N [Z]`, decimalni zarezi OK, klikovi uzimaju
   točke redom iz liste), EPSG polje (default 3765 = HTRS96/TM),
   "swap E/N" checkbox. Primjena: vektori (`polylines_out`/`rings_out`)
   se preračunaju NA MJESTU (bez re-inferencea), piše se **sidecar**
   `<fotka>.jpg.gcps.json` (auto-reapply na svako otvaranje, i single i
   batch_open!) i **world file** (`.jgw`/`.pgw`/`.tfw` + `.prj`) pa se
   ORIGINALNA fotka otvara georeferencirana u QGIS-u. `clear:true` briše.
   Sidecar autoload živi u `read_image` (readers.py, lazy import) pa vrijedi
   za SVE putove: single open, batch i batch-reopen.
5. **Kombinirani `catalog.pdf`** — batch s uključenom "Plate (PDF)" opcijom
   uz per-image table piše i jedan `catalog.pdf` (grob po stranici) u root
   batch outputa. `plate.py` = `render_plate_figure()` + `save_plate_pdf()`;
   `process_folder` akumulira kroz `PdfPages` (zatvara se u `finally`,
   prazan se briše).

Testovi: `python tests/smoke_test.py` (E2E, mora proći!),
`python tests/test_georef_fit.py`, `python tests/test_plate.py`.

## Kako verificirati izmjene na ovom stroju

- Preview server: launch config `boneseg-studio` (port 7861) u
  `~/.claude/launch.json`. ⚠️ `preview_screenshot` na ovom stroju VISI
  (timeout) iako je app zdrav — koristi `preview_eval` / `preview_inspect` /
  `preview_snapshot` / `preview_console_logs`.
- Frontend funkcije (`gcpAddPoint`, `stageBatchDrop`, `applySummary`…) su
  top-level u `app.js` → dostupne direktno iz `preview_eval`.
- PowerShell tool: commit poruke s dvostrukim navodnicima se raspadnu —
  piši commit poruke BEZ `"` znakova (koristi here-string `@'...'@`).
- Nakon izmjene `app.js`: cache-busting `?v=<mtime>` postoji, ali reci
  korisniku Ctrl+F5 ako vidi staro ponašanje.

## Poznata ograničenja / sitnice (namjerno ostavljeno)

- **Affine ≠ kosa fotka**: georeferenciranje je egzaktno za orto/nadir
  snimke; kose fotke → veliki rezidiuali (UI upozorava >15 cm). Prava
  podrška = homografija (vektori se transformiraju točno točku-po-točku,
  ali maska GeoTIFF bi trebala warp; rasterio GeoRef je affine-only).
- Parser točaka: linija `E N Z` BEZ ID-a (3 broja) se krivo čita kao
  `ID=E, E=N, N=Z` — konvencija je "prvi token = ID". Korisnik vidi
  vrijednosti u tablici pa može ispraviti.
- Nakon `batch_open` threshold slider je inertan (maska je binarni "prob")
  dok se ne pokrene novi inference — namjerno.
- Re-export nakon batch_open ide u podfolder itema; **master** ostaje u
  batch ROOTU — replace mastera traži ručno outdir=root + checkbox.
- Pola-piksela: rasterio `transform.xy` koristi offset="center" za
  centerline export, GCP fit i rings koriste kontinuirane koordinate
  direktno → interna nekonzistencija ≤0.5 px (sub-milimetar). Ignorirano.
- `README.md` NIJE ažuriran za današnje featuree.

## Sljedeći koraci (redom po vrijednosti, NIŠTA nije obećano korisniku kao gotovo)

1. **Metapodaci po grobu na tabli**: dubina, orijentacija, opis, popis
   nalaza — mala forma u UI (per-image, spremati u sidecar JSON uz fotku
   kao gcps). Katalozi grobova to standardno imaju.
2. **Redoslijed/odabir grobova u catalog.pdf** (sad = abecedno po datoteci).
3. **Drawing-style tabla**: varijanta s bijelom podlogom + crne
   centerline + sivi outline (postojeći "Clean" pogled kao podloga) —
   klasičan izgled crteža u publikacijama.
4. Homografija za kose fotke (vidi gore).
5. README update.

## Kontekst korisnika (za ton i prioritete)

- Cilj: novi podaci → retrain → zamjena modela (`models/registry.py`,
  checkpoint u `Desktop\modeli\`). App se tada NE dira, samo registry.
- Ovaj alat je i vitrina za karijeru (digital archaeology / ML) — kvaliteta
  outputa (katalozi, GIS interop) ima direktnu vrijednost.
- Koordinate korisnika: total-station txt, često "AutoCAD" = projicirane
  metričke koordinate (HTRS96/TM EPSG:3765 ili stari GK). Isti brojevi rade
  u GIS-u čim se zada EPSG; bez EPSG-a = lokalna mreža (CAD radi, GIS bez
  CRS taga). Y/X zamjena je klasična zamka → swap checkbox + rezidiuali.
