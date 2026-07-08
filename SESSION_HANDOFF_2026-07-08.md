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

6. **GCP fajl s točkama (txt/csv/xlsx)** — `boneseg/data/gcp_parse.py` +
   `POST /api/gcp_points`; drop-zona i "browse…" u GCP panelu; drop bilo
   gdje na prozor dok je GCP alat aktivan također radi (routing po
   ekstenziji). Paste ("Load points") ide kroz ISTI backend parser (stari
   JS parser izbačen). Parser: decimalni zarezi, `;`/tab/space/`,`,
   `ID E N [Z]` i `E N [Z]` bez ID-a (prvi token je ID samo ako je kratki
   integer — stara "E N Z bez ID-a" zamka time RIJEŠENA), Excel s
   headerom (geodetski: Y=easting, X=northing) ili bez, cp1250 fallback.
   openpyxl u requirements (opcionalan; .xls odbijen s porukom).
7. **Auto-match GCP-ova — redoslijed klikanja NEBITAN** —
   `auto_assign_gcps()` u `georef_fit.py`: seed = najrašireniji trokut
   klikova × sve uređene trojke točaka (vektorizirano, cap 40 točaka),
   greedy dodjela + refit; orijentacija det<0 (foto→svijet je uvijek
   "zrcalan" jer row raste dolje, N gore); det>0 pobjednik = E/N
   zamijenjeni u fajlu → **auto-swap + poruka u UI**. Višak neklikanih
   točaka u fajlu je OK; simetričan raspored → "ambiguous" upozorenje.
   UI: checkbox "auto-match (order-free)" default ON; tablica ima ID
   stupac i nakon Applyja pokazuje stvarno sparivanje. Ručni unos E/N
   po kliku (bez fajla) = staro ponašanje. EPSG polje dobilo datalist
   (3765, GK 31275/6, FR 2154, UTM…) + `lonlat_warning` za stupnjeve.
8. **GeoTIFF fotke u exportu** — `save_photo_geotiff()` u writers.py
   (JPEG-in-TIFF q92, fallback deflate, tiled 256, overviews 2-16×);
   checkbox "GeoTIFF photo" (isti #exportchoices služi single i batch);
   piše `<stem>_photo.tif`. Bez georeference: preskočen + `note` u
   odgovoru koji UI pokaže kao toast/status.

9. **Parser fix (user report)**: total-station red s KODOM na kraju
   (`1,576362.2191,5016382.8409,229.3286,GR30`) se odbijao — sad se
   koordinate čitaju iz VODEĆEG niza brojeva, sve nakon prvog
   ne-brojčanog tokena (kod/opis) se ignorira. Regresijski test s točnim
   sadržajem korisnikovog fajla.
10. **Fotka u DXF-u (CAD underlay)**: `_embed_dxf_photo()` u writers.py —
    per-grave DXF sadrži IMAGE entitet na layeru PHOTO s EGZAKTNIM
    per-pixel u/v vektorima iz affine (insert=T*(0,h), u=(a,d),
    v=(-b,-e)); fotka se KOPIRA uz DXF (relativna referenca → folder
    prenosiv) pa i obični AutoCAD/LT i BricsCAD otvore fotku ispod
    vektora bez Map 3D-a. Radi i u pixel modu ((0,-h), unit u/v).
    IMAGE se piše PRVI (ostaje ispod), greška nikad ne ruši vektorski
    export. master.dxf NEMA image (verificirano). NB: broj entiteta u
    smoke [7/7] varira 24↔25 zbog poznate nedeterminističke
    skeletonizacije, nije bug.

11. **HOMOGRAFIJA za kose fotke (user: "razvučeno u QGIS/CAD-u")** —
    affine je kosu fotku shearao u paralelogram, a shearani u/v vektori
    su i razlog zašto obični AutoCAD NIJE prikazivao DXF underlay (CAD
    ne renderira iskošene rastere). Rješenje: `fit_homography_gcps`
    (normalizirani DLT — svjetske koordinate ~5e6 m TRAŽE Hartley
    normalizaciju!), `apply_homography`, `rectify_params` (north-up grid
    na GSD centra slike, cap 4× piksela izvora). **Policy u
    `georef_from_gcps`**: 3 točke → affine; 4+ → affine ako mu je RMS ≤
    5 cm (nadir), inače homografija u `GeoRef.homography` (affine ostaje
    kao aproksimacija za potrošače koji ne znaju perspektivu — plate
    mjerka). Vektori idu kroz homografiju EGZAKTNO
    (vectorize.py); `save_photo_geotiff`/`save_mask_raster` warpaju
    (rektificiraju) na north-up grid + internal validity mask za rubove;
    **DXF underlay se UVIJEK rektificira kad je georeferenciran** (osno
    poravnati u/v → svaki CAD prikazuje); world file se u perspective
    modu PRESKAČE (nosi samo affine = razvučeni izgled) + stari
    .jgw/.prj se BRIŠU; `batch_open` unwarpa rektificiranu batch masku
    natrag u pixel space. UI status javlja mod (affine/perspective) i
    upućuje na GeoTIFF/DXF export; 4 točke = bez redundancije (rezidiuali
    0) → upozorenje da se doda 5.

12. **Auto-match tie-break za SIMETRIČNE rasporede (user: "sve isto,
    razvučeno")** — stvarni slučaj: 4 GCP-a čine pravokutnik ~0.98×0.44 m.
    Affine preslikava pravokutnik na SVAKU cikliranu verziju samog sebe
    EGZAKTNO → sve 4 rotacije sparivanja fitaju unutar šuma klika, čisti
    min-RMS je deterministički birao KRIVU → fotka zgnječena za aspect²
    (~4.8×) = "razvučeni" izgled u QGIS/CAD-u pri RMS 0.9 cm i affine
    modu (rezidiuali NE detektiraju krivo sparivanje simetričnog
    rasporeda!). Fix u `auto_assign_gcps`: statistički izjednačeni
    kandidati (ista orijentacija, RMS ≤ max(2×best, best+2cm)) rangiraju
    se po (1) najmanjoj distorziji (omjer singularnih vrijednosti — kriva
    rotacija MORA gnječiti), (2) RMS u 5 mm koracima, (3) redoslijedu
    klikanja (identitet za savršenu simetriju). Regresija s TOČNIM GR30
    koordinatama (klik po redu / izmiješano / s 5. točkom); 5 kliknutih
    točaka razbija simetriju i gasi "ambiguous". UI poruka objašnjava
    tie-break + savjetuje 5. točku.

13. **CAD fotka — pravi uzrok nađen i riješen**: XREF paleta pokazala
    IMAGEDEF "Unreferenced" → ezdxf defaulta ključ ACAD_IMAGE_DICT-a na
    FILENAME (apsolutna putanja s `\` `:` `.`) što NIJE valjano
    simboličko ime → AutoCAD odbaci definiciju, raster se nikad ne učita
    (samo okvir). Fix: eksplicitni `name="<STEM>_PHOTO"` u
    `add_image_def`. **Pouka: kod ezdxf slika UVIJEK zadati `name`.**
14. **Plate PDF = drawing stil po defaultu** (crne centerline na
    bijelom, bez fotke; `style="photo"` čuva stari izgled) — user
    zahtjev; mjerilo/mjerka/N/title block nepromijenjeni; vrijedi i za
    batch catalog.pdf.

Testovi: `python tests/smoke_test.py` (E2E, mora proći!),
`python tests/test_georef_fit.py` (auto-assign + swap + homografija +
rectify grid + GR30 simetrija), `python tests/test_gcp_parse.py`,
`python tests/test_dxf_image.py` (rektificirani underlay),
`python tests/test_plate.py`.

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

- ~~Affine ≠ kosa fotka~~ RIJEŠENO (homografija, vidi #11). Ostatak:
  homografija pretpostavlja RAVNU plohu groba — točke na različitim
  dubinama (Z raspon velik) i dalje daju rezidiuale; plate mjerka u
  perspective modu koristi affine aproksimaciju (mjerilo kose fotke
  ionako nije uniformno na papiru).
- Auto-match je O(m³) po broju točaka u fajlu — cap na 40 točaka
  (ValueError s porukom); za grob-po-grob fajlove (<20 točaka) trenutno.
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
