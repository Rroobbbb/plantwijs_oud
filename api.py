r"""
PlantWijs API — v3.9.2 (Gt 1..19 → Ia..VIIId)
=============================================
Wijzigingen t.o.v. v3.9.1:
- PDOK Gt `value_list` (1..19) wordt nu omgezet naar officiële Gt-code (Ia..VIIId)
- UI toont altijd de nette Gt-code (niet langer 1..19)

Starten (Windows)
  cd C:/PlantWijs
  venv/Scripts/uvicorn api:app --reload --port 9000
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pyproj import Transformer

# ───────────────────── FastAPI app
app = FastAPI(title="PlantWijs API", version="3.9.2")

# CORS (zodat je lokaal kunt testen of met andere frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # evt. beperken tot specifieke hosts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────── Headers / UA
HEADERS = {"User-Agent": "PlantWijs/3.9.2"}
FMT_JSON = "application/json;subtype=geojson"

# WFS FGR
PDOK_FGR_WFS = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0"
    "?service=WFS&version=2.0.0"
)
FGR_WMS = "https://service.pdok.nl/ez/fysischgeografischeregios/wms/v1_0"

# WMS Bodemkaart (BRO)
BODEM_WMS = "https://service.pdok.nl/bzk/bro-bodemkaart/wms/v1_0"

# WMS Grondwaterspiegeldiepte (BRO)
GWD_WMS = "https://service.pdok.nl/bzk/bro-grondwaterspiegeldiepte/wms/v2_0"

# WMS AHN (hoogte)
AHN_WMS = "https://service.pdok.nl/rws/ahn/wms/v1_0"

# ───────────────────── Proj
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ──────────────── PlantWijs CSV
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "plantwijs_full.csv"),
    os.path.join(os.path.dirname(__file__), "plantwijs_full_v3.9.2.csv"),
    os.path.join(os.path.dirname(__file__), "plantwijs_full_v3.9.csv"),
]

# ───────────────────── helpers

def _get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("headers", HEADERS)
    kwargs.setdefault("timeout", 10)
    return requests.get(url, **kwargs)

# ───────────────────── FGR helpers

@dataclass
class FgrFeature:
    geom: List[Tuple[float, float]]  # lijst van (x,y)
    name: str

_FGR_CACHE: List[FgrFeature] = []


def _load_fgr() -> List[FgrFeature]:
    global _FGR_CACHE
    if _FGR_CACHE:
        return _FGR_CACHE
    print("[FGR] laden FGR geometrie…")
    feats: List[FgrFeature] = []
    url = PDOK_FGR_WFS + "&request=GetFeature&typenames=fysischgeografischeregios:fysischgeografischeregios&outputFormat=" + FMT_JSON
    r = _get(url)
    r.raise_for_status()
    data = r.json()
    for f in data.get("features", []):
        props = f.get("properties") or {}
        naam = props.get("NAAM") or props.get("naam") or "Onbekend"
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or []
        # geometrie kan MultiPolygon of Polygon zijn
        if geom.get("type") == "MultiPolygon":
            # flatten
            pl = []
            for poly in coords:
                for ring in poly:
                    pl.extend([(float(x), float(y)) for x, y in ring])
            if pl:
                feats.append(FgrFeature(pl, naam))
        elif geom.get("type") == "Polygon":
            pl = []
            for ring in coords:
                pl.extend([(float(x), float(y)) for x, y in ring])
            if pl:
                feats.append(FgrFeature(pl, naam))
    _FGR_CACHE = feats
    print(f"[FGR] geladen: {len(feats)} regio's")
    return feats


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # standaard ray-casting algo
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x1, y1 = poly[0]
    for i in range(1, n + 1):
        x2, y2 = poly[i % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside

# ───────────────────── WMS metadata

_WMSMETA: Dict[str, Dict[str, str]] = {}

def _capabilities(url: str) -> Optional[ET.Element]:
    try:
        r = _get(f"{url}?service=WMS&request=GetCapabilities")
        r.raise_for_status()
        return ET.fromstring(r.text)
    except Exception as e:
        print("[CAP] fout:", e)
        return None

# Zoek laag op Title of Name (case-insensitive substrings)
def _find_layer_name(url: str, want: List[str]) -> Optional[Tuple[str, str]]:
    root = _capabilities(url)
    if root is None:
        return None
    layers = root.findall(".//{http://www.opengis.net/wms}Layer")
    best = None
    for layer in layers:
        name_el = layer.find("{http://www.opengis.net/wms}Name")
        title_el = layer.find("{http://www.opengis.net/wms}Title")
        name = (name_el.text if name_el is not None else "") or ""
        title = (title_el.text if title_el is not None else "") or ""
        s = (name + " " + title).lower()
        if all(w.lower() in s for w in want):
            best = (name, title or name)
            break
    return best

def _resolve_layers() -> None:
    global _WMSMETA
    meta: Dict[str, Dict[str, str]] = {}
    fgr = _find_layer_name(FGR_WMS, ["fysisch", "fgr"]) or ("fysischgeografischeregios", "FGR")
    bodem = _find_layer_name(BODEM_WMS, ["bodemvlakken", "bodem"]) or ("Bodemvlakken", "Bodemvlakken")
    gt = _find_layer_name(GWD_WMS, ["grondwatertrappen", "gt"]) or ("BRO Grondwaterspiegeldiepte Grondwatertrappen Gt", "Gt")
    ghg = _find_layer_name(GWD_WMS, ["ghg"]) or ("BRO Grondwaterspiegeldiepte GHG", "GHG")
    glg = _find_layer_name(GWD_WMS, ["glg"]) or ("BRO Grondwaterspiegeldiepte GLG", "GLG")
    ahn = _find_layer_name(AHN_WMS, ["ahn", "dtm", "maaiveld"]) or ("ahn", "AHN")
    meta["fgr"] = {"url": FGR_WMS, "layer": fgr[0], "title": fgr[1]}
    meta["bodem"] = {"url": BODEM_WMS, "layer": bodem[0], "title": bodem[1]}
    meta["gt"] = {"url": GWD_WMS, "layer": gt[0], "title": gt[1]}
    meta["ghg"] = {"url": GWD_WMS, "layer": ghg[0], "title": ghg[1]}
    meta["glg"] = {"url": GWD_WMS, "layer": glg[0], "title": glg[1]}
    meta["ahn"] = {"url": AHN_WMS, "layer": ahn[0], "title": ahn[1]}
    _WMSMETA = meta

    print("[WMS] resolved:", meta)

_resolve_layers()

# ───────────────────── WFS/WMS helpers

def _wfs(url: str) -> List[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        if "json" not in r.headers.get("Content-Type", "").lower():
            return []
        return (r.json() or {}).get("features") or []
    except Exception as e:
        print("[WFS] fout:", e)
        return []

# WMS → featureinfo
_DEF_INFO_FORMATS = [
    "application/json",
    "application/vnd.ogc.gml",
    "application/vnd.ogc.wms_xml",
    "text/xml",
    "text/plain",
]


def _wms_getfeatureinfo(base_url: str, layer: str, lat: float, lon: float) -> dict | None:
    cx, cy = TX_WGS84_WEB.transform(lon, lat)
    m = 200.0
    bbox = f"{cx-m},{cy-m},{cx+m},{cy+m}"
    params_base = {
        "service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
        "layers": layer, "query_layers": layer, "styles": "",
        "crs": "EPSG:3857", "width": 101, "height": 101, "i": 50, "j": 50,
        "bbox": bbox,
    }
    params_base["feature_count"] = 10
    for fmt in _DEF_INFO_FORMATS:
        params = dict(params_base)
        params["info_format"] = fmt
        try:
            r = requests.get(base_url, params=params, headers=HEADERS, timeout=10)
            if not r.ok:
                continue
            ctype = r.headers.get("Content-Type", "").lower()
            if "json" in ctype:
                data = r.json() or {}
                feats = data.get("features") or []
                if feats:
                    props = feats[0].get("properties") or {}
                    if props:
                        return props
            text = r.text
            if text and fmt in ("text/plain", "text/xml", "application/vnd.ogc.gml"):
                return {"_text": text}
        except Exception:
            continue
    return None

# ───────────────────── PDOK value extractors

def fgr_from_point(lat: float, lon: float) -> str | None:
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None
    b = 100
    x1, y1, x2, y2 = round(x-b), round(y-b), round(x+b), round(y+b)
    feats = _load_fgr()
    for f in feats:
        for (px, py) in f.geom:
            if x1 <= px <= x2 and y1 <= py <= y2:
                if _point_in_poly(x, y, f.geom):
                    return f.name
    return None

_SOIL_MAP = {
    "zand": "zand",
    "lemig zand": "zand",
    "klei": "klei",
    "zavel": "leem",
    "leem": "leem",
    "veen": "veen",
    "moerig": "veen",
}

def _soil_from_text(s: str) -> Optional[str]:
    low = s.lower()
    for k, v in _SOIL_MAP.items():
        if k in low:
            return v
    return None

def _parse_kv_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out

def bodem_from_bodemkaart(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    layer = _WMSMETA.get("bodem", {}).get("layer") or "Bodemvlakken"
    props = _wms_getfeatureinfo(BODEM_WMS, layer, lat, lon) or {}

    # 1) JSON-properties
    for k in (
        "grondsoort", "bodem", "BODEM", "BODEMTYPE", "soil", "bodemtype", "SOILAREA_NAME", "NAAM",
        "first_soilname", "normal_soilprofile_name",
    ):
        if k in props and props[k]:
            val = str(props[k])
            return _soil_from_text(val) or val, props

    # 2) Tekst/GML fallback → key:value of vrije tekst scannen
    if "_text" in props:
        kv = _parse_kv_text(props["_text"]) or {}
        for k in ("grondsoort", "BODEM", "bodemtype", "BODEMNAAM", "NAAM", "omschrijving",
                  "first_soilname", "normal_soilprofile_name"):
            if k in kv and kv[k]:
                val = str(kv[k])
                return _soil_from_text(val) or val, props

    return None, props

_GT_CODE_MAP = {
    # Bron: BRO-documentatie → 1..19 naar Ia..VIIId
    "1": "Ia",  "2": "Ib",  "3": "IIa", "4": "IIb",
    "5": "IIIa","6": "IIIb","7": "IVa","8": "IVb",
    "9": "V",   "10":"VI",  "11":"VII","12":"VIII",
    "13":"IIIc","14":"IVc", "15":"Vb", "16":"VIb",
    "17":"VIIb","18":"VIIIb","19":"VIIIc",
}

def _gt_pretty(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # als het al iets als "IIIc" is, laat zo
    if re.match(r"^[IVX]+[a-z]?$", s, re.I):
        return s.upper()
    # anders value_list mapping
    if s in _GT_CODE_MAP:
        return _GT_CODE_MAP[s]
    return s

def _vochtklasse_from_gt_code(raw: Optional[str]) -> Optional[str]:
    code = _gt_pretty(raw)
    if not code:
        return None
    c = code.upper()
    if c.startswith(("I", "II")):
        return "zeer nat"
    if c.startswith(("III", "IV")):
        return "nat"
    if c.startswith("V"):
        return "vochtig"
    if c.startswith("VI"):
        return "droog"
    if c.startswith("VII") or c.startswith("VIII"):
        return "zeer droog"
    return None

def vocht_from_gwt(lat: float, lon: float) -> Tuple[Optional[str], dict, Optional[str]]:
    gt_layer = _WMSMETA.get("gt", {}).get("layer") or "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt"
    props = _wms_getfeatureinfo(GWD_WMS, gt_layer, lat, lon) or {}

    gt_raw: Optional[str] = None  # bv. 'IVc' of '8'

    # 1) directe property
    for k in ("gt", "grondwatertrap", "GT", "Gt"):
        if k in props and props[k]:
            gt_raw = str(props[k]).strip()
            break

    # 2) uit plaintext
    if not gt_raw and "_text" in props:
        kv = _parse_kv_text(props["_text"])
        for k in ("gt", "grondwatertrap", "GT"):
            if k in kv and kv[k]:
                gt_raw = str(kv[k]).strip()
                break
        if not gt_raw:
            m = re.search(r"\bGT\s*([IVX]+[a-z]?)\b", props["_text"], re.I)
            if m:
                gt_raw = m.group(1).strip()

    # 3) ordinaal uit value_list (1..19)
    if not gt_raw and "value_list" in props and str(props["value_list"]).strip():
        gt_raw = str(props["value_list"]).strip()

    klass = _vochtklasse_from_gt_code(gt_raw)

    # Fallback via GLG/GHG
    if not klass:
        for key in ("glg", "ghg"):
            lyr = _WMSMETA.get(key, {}).get("layer")
            if not lyr:
                continue
            p2 = _wms_getfeatureinfo(GWD_WMS, lyr, lat, lon) or {}
            txt = " ".join(str(v) for v in p2.values())
            m = re.search(r"(GLG|GHG)\s*[:=]?\s*(\d{1,3})", txt, re.I)
            depth = int(m.group(2)) if m else None
            if depth is not None:
                if depth < 25:   klass = "zeer nat"
                elif depth < 40: klass = "nat"
                elif depth < 80: klass = "vochtig"
                elif depth < 120:klass = "droog"
                else:            klass = "zeer droog"
                return klass, p2, _gt_pretty(gt_raw)

    return klass, props, _gt_pretty(gt_raw)


def ahn_from_point(lat: float, lon: float) -> Tuple[Optional[float], dict]:
    """Geef AHN-hoogte (m t.o.v. NAP) en ruwe properties terug voor de locatie."""
    meta = _WMSMETA.get("ahn") or {}
    base_url = meta.get("url") or AHN_WMS
    layer = meta.get("layer") or "ahn"
    props = _wms_getfeatureinfo(base_url, layer, lat, lon) or {}

    val: Optional[float] = None

    # 1) Probeer directe numerieke properties (bijv. GRAY_INDEX, value)
    for k, v in props.items():
        if k == "_text" or v is None:
            continue
        s = str(v).strip().replace(",", ".")
        try:
            val = float(s)
            break
        except ValueError:
            continue

    # 2) Fallback: tekst pars
    if val is None and "_text" in props:
        kv = _parse_kv_text(props.get("_text") or "") or {}
        for v in kv.values():
            s = str(v).strip().replace(",", ".")
            try:
                val = float(s)
                break
            except ValueError:
                continue

    return val, props

# ───────────────────── filtering helpers

def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()

def _split_tokens(cell: Any) -> List[str]:
    return [t.strip().lower() for t in str(cell or "").replace("/",";").replace("|",";").split(";") if t.strip()]

def _match_multival(cell: Any, choices: List[str]) -> bool:
    if not choices:
        return True
    tokens = set(_split_tokens(cell))
    want = set(w.strip().lower() for w in choices if w.strip())
    return bool(tokens.intersection(want))

def _match_bodem_row(row: pd.Series, keuzes: List[str]) -> bool:
    bodem = str(row.get("bodem") or "").strip().lower()
    if not keuzes:
        return True
    for k in keuzes:
        k = k.strip().lower()
        if not k:
            continue
        if k in bodem:
            return True
    return False

# ───────────────────── Data loading

@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";", dtype=str)
            print(f"[DATA] geladen: {path} ({len(df)} rijen)")
            return df
    raise FileNotFoundError("Geen plantwijs_full*.csv gevonden")

# ───────────────────── JSON cleaning

def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    return o

# ───────────────────── API: diagnose/meta
@app.get("/api/wms_meta")
def api_wms_meta():
    return JSONResponse(_clean(_WMSMETA))

@app.get("/api/diag/featureinfo")
def api_diag(service: str = Query(..., pattern="^(bodem|gt|ghg|glg|fgr)$"), lat: float = Query(...), lon: float = Query(...)):
    if service == "fgr":
        return JSONResponse({"fgr": fgr_from_point(lat, lon)})
    base = {"bodem": BODEM_WMS, "gt": GWD_WMS, "ghg": GWD_WMS, "glg": GWD_WMS}[service]
    layer = _WMSMETA.get(service, {}).get("layer")
    props = _wms_getfeatureinfo(base, layer, lat, lon)
    return JSONResponse(_clean({"base": base, "layer": layer, "props": props}))

# ───────────────────── API: data
@app.get("/api/plants")
def api_plants(
    q: str = Query("", description="Zoekterm in naam/wetenschappelijke naam"),
    licht: List[str] = Query([], description="Licht: schaduw, halfschaduw, zon"),
    vocht: List[str] = Query([], description="Vocht: zeer nat, nat, vochtig, droog, zeer droog"),
    bodem: List[str] = Query([], description="Bodem: zand, klei, leem, veen"),
    inheems_only: bool = Query(True, description="Alleen inheemse soorten"),
    exclude_invasief: bool = Query(True, description="Sluit invasieve soorten uit"),
    limit: int = Query(500, ge=1, le=1000),
):
    t0 = time.time()
    df = get_df()

    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]

    if q:
        qlow = q.lower()
        mask = (
            df["naam"].fillna("").str.lower().str.contains(qlow)
            | df["wetenschappelijke_naam"].fillna("").str.lower().str.contains(qlow)
        )
        df = df[mask]

    if licht:
        df = df[df["standplaats_licht"].map(lambda v: _match_multival(v, licht))]
    if vocht:
        df = df[df["vocht"].map(lambda v: _match_multival(v, vocht))]
    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]

    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]

    items = df[cols].head(limit).to_dict(orient="records")
    return JSONResponse(_clean({"count": int(len(df)), "items": items}))

@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    limit: int = Query(150, ge=1, le=1000),
):
    t0 = time.time()
    fgr = fgr_from_point(lat, lon) or "Onbekend"
    bodem_raw, props_bodem = bodem_from_bodemkaart(lat, lon)
    vocht_raw, props_gwt, gt_code = vocht_from_gwt(lat, lon)
    ahn_val, props_ahn = ahn_from_point(lat, lon)

    # Belangrijk: géén default op vocht als het onbekend is
    bodem_val = bodem_raw or "leem"  # bodem blijft zoals in v3.9
    vocht_val = vocht_raw  # None → géén filter

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]

    # Alleen filter toepassen als er een waarde is
    if vocht_val:
        df = df[df["vocht"].map(lambda v: _match_multival(v, [vocht_val]))]
    df = df[df.apply(lambda r: _match_bodem_row(r, [bodem_val]) if bodem_val else True, axis=1)]

    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_val,
        "bodem_bron": "BRO Bodemkaart WMS" if bodem_raw else "onbekend",
        "gt_code": gt_code,                     # ← altijd 'Ia'..'VIIId'
        "vocht": vocht_raw,                     # kan None zijn
        "vocht_bron": "BRO Gt/GLG WMS" if vocht_raw else "onbekend",
        "ahn": ahn_val,
        "ahn_bron": "AHN WMS" if ahn_val is not None else "onbekend",
        "advies": items,
        "elapsed_ms": int((time.time()-t0)*1000),
    }
    return JSONResponse(_clean(out))

# ───────────────────── UI
@app.get("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = r"""
<!doctype html>
<html lang=nl>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>PlantWijs v3.9.2</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --bg:#0b1321; --panel:#0f192e; --muted:#9aa4b2; --fg:#e6edf3; --border:#1c2a42; }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial; background:var(--bg); color:var(--fg); }
    header { padding:10px 14px; border-bottom:1px solid var(--border); background:#050815; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    header h1 { margin:0; font-size:18px; }
    .wrap { display:grid; grid-template-columns:minmax(520px, 68%) minmax(320px, 32%); gap:12px; padding:12px; height:calc(100vh - 64px); } /* alleen rechter kolom scrolt */
    #map { height:100%; border-radius:12px; border:1px solid var(--border); box-shadow:0 0 0 1px rgba(255,255,255,.05) inset; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px; }
    .panel-right { height:100%; overflow:auto; }
    .checks label { display:inline-flex; gap:6px; align-items:center; background:#0b1226; border:1px solid #1f2c49; padding:6px 8px; border-radius:8px; margin-right:6px; }
    input[type=checkbox] { accent-color:#5aa9ff; }
    .muted { color:var(--muted); }
    .chips { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
    .chip { background:#0b1226; border:1px solid #1f2c49; padding:4px 8px; border-radius:999px; font-size:12px; }

    /* Leaflet info/legenda control */
    .pw-ctl { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:12px; padding:10px; box-shadow:0 2px 12px rgba(0,0,0,.35); width:260px; }
    .pw-ctl h3 { margin:0 0 6px; font-size:14px; }
    .pw-ctl .sec { margin-top:8px; }
  </style>
</head>
<body>
  <header>
    <h1>🌿 PlantWijs</h1>
    <button id="btnLocate" class="btn" style="background:#11325a;color:#e6edf3;border:1px solid #27518a;padding:6px 10px;border-radius:8px;cursor:pointer;">📍 Mijn locatie</button>
    <label class="muted"><input id="inhOnly" type="checkbox" checked> alleen inheemse</label>
    <label class="muted"><input id="exInv" type="checkbox" checked> sluit invasieve uit</label>
    <div class="checks">
      <span class="muted">Licht:</span>
      <label><input type="checkbox" name="licht" value="schaduw"> schaduw</label>
      <label><input type="checkbox" name="licht" value="halfschaduw"> halfschaduw</label>
      <label><input type="checkbox" name="licht" value="zon"> zon</label>
    </div>
    <div class="checks">
      <span class="muted">Vocht:</span>
      <label><input type="checkbox" name="vocht" value="zeer nat"> zeer nat</label>
      <label><input type="checkbox" name="vocht" value="nat"> nat</label>
      <label><input type="checkbox" name="vocht" value="vochtig"> vochtig</label>
      <label><input type="checkbox" name="vocht" value="droog"> droog</label>
      <label><input type="checkbox" name="vocht" value="zeer droog"> zeer droog</label>
    </div>
    <div class="checks">
      <span class="muted">Bodem:</span>
      <label><input type="checkbox" name="bodem" value="zand"> zand</label>
      <label><input type="checkbox" name="bodem" value="klei"> klei</label>
      <label><input type="checkbox" name="bodem" value="leem"> leem</label>
      <label><input type="checkbox" name="bodem" value="veen"> veen</label>
    </div>
  </header>

  <div class="wrap">
    <div id="map"></div>
    <div class="panel panel-right">
      <div id="chips" class="chips"></div>
      <div class="muted" id="count"></div>
      <table id="tbl" style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">Naam</th>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">Wetenschappelijke naam</th>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">Licht</th>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">Vocht</th>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">Bodem</th>
            <th style="text-align:left;color:#b0b8c6;border-bottom:1px solid #182742;padding:8px 10px;">WHZ</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <script>
    const ui = { meta:null };
    const overlays = {};
    const map = L.map('map', {
      center:[52.1,5.2],
      zoom:8,
      zoomControl:true,
      attributionControl:false
    });

    // Basemap
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom:19,
      attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
    }).addTo(map);

    // Coole info/legenda control rechtsboven
    const InfoCtl = L.Control.extend({
      onAdd: function(map){
        const div = L.DomUtil.create('div', 'pw-ctl');
        div.innerHTML = `
          <h3>Legenda & info</h3>
          <div class="sec" id="clickInfo">
            <div id="uiF" class="muted">FGR: —</div>
            <div id="uiB" class="muted">Bodem: —</div>
            <div id="uiG" class="muted">Gt: —</div>
            <div id="uiH" class="muted">AHN hoogte: —</div>
          </div>
        `;
        L.DomEvent.disableClickPropagation(div);
        return div;
      }
    });
    const infoCtl = new InfoCtl({ position:'topright' }).addTo(map);

    function setClickInfo({fgr,bodem,bodem_bron,gt,vocht,ahn}){
      document.getElementById('uiF').textContent = 'FGR: ' + (fgr || '—');
      const btxt = (bodem || '—') + (bodem_bron ? ` (${bodem_bron})` : '');
      document.getElementById('uiB').textContent = 'Bodem: ' + btxt;
      document.getElementById('uiG').textContent = 'Gt: ' + (gt || '—') + (vocht ? ` → ${vocht}` : ' (onbekend)');
      const ahnTxt = (ahn == null) ? 'AHN hoogte: —' : `AHN hoogte: ${ahn.toFixed(2)} m NAP`;
      document.getElementById('uiH').textContent = ahnTxt;
    }


    
    async function loadWms(){
      ui.meta = await (await fetch('/api/wms_meta')).json();
      const make = (m, opacity)=> L.tileLayer.wms(m.url, { layers:m.layer, styles:'', format:'image/png', transparent:true, opacity: opacity, version:'1.3.0', crs: L.CRS.EPSG3857 });
      overlays['BRO Bodemkaart (Bodemvlakken)'] = make(ui.meta.bodem, 0.55).addTo(map);
      overlays['BRO Grondwatertrappen (Gt)']    = make(ui.meta.gt,    0.45).addTo(map);
      overlays['AHN hoogte']                    = make(ui.meta.ahn,   0.45).addTo(map);
      overlays['FGR']                           = make(ui.meta.fgr,   0.45).addTo(map);

      // Lagen-schakelaar: alleen overlays, linksonder
      const ctlLayers = L.control.layers({}, overlays, { collapsed:false, position:'bottomleft' }).addTo(map);

      // Verwijder eventuele base-sectie + separator (we tonen geen OSM hier)
      const cont = ctlLayers.getContainer();
      const baseList = cont.querySelector('.leaflet-control-layers-base');
      if(baseList) baseList.remove();
      const sep = cont.querySelector('.leaflet-control-layers-separator');
      if(sep) sep.remove();

      // Titel "Kaartlagen" boven de lijst plaatsen
      const overlaysList = cont.querySelector('.leaflet-control-layers-overlays');
      const title = document.createElement('div');
      title.textContent = 'Kaartlagen';
      title.style.fontWeight = '700';
      title.style.fontSize = '15px';
      title.style.margin = '6px 10px 6px 10px';
      title.style.color = '#000';
      overlaysList.parentNode.insertBefore(title, overlaysList); // boven de lijst
    }

    async function fetchList(){
      const url = new URL(location.origin + '/api/plants');
      if(document.getElementById('inhOnly').checked) url.searchParams.set('inheems_only','true');
      if(document.getElementById('exInv').checked)   url.searchParams.set('exclude_invasief','true');
      for(const v of getChecked('licht')) url.searchParams.append('licht', v);
      for(const v of getChecked('vocht')) url.searchParams.append('vocht', v);
      for(const v of getChecked('bodem')) url.searchParams.append('bodem', v);
      const res = await fetch(url);
      const data = await res.json();
      document.getElementById('count').textContent = (data.count||0) + ' resultaten';
      renderRows(data.items||[]);
    }

    function getChecked(name){
      return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(el=>el.value);
    }

    function renderRows(items){
      const tbody = document.querySelector('#tbl tbody');
      tbody.innerHTML = '';
      for(const it of items){
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.naam||''}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.wetenschappelijke_naam||''}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.standplaats_licht||''}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.vocht||''}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.bodem||''}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #182742;">${it.winterhardheidszone||''}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    map.on('click', async (e)=>{
      const { lat, lng:lon } = e.latlng;
      const url = new URL(location.origin + '/advies/geo');
      url.searchParams.set('lat', lat);
      url.searchParams.set('lon', lon);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      const j = await (await fetch(url)).json();

      setClickInfo({ fgr:j.fgr, bodem:j.bodem, bodem_bron:j.bodem_bron, gt:j.gt_code, vocht:j.vocht, ahn:j.ahn });

      const data = { items: j.advies || [] };
      document.getElementById('count').textContent = (data.items.length||0) + ' resultaten (auto-filter)';
      renderRows(data.items||[]);
    });

    document.getElementById('btnLocate').addEventListener('click', ()=>{
      if(!navigator.geolocation){ alert('Geolocatie niet ondersteund'); return; }
      navigator.geolocation.getCurrentPosition(async pos=>{
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        if(window._marker) window._marker.remove(); window._marker = L.marker([lat,lon]).addTo(map);
        map.fire('click', { latlng:{ lat, lng:lon } });
      }, err=>{ alert('Kon locatie niet ophalen'); });
    });

    for(const el of document.querySelectorAll('input[name="licht"], input[name="vocht"], input[name="bodem"], #inhOnly, #exInv')){
      el.addEventListener('change', fetchList);
    }

    loadWms().then(fetchList);
  </script>
</body>
</html>
"""
    return html
