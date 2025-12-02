
# PlantWijs API — v3.9.7
# - FIX: PDOK Locatieserver → nieuwe endpoint (api.pdok.nl … /search/v3_1) met CORS
# - UI: Kolomtitel opent filter; kolommen tonen/verbergen; sticky header; thema toggle; CSV/XLSX export
# - HTML triple-quoted string correct afgesloten
# Starten:
#   cd C:/PlantWijs
#   venv/Scripts/uvicorn api:app --reload --port 9000

from __future__ import annotations

import io
import math
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
import tempfile  # ← toevoegen
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pyproj import Transformer

# ───────────────────── PDOK endpoints
HEADERS = {"User-Agent": "plantwijs/3.9.7"}
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

# AHN WMS (Actueel Hoogtebestand Nederland, DTM 0.5m)
AHN_WMS = "https://service.pdok.nl/rws/ahn/wms/v1_0"

# BRO Geomorfologische kaart (GMM) WMS
GMM_WMS = "https://service.pdok.nl/bzk/bro-geomorfologischekaart/wms/v2_0"

# ───────────────────── Proj
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ───────────────────── Dataset cache
DATA_PATHS = [
    "out/plantwijs_full_semicolon.csv",
    "out/plantwijs_full.csv",
]

# Online CSV (GitHub raw) fallback
ONLINE_CSV_URLS = [
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full_semicolon.csv",
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full.csv",
]

_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None, "source": None}

# ───────────────────── Dataset cache
DATA_PATHS = [
    "out/plantwijs_full_semicolon.csv",
    "out/plantwijs_full.csv",
]
_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None}

# Online CSV fallback (GitHub raw)
ONLINE_CSV_URLS = [
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full.csv",
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full_semicolon.csv",
]

def _detect_sep(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"

def _load_df(path: str) -> pd.DataFrame:
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding_errors="ignore")
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    if "naam" not in df.columns and "nederlandse_naam" in df.columns:
        df = df.rename(columns={"nederlandse_naam": "naam"})
    if "wetenschappelijke_naam" not in df.columns:
        for k in ("taxon", "species"):
            if k in df.columns:
                df = df.rename(columns={k: "wetenschappelijke_naam"})
                break
    for must in ("standplaats_licht", "vocht", "inheems", "invasief"):
        if must not in df.columns:
            df[must] = ""
    return df

def _fetch_csv_online(url: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        text = r.content.decode("utf-8", errors="ignore")
        sep = ";" if text.count(";") >= text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, encoding_errors="ignore")
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        if "naam" not in df.columns and "nederlandse_naam" in df.columns:
            df = df.rename(columns={"nederlandse_naam": "naam"})
        if "wetenschappelijke_naam" not in df.columns:
            for k in ("taxon", "species"):
                if k in df.columns:
                    df = df.rename(columns={k: "wetenschappelijke_naam"})
                    break
        for must in ("standplaats_licht", "vocht", "inheems", "invasief"):
            if must not in df.columns:
                df[must] = ""
        return df
    except Exception as e:
        print("[ONLINE CSV] fout bij", url, "→", e)
        return None

def get_df() -> pd.DataFrame:
    # 1) Probeer lokaal (development)
    path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if path:
        m = os.path.getmtime(path)
        if _CACHE["df"] is None or _CACHE["mtime"] != m or _CACHE["path"] != path:
            df = _load_df(path)
            _CACHE.update({"df": df, "mtime": m, "path": path, "source": "local"})
            print(f"[DATA] geladen (lokaal): {path} — {len(df)} rijen, {df.shape[1]} kolommen")
        return _CACHE["df"].copy()

    # 2) Fallback: online CSV (GitHub raw)
    if _CACHE["df"] is not None and _CACHE.get("source") == "online":
        return _CACHE["df"].copy()

    for url in ONLINE_CSV_URLS:
        df = _fetch_csv_online(url)
        if df is not None and not df.empty:
            _CACHE.update({"df": df, "mtime": time.time(), "path": url, "source": "online"})
            print(f"[DATA] geladen (online): {url} — {len(df)} rijen, {df.shape[1]} kolommen")
            return _CACHE["df"].copy()

    # 3) Niets gevonden → duidelijke foutmelding
    raise FileNotFoundError(
        "Geen dataset gevonden. Lokaal ontbreekt out/plantwijs_full.csv én online CSV kon niet worden opgehaald."
    )

    # 2) Fallback: online CSV (GitHub raw)
    if _CACHE["df"] is not None and _CACHE.get("source") == "online":
        return _CACHE["df"].copy()

    for url in ONLINE_CSV_URLS:
        df = _fetch_csv_online(url)
        if df is not None and not df.empty:
            _CACHE.update({"df": df, "mtime": time.time(), "path": url, "source": "online"})
            print(f"[DATA] geladen (online): {url} — {len(df)} rijen, {df.shape[1]} kolommen")
            return _CACHE["df"].copy()

    # 3) Niets gevonden → duidelijke foutmelding
    raise FileNotFoundError(
        "Geen dataset gevonden. Lokaal ontbreekt out/plantwijs_full.csv én online CSV kon niet worden opgehaald."
    )
    
# ───────────────────── HTTP utils
@lru_cache(maxsize=32)
def _get(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=12)

@lru_cache(maxsize=16)
def _capabilities(url: str) -> Optional[ET.Element]:
    try:
        r = _get(f"{url}?service=WMS&request=GetCapabilities")
        r.raise_for_status()
        return ET.fromstring(r.text)
    except Exception as e:
        print("[CAP] fout:", e)
        return None

def _find_layer_name(url: str, want: List[str]) -> Optional[Tuple[str, str]]:
    root = _capabilities(url)
    if root is None:
        return None
    layers = root.findall(".//{*}Layer")
    cand: List[Tuple[str,str]] = []
    for layer in layers:
        name_el = layer.find("{*}Name")
        title_el = layer.find("{*}Title")
        name = (name_el.text if name_el is not None else "")
        title = (title_el.text if title_el is not None else "")
        if not name and not title:
            continue
        cand.append((name, title))
    lwant = [w.lower() for w in want]
    for name, title in cand:
        t = (title or "").lower()
        if any(w in t for w in lwant) and name:
            return name, title
    for name, title in cand:
        n = (name or "").lower()
        if any(w in n for w in lwant) and name:
            return name, title
    for name, title in cand:
        if name:
            return name, title
    return None

# Resolve alle laagnamen één keer bij startup
_WMSMETA: Dict[str, Dict[str, str]] = {}

def _resolve_layers() -> None:
    global _WMSMETA
    meta: Dict[str, Dict[str, str]] = {}
    fgr = _find_layer_name(FGR_WMS, ["fysisch", "fgr"]) or ("fysischgeografischeregios", "FGR")
    bodem = _find_layer_name(BODEM_WMS, ["bodemvlakken", "bodem"]) or ("Bodemvlakken", "Bodemvlakken")
    gt = _find_layer_name(GWD_WMS, ["grondwatertrappen", "gt"]) or ("BRO Grondwaterspiegeldiepte Grondwatertrappen Gt", "Gt")
    ghg = _find_layer_name(GWD_WMS, ["ghg"]) or ("BRO Grondwaterspiegeldiepte GHG", "GHG")
    glg = _find_layer_name(GWD_WMS, ["glg"]) or ("BRO Grondwaterspiegeldiepte GLG", "GLG")
    ahn = _find_layer_name(AHN_WMS, ["dtm_05m", "dtm", "ahn"]) or ("dtm_05m", "AHN hoogte (DTM 0.5m)")
    gmm = _find_layer_name(GMM_WMS, ["geomorfologische", "geomorphological"]) or ("geomorphological_area", "Geomorfologische kaart (GMM)")
    meta["fgr"] = {"url": FGR_WMS, "layer": fgr[0], "title": fgr[1]}
    meta["bodem"] = {"url": BODEM_WMS, "layer": bodem[0], "title": bodem[1]}
    meta["gt"] = {"url": GWD_WMS, "layer": gt[0], "title": gt[1]}
    meta["ghg"] = {"url": GWD_WMS, "layer": ghg[0], "title": ghg[1]}
    meta["glg"] = {"url": GWD_WMS, "layer": glg[0], "title": glg[1]}
    meta["ahn"] = {"url": AHN_WMS, "layer": ahn[0], "title": ahn[1]}
    meta["gmm"] = {"url": GMM_WMS, "layer": gmm[0], "title": gmm[1]}
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
        return (r.json() or {}).get("features", [])
    except Exception:
        return []

_kv_re = re.compile(r"^\s*([A-Za-z0-9_\-\. ]+?)\s*[:=]\s*(.+?)\s*$")

def _parse_kv_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        m = _kv_re.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    if not out:
        stripped = re.sub(r"<[^>]+>", "\n", text)
        for line in stripped.splitlines():
            m = _kv_re.match(line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
    return out

_DEF_INFO_FORMATS = [
    "application/json",
    "application/geo+json",
    "application/json;subtype=geojson",
    "application/vnd.ogc.gml",
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
    x1, y1, x2, y2 = round(x-b, 3), round(y-b, 3), round(x+b, 3), round(y+b, 3)
    layer_name = "fysischgeografischeregios:fysischgeografischeregios"
    url_rd = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={layer_name}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:28992&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs(url_rd)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={layer_name}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs(url_pt)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    return None

_SOIL_TOKENS = {
    "veen": {"veen"},
    "klei": {"klei", "zware klei", "lichte klei"},
    "leem": {"leem", "loess", "löss", "zavel"},
    "zand": {"zand", "dekzand"},
}

def _soil_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for soil, keys in _SOIL_TOKENS.items():
        for k in keys:
            if k in t:
                return soil
    return None

def bodem_from_bodemkaart(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    layer = _WMSMETA.get("bodem", {}).get("layer") or "Bodemvlakken"
    props = _wms_getfeatureinfo(BODEM_WMS, layer, lat, lon) or {}

    for k in (
        "grondsoort", "bodem", "BODEM", "BODEMTYPE", "soil", "bodemtype", "SOILAREA_NAME", "NAAM",
        "first_soilname", "normal_soilprofile_name",
    ):
        if k in props and props[k]:
            val = str(props[k])
            return _soil_from_text(val) or val, props

    if "_text" in props:
        kv = _parse_kv_text(props["_text"]) or {}
        for k in ("grondsoort", "BODEM", "bodemtype", "BODEMNAAM", "NAAM", "omschrijving",
                  "first_soilname", "normal_soilprofile_name"):
            if k in kv and kv[k]:
                val = kv[k]
                return _soil_from_text(val) or val, props
        so = _soil_from_text(props["_text"]) or None
        return so, props

    return None, props


def ahn_from_wms(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    """
    Haal een AHN-hoogte (DTM) op via de PDOK AHN WMS.
    Retourneert (hoogte_meter, raw_props) waarbij hoogte_meter als string is geformatteerd.
    """
    layer = _WMSMETA.get("ahn", {}).get("layer") or "dtm_05m"
    props = _wms_getfeatureinfo(AHN_WMS, layer, lat, lon) or {}

    def _first_numeric_value(d: dict) -> Optional[float]:
        for v in d.values():
            s = str(v).strip()
            if re.fullmatch(r"-?\d+(\.\d+)?", s):
                try:
                    return float(s)
                except Exception:
                    continue
        return None

    val: Optional[float] = None
    if props:
        val = _first_numeric_value(props)
    if val is None and "_text" in props:
        kv = _parse_kv_text(props.get("_text", "")) or {}
        val = _first_numeric_value(kv)
        if val is None:
            m = re.search(r"(-?\d+(?:\.\d+)?)", str(props.get("_text", "")))
            if m:
                try:
                    val = float(m.group(1))
                except Exception:
                    val = None

    if val is None:
        return None, props
    # Format met 2 decimalen; UI toont dit rechtstreeks
    return f"{val:.2f}", props


def gmm_from_wms(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    """
    Haal een geomorfologische eenheid op via de BRO Geomorfologische kaart (GMM) WMS.
    Retourneert (code_of_omschrijving, raw_props).
    """
    layer = _WMSMETA.get("gmm", {}).get("layer") or "geomorphological_area"
    props = _wms_getfeatureinfo(GMM_WMS, layer, lat, lon) or {}

    def _norm_key(k: str) -> str:
        return k.lower().replace("_", "").replace("-", "")

    def _first_from_keys(d: dict, candidates) -> Optional[str]:
        if not d:
            return None
        kl = { _norm_key(k): k for k in d.keys() }
        for wanted in candidates:
            want_norm = wanted.lower().replace("_", "").replace("-", "")
            for nk, orig in kl.items():
                if want_norm in nk:
                    v = d.get(orig)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if not s:
                        continue
                    # filter waardes die we expliciet niet willen tonen
                    sl = s.lower()
                    if sl == "nee":
                        continue
                    if s.lstrip().startswith("<?xml") or "msGMLOutput" in s:
                        continue
                    if sl.startswith("geom50000"):
                        continue
                    return s
        return None

    # 1) Probeer rechtstreeks uit props de landvormsubgroep-code te halen
    prefer_code_keys = [
        "landform_subgroup_code",
        "landformsubgroup_code",
        "landvormsubgroep_code",
        "landvormsubgroepcode",
    ]
    val: Optional[str] = _first_from_keys(props, prefer_code_keys)

    # 2) Zo niet, kijk of _text key/value-achtige info bevat
    if val is None and "_text" in props:
        kv = _parse_kv_text(props.get("_text", "")) or {}
        val = _first_from_keys(kv, prefer_code_keys)

    if not val:
        return None, props

    sval = str(val).strip()
    sl = sval.lower()
    if not sval or sl == "nee":
        return None, props
    if sval.lstrip().startswith("<?xml") or "msGMLOutput" in sval:
        return None, props
    if sl.startswith("geom50000"):
        return None, props

    return sval, props

# ───────────────────── PDOK value → vochtklasse
GT_ORDINAL_TO_CODE = {
    1:"Ia",  2:"Ib",  3:"IIa", 4:"IIb", 5:"IIc",
    6:"IIIa",7:"IIIb",
    8:"IVu", 9:"IVc",
    10:"Vao",11:"Vad",12:"Vbo",13:"Vbd",
    14:"VIo",15:"VId",
    16:"VIIo",17:"VIId",
    18:"VIIIo",19:"VIIId",
}

def _gt_pretty(gt: Optional[str]) -> Optional[str]:
    if not gt:
        return None
    s = str(gt).strip()
    if s.isdigit():
        try:
            v = int(float(s.replace(",", ".")))
        except Exception:
            return s
        return GT_ORDINAL_TO_CODE.get(v, s)
    return s.upper()

def _vochtklasse_from_gt_code(gt: Optional[str]) -> Optional[str]:
    if not gt:
        return None
    s = str(gt).strip()
    if s.isdigit():
        try:
            v = int(float(s.replace(",", ".")))
        except Exception:
            return None
        if 1 <= v <= 5:    return "zeer nat"
        if 6 <= v <= 7:    return "nat"
        if 8 <= v <= 13:   return "vochtig"
        if 14 <= v <= 15:  return "droog"
        if 16 <= v <= 19:  return "zeer droog"
        return None
    s_up = s.upper()
    m = re.match(r"^(I{1,3}|IV|V|VI|VII|VIII)", s_up)
    base = m.group(1) if m else s_up
    if base in ("I", "II"): return "zeer nat"
    if base == "III":       return "nat"
    if base in ("IV", "V"): return "vochtig"
    if base == "VI":        return "droog"
    if base in ("VII","VIII"): return "zeer droog"
    return None

def vocht_from_gwt(lat: float, lon: float) -> Tuple[Optional[str], dict, Optional[str]]:
    gt_layer = _WMSMETA.get("gt", {}).get("layer") or "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt"
    props = _wms_getfeatureinfo(GWD_WMS, gt_layer, lat, lon) or {}

    def _first_numeric(d: dict) -> Optional[str]:
        for k, v in d.items():
            ks = str(k).lower()
            if any(w in ks for w in ("value_list", "value", "class", "raster", "pixel", "waarde", "val")):
                s = str(v).strip()
                if re.fullmatch(r"\d+(\.\d+)?", s):
                    return s
        return None

    gt_raw: Optional[str] = None

    for k in ("gt", "grondwatertrap", "GT", "Gt"):
        if k in props and props[k]:
            gt_raw = str(props[k]).strip()
            break

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

    if not gt_raw:
        if "value_list" in props and str(props["value_list"]).strip():
            gt_raw = str(props["value_list"]).strip()
        if not gt_raw:
            hint = _first_numeric(props)
            if hint:
                gt_raw = hint

    klass = _vochtklasse_from_gt_code(gt_raw)

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

# ───────────────────── filtering helpers
def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()

def _split_tokens(cell: Any) -> List[str]:
    return [t.strip().lower()
            for t in re.split(r"[/|;,]+", str(cell or ""))
            if t.strip()]

_SOIL_CANON = {"zand", "klei", "leem", "veen"}
_RE_ALL = re.compile(r"\balle\s+grondsoorten\b", re.I)

def _canon_soil_token(tok: str) -> Optional[str]:
    t = str(tok or "").strip().lower()
    if not t:
        return None
    t = t.replace("ö", "o")
    if _RE_ALL.search(t):
        return "__ALL__"
    if re.search(r"\b(loess|loss|löss|leem|zavel)\b", t):
        return "leem"
    if re.search(r"\bdekzand\b|\bzand\b", t):
        return "zand"
    if re.search(r"\bklei\b", t):
        return "klei"
    if re.search(r"\bveen\b", t):
        return "veen"
    return None

def _ebben_grounds_to_cats(gs: Any) -> set[str]:
    raw = re.split(r"[|/;,]+", str(gs or ""))
    cats: set[str] = set()
    saw_all = False
    for r in raw:
        c = _canon_soil_token(r)
        if c == "__ALL__":
            saw_all = True
        elif c:
            cats.add(c)
    return set(_SOIL_CANON) if saw_all else cats

def _row_bodem_cats(row: pd.Series) -> set[str]:
    cats: set[str] = set()
    if "bodem" in row:
        for t in re.split(r"[|/;]+", str(row.get("bodem") or "")):
            c = _canon_soil_token(t)
            if c and c != "__ALL__":
                cats.add(c)
    cats |= _ebben_grounds_to_cats(row.get("grondsoorten", ""))
    return cats

def _match_bodem_row(row: pd.Series, keuzes: List[str]) -> bool:
    if not keuzes:
        return True
    want = {_canon_soil_token(k) or str(k).strip().lower() for k in keuzes}
    want = {w for w in want if w in _SOIL_CANON}
    if not want:
        return True
    have = _row_bodem_cats(row)
    return bool(have & want)

# ───────────────────── app + cleaners
app = FastAPI(title="PlantWijs API v3.9.7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k:_clean(v) for k,v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
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

# ───────────────────── filtering core
def _filter_plants_df(
    q: str,
    inheems_only: bool,
    exclude_invasief: bool,
    licht: List[str],
    vocht: List[str],
    bodem: List[str],
    sort: str,
    desc: bool,
) -> pd.DataFrame:
    df = get_df()

    def _has_any(cell: Any, choices: List[str]) -> bool:
        if not choices:
            return True
        tokens = {
            t.strip().lower()
            for t in re.split(r"[;/|]+", str(cell or ""))
            if t.strip()
        }
        want = {str(w).strip().lower() for w in choices if str(w).strip()}
        return bool(tokens & want)

    if q:
        df = df[df.apply(
            lambda r: _contains_ci(r.get("naam"), q) or _contains_ci(r.get("wetenschappelijke_naam"), q),
            axis=1
        )]

    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]

    if licht:
        df = df[df["standplaats_licht"].apply(lambda v: _has_any(v, licht))]
    if vocht:
        df = df[df["vocht"].apply(lambda v: _has_any(v, vocht))]
    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]

    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)

    return df

# ───────────────────── API: data
@app.get("/api/plants")
def api_plants(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    limit: Optional[int] = Query(None),  # genegeerd → geen limiet
    sort: str = Query("naam"),
    desc: bool = Query(False),
):
    df = _filter_plants_df(q, inheems_only, exclude_invasief, licht, vocht, bodem, sort, desc)
    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "ellenberg_l_min","ellenberg_l_max","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t_min","ellenberg_t_max","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r_min","ellenberg_r_max","ellenberg_s_min","ellenberg_s_max",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]
    items = df[cols].to_dict(orient="records")
    return JSONResponse(_clean({"count": int(len(df)), "items": items}))

# ───────────────────── API: export
@app.get("/export/csv")
def export_csv(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    sort: str = Query("naam"),
    desc: bool = Query(False),
):
    df = _filter_plants_df(q, inheems_only, exclude_invasief, licht, vocht, bodem, sort, desc)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = "plantwijs_export.csv"
    return StreamingResponse(iter([buf.getvalue()]),
                             media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.get("/export/xlsx")
def export_xlsx(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    sort: str = Query("naam"),
    desc: bool = Query(False),
):
    df = _filter_plants_df(q, inheems_only, exclude_invasief, licht, vocht, bodem, sort, desc)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="PlantWijs")
    buf.seek(0)
    filename = "plantwijs_export.xlsx"
    return StreamingResponse(buf,
                             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})

# (Optioneel maar handig) Admin-reload endpoint
@app.get("/api/admin/reload")
def api_admin_reload(key: str = Query(...)):
    """Wist de in-memory cache en haalt de remote CSV opnieuw op.
    Beveiligd met een simpele key in env: PLANTWIJS_ADMIN_KEY
    """
    admin_key = os.getenv("PLANTWIJS_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    try:
        _CACHE.update({"df": None, "mtime": None, "path": None})
        if DATA_URL:
            _download_if_needed(DATA_URL)  # prefetch
        return JSONResponse({"ok": True, "msg": "dataset cache cleared/refreshed"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ───────────────────── API: advies/geo
@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    limit: Optional[int] = Query(None),  # genegeerd
):
    t0 = time.time()
    fgr = fgr_from_point(lat, lon) or "Onbekend"
    bodem_raw, _props_bodem = bodem_from_bodemkaart(lat, lon)
    vocht_raw, _props_gwt, gt_code = vocht_from_gwt(lat, lon)
    ahn_val, _props_ahn = ahn_from_wms(lat, lon)
    gmm_val, _props_gmm = gmm_from_wms(lat, lon)

    bodem_val = bodem_raw
    vocht_val = vocht_raw

    def _has_any(cell: Any, choices: List[str]) -> bool:
        if not choices:
            return True
        tokens = {t.strip().lower() for t in re.split(r"[;/|]+", str(cell or "")) if t.strip()}
        want = {w.strip().lower() for w in choices if str(w).strip()}
        return bool(tokens & want)

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]

    if vocht_val:
        df = df[df["vocht"].apply(lambda v: _has_any(v, [vocht_val]))]
    if bodem_val:
        df = df[df.apply(lambda r:
                         _has_any(r.get("bodem", ""), [bodem_val]) or
                         _has_any(r.get("grondsoorten", ""), [bodem_val]),
                         axis=1)]

    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief",
        "standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]
    items = df[cols].to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_val,
        "bodem_bron": "BRO Bodemkaart WMS" if bodem_raw else "onbekend",
        "gt_code": gt_code,
        "vocht": vocht_raw,
        "vocht_bron": "BRO Gt/GLG WMS" if vocht_raw else "onbekend",
        "ahn": ahn_val,
        "ahn_bron": "PDOK AHN WMS (DTM 0.5m)" if ahn_val else "onbekend",
        "gmm": gmm_val,
        "gmm_bron": "BRO Geomorfologische kaart (GMM) WMS" if gmm_val else "onbekend",
        "advies": items,
        "elapsed_ms": int((time.time()-t0)*1000),
    }
    return JSONResponse(_clean(out))

# ───────────────────── UI
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = '''
<!doctype html>
<html lang=nl>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>PlantWijs v3.9.7</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --bg:#0b1321; --panel:#0f192e; --muted:#9aa4b2; --fg:#e6edf3; --border:#1c2a42; }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial; background:var(--bg); color:var(--fg); }
    header { padding:10px 14px; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg); z-index:10; display:flex; gap:10px; align-items:center; justify-content:space-between; }
    header h1 { margin:0; font-size:18px; }
   /* Mobile-first: 1 kolom, map bovenaan, paneel eronder */
.wrap {
  display:grid;
  grid-template-columns:1fr;
  grid-auto-rows:auto;
  gap:12px;
  padding:12px;
  /* geen geforceerde vaste hoogte op mobiel; laat de pagina scrollen */
  min-height:calc(100vh - 56px);
}

/* Map: op mobiel ~halve viewport hoogte */
#map {
  height:55vh;               /* prettige hoogte op mobiel */
  min-height:320px;          /* zodat het nooit te klein wordt */
  border-radius:12px;
  border:1px solid var(--border);
  box-shadow:0 0 0 1px rgba(255,255,255,.05) inset;
  position:relative;
}

/* Paneel: op mobiel gewoon mee in de flow */
.panel-right {
  height:auto;
  overflow:visible;
}

/* Zoekbalk control: breedte schaalt mee op mobiel */
.pw-search { width:min(92vw, 320px); margin:8px 8px 0 8px; }

/* Vanaf 900px → 2 kolommen en full-height layout zoals desktop */
@media (min-width: 900px) {
  .wrap {
    grid-template-columns:1fr 1fr;
    height:calc(100vh - 56px);
  }
  #map { height:100%; }
  .panel-right { height:100%; overflow:auto; }
}

/* Extra: op hele brede schermen map iets breder dan paneel */
@media (min-width: 1400px) {
  .wrap { grid-template-columns:1.2fr 1fr; }
}

    .panel-right { height:100%; overflow:auto; }
    .muted { color:var(--muted); }

    .leaflet-control.pw-locate { background:transparent; border:0; box-shadow:none; }
    .pw-locate-btn { width:36px; height:36px; border-radius:999px; border:1px solid #1f2c49; background:#0c1730; color:#e6edf3; display:flex; align-items:center; justify-content:center; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.35); }
    .pw-locate-btn:hover { background:#13264a; }

    .pw-ctl { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:12px; padding:10px; box-shadow:0 2px 12px rgba(0,0,0,.35); width:260px; }
    .pw-ctl h3 { margin:0 0 6px; font-size:14px; }
    .pw-ctl .sec { margin-top:8px; }

    /* Zoekbalk (topleft, boven zoom) */
    .pw-search {
      background:var(--panel); color:var(--fg);
      border:1px solid var(--border); border-radius:10px;
      padding:8px; width:260px;
      box-shadow:0 2px 12px rgba(0,0,0,.35);
    }
    .pw-search input {
      width:100%; padding:6px 8px;
      border:1px solid var(--border); border-radius:6px;
      background:transparent; color:inherit;
    }
    .pw-sugg { margin-top:6px; max-height:240px; overflow:auto; }
    .pw-sugg div { padding:6px 8px; border-radius:6px; cursor:pointer; }
    .pw-sugg div:hover { background:rgba(255,255,255,.06); }

    .filters { display:block; margin-bottom:10px; }
    .filters .group { margin:8px 0 0; }
    .filters .title { display:block; font-weight:600; margin-bottom:6px; }
    .checks { display:flex; gap:6px; flex-wrap:wrap; }
    .checks label { display:inline-flex; gap:6px; align-items:center; background:#0c1730; border:1px solid #1f2c49; padding:6px 8px; border-radius:8px; }
    input[type=checkbox] { accent-color:#5aa9ff; }
    .hint { font-size:12px; color:var(--muted); margin-top:4px; }

    .more-toggle { width:100%; margin:10px 0 0; background:#0c1730; border:1px solid #1f2c49; padding:6px 10px; border-radius:8px; display:flex; align-items:center; justify-content:space-between; cursor:pointer; user-select:none; }
    .more-toggle span.arrow { font-size:12px; }
    #moreFilters { display:none; margin-top:8px; }
    #moreFilters.open { display:block; }

    #filterStatus { margin:6px 0 10px; }
    .flag { display:inline-flex; gap:8px; align-items:flex-start; padding:8px 10px; border-radius:8px; border:1px solid; }
    .flag.ok   { color:#38d39f; border-color:rgba(56,211,159,.35); background:rgba(56,211,159,.08); }
    .flag.warn { color:#ff6b6b; border-color:rgba(255,107,107,.35); background:rgba(255,107,107,.08); }
    .flag .icon { line-height:1; }
    .flag .text { color:inherit; }

    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:8px 0 10px; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; }
    .btn { background:#0c1730; border:1px solid #1f2c49; color:var(--fg); padding:6px 10px; border-radius:8px; cursor:pointer; }
    .btn:hover { background:#13264a; }
    .btn-ghost { background:transparent; color:var(--fg); border:1px solid var(--border); padding:6px 10px; border-radius:8px; cursor:pointer; }
    .btn-ghost:hover { background:rgba(255,255,255,.06); }

    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px 10px; border-bottom:1px solid #182742; text-align:left; vertical-align:top; }
    thead th { color:#b0b8c6; position:sticky; top:0; z-index:1; background:var(--panel); }
    th .th-wrap { display:flex; align-items:center; gap:6px; }
    th.col-filter { cursor:pointer; }
    th.col-filter .th-text::after { content:"▾"; font-size:11px; opacity:.65; margin-left:6px; }
    .dropdown { position:fixed; display:none; z-index:9999; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:8px; max-height:260px; overflow:auto; box-shadow:0 6px 24px rgba(0,0,0,.35); min-width:220px; }
    .dropdown.show { display:block; }
    .dropdown h4 { margin:0 0 6px; font-size:13px; }
    .dropdown .opt { display:flex; align-items:center; gap:6px; margin:4px 0; }
    .dropdown .actions { display:flex; gap:6px; margin-top:8px; }
    .dropdown .actions .btn { padding:4px 8px; }

    #colMenu { position:fixed; display:none; z-index:9999; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:8px; box-shadow:0 6px 24px rgba(0,0,0,.35); min-width:240px; }
    #colMenu.show { display:block; }
    #colMenu .opt { display:flex; align-items:center; gap:8px; margin:6px 0; }

    body.light {
      --bg:#f6f8fc; --panel:#ffffff; --muted:#667085; --fg:#111827; --border:#e5e7eb;
    }
    body.light .checks label,
    body.light .more-toggle { background:#f2f4f7; border-color:#e5e7eb; }
    body.light .pw-ctl { background:#ffffff; border-color:#e5e7eb; }
    body.light .pw-locate-btn { background:#f2f4f7; color:#111827; border-color:#e5e7eb; }
    body.light .pw-locate-btn:hover { background:#eaeef3; }
    body.light .btn { background:#f2f4f7; color:#111827; border-color:#e5e7eb; }
    body.light .btn:hover { background:#eaeef3; }
    body.light .btn-ghost { border-color:#e5e7eb; }
    body.light thead th { background:#ffffff; color:#475569; }
    
    /* Leaflet controls theming (zoom + layers) */
.leaflet-control-zoom,
.leaflet-control-layers {
  background: var(--panel) !important;
  color: var(--fg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0,0,0,.35);
}

/* Zoom knoppen */
.leaflet-bar a,
.leaflet-bar a:focus {
  background: var(--panel) !important;
  color: var(--fg) !important;
  border-bottom: 1px solid var(--border) !important;
  box-shadow: none !important;
}
.leaflet-bar a:last-child { border-bottom: 0 !important; }
.leaflet-bar a:hover { background: #13264a !important; } /* dark hover */
body.light .leaflet-bar a:hover { background: #eaeef3 !important; } /* light hover */

/* Layers control (uitgeklapt) */
.leaflet-control-layers-expanded {
  padding: 8px !important;
}
.leaflet-control-layers-list,
.leaflet-control-layers label {
  color: var(--fg) !important;
}
.leaflet-control-layers-separator {
  border-top-color: var(--border) !important;
}
.leaflet-control-layers-overlays label {
  display: flex; gap: 8px; align-items: center;
}
.leaflet-control-layers input.leaflet-control-layers-selector {
  accent-color: #5aa9ff; /* match je overige checkboxes */
}

/* Light mode fine-tuning (schaduw iets subtieler) */
body.light .leaflet-control-zoom,
body.light .leaflet-control-layers {
  box-shadow: 0 2px 12px rgba(0,0,0,.12);
}
/* --- Responsive tuning voor Leaflet controls --- */
.leaflet-control { font-size: 13px; }
.leaflet-control-layers { max-width: 360px; }
.leaflet-control-layers-expanded {
  width: clamp(220px, 80vw, 360px);
  max-height: 45vh;
  overflow: auto;
}

/* Mobiel: compact, niet overlappen */
@media (max-width: 768px) {
  /* randen wat dichter op het scherm */
  .leaflet-top.leaflet-right  { margin-right: 8px; }
  .leaflet-top.leaflet-left   { margin-left:  8px; }
  .leaflet-bottom.leaflet-right,
  .leaflet-bottom.leaflet-left { margin-bottom: 8px; }

  /* kleinere zoomknoppen */
  .leaflet-control-zoom a { width: 32px; height: 32px; line-height: 32px; }

  /* zoekcontrol smaller */
  .pw-search { width: min(92vw, 320px); padding: 6px; }
  .pw-search input { padding: 6px 8px; }

  /* legenda & info compacter */
  .pw-ctl { width: min(70vw, 240px); padding: 8px; }
  .pw-ctl h3 { font-size: 13px; }
  .pw-ctl .sec { font-size: 12px; }
}
/* ——— Mobile layout (≤768px) ——— */
.legend-inline{ display:none; }  /* default verborgen; alleen mobiel tonen */
@media (max-width: 768px){
  .wrap { grid-template-columns: 1fr; height:auto; }
  #map { height: 62vh; }

  /* zoekbalk compacter linksboven */
  .pw-search { width: 210px; padding:6px; border-radius:8px; }
  .pw-search input { padding:5px 7px; font-size:14px; }

  /* verberg de zwevende legenda op de kaart */
  .leaflet-control.pw-ctl { display:none; }

  /* toon de legenda onder de kaart als paneel */
  .legend-inline{ display:block; margin:10px 0 14px; }

  /* wat lucht aan de randen van knoppen */
  .leaflet-control { margin: 8px; }
}
/* Mobiel: verberg de in-kaart legenda (InfoCtl) */
@media (max-width: 768px){
  .leaflet-control.pw-ctl { display: none !important; }
}

  </style>
</head>
<body>
  <header>
    <h1>🌿 PlantWijs</h1>
    <button id="btnTheme" class="btn-ghost" title="Schakel licht/donker">🌓 Thema</button>
  </header>

 <div class="wrap">
  <div id="map"></div>

  <!-- Mobiele legenda (staat buiten/onder de kaart); desktop: verborgen -->
  <div id="legendInline" class="panel legend-inline" aria-live="polite">
    <h3>Legenda &amp; info</h3>
    <div id="uiF2" class="muted">Fysisch Geografische Regio's: —</div>
    <div id="uiB2" class="muted">Bodem: —</div>
    <div id="uiG2" class="muted">Gt: —</div>
    <div id="uiH2" class="muted">AHN (m): —</div>
    <div id="uiM2" class="muted">Geomorfologie (GMM): —</div>
  </div>

    <div class="panel panel-right">
      <div class="filters">
        <div class="group">
          <span class="title">Licht</span>
          <div class="checks" id="lichtChecks">
            <label><input type="checkbox" name="licht" value="schaduw"> schaduw</label>
            <label><input type="checkbox" name="licht" value="halfschaduw"> halfschaduw</label>
            <label><input type="checkbox" name="licht" value="zon"> zon</label>
          </div>
          <div class="hint">Selecteer hier het lichtniveau van de locatie voor een duidelijker en beter passend resultaat.</div>
        </div>

        <div id="moreBar" class="more-toggle" title="Meer filters tonen/verbergen">
          <strong>Meer filters en opties</strong><span class="arrow">▾</span>
        </div>

        <div id="moreFilters">
          <div class="group">
            <span class="title">Vocht</span>
            <div class="checks">
              <label><input type="checkbox" name="vocht" value="zeer droog"> zeer droog</label>
              <label><input type="checkbox" name="vocht" value="droog"> droog</label>
              <label><input type="checkbox" name="vocht" value="vochtig"> vochtig</label>
              <label><input type="checkbox" name="vocht" value="nat"> nat</label>
              <label><input type="checkbox" name="vocht" value="zeer nat"> zeer nat</label>
            </div>
            <div class="hint">Wijkt de vochttoestand op de gekozen plek af van wat de kaarten aangeven? Kies hier een waarde om de kaartwaarde te overschrijven.</div>
          </div>

          <div class="group">
            <span class="title">Bodem</span>
            <div class="checks">
              <label><input type="checkbox" name="bodem" value="zand"> zand</label>
              <label><input type="checkbox" name="bodem" value="klei"> klei</label>
              <label><input type="checkbox" name="bodem" value="leem"> leem</label>
              <label><input type="checkbox" name="bodem" value="veen"> veen</label>
            </div>
            <div class="hint">Komt het bodemtype ter plekke niet overeen met de kaart? Selecteer hier een bodem om de kaartwaarde te overschrijven.</div>
          </div>

          <div class="group">
            <span class="title">Opties</span>
            <div class="checks">
              <label class="muted"><input id="inhOnly" type="checkbox" checked> alleen inheemse</label>
              <label class="muted"><input id="exInv" type="checkbox" checked> sluit invasieve uit</label>
            </div>
          </div>
        </div>
      </div>

      <div class="toolbar">
        <div class="muted" id="count"></div>
        <div class="actions">
          <button id="btnCols" class="btn-ghost" title="Kolommen tonen/verbergen">☰ Kolommen</button>
          <button id="btnCSV" class="btn" title="Exporteer huidige selectie als CSV">⬇️ CSV</button>
          <button id="btnXLSX" class="btn" title="Exporteer huidige selectie als Excel">⬇️ Excel</button>
        </div>
      </div>

      <div id="filterStatus"></div>

      <div id="colMenu"></div>

      <div id="colFilterMenu" class="dropdown"></div>

      <table id="tbl">
        <thead><tr id="theadRow"></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <script>
  const map = L.map('map').setView([52.1, 5.3], 8);
  const isMobile = window.matchMedia('(max-width: 768px)').matches;
// Zoomknoppen linksonder op mobiel
if (isMobile) map.zoomControl.setPosition('bottomleft');

  // ⬇️ NIEUW: simpele mobiele-vlag
  const IS_MOBILE = window.matchMedia('(max-width: 768px)').matches;

    // Zorg dat Leaflet z'n grootte herkent bij layout/rotatie
function fixMapSize(){ setTimeout(()=> map.invalidateSize(), 60); }
window.addEventListener('resize', fixMapSize);
window.addEventListener('orientationchange', fixMapSize);
// eerste keer na opbouwen
setTimeout(fixMapSize, 0);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap' }).addTo(map);

    let overlays = {};
    let ui = { meta:null, ctx:{ vocht:null, bodem:null } };
    window._lastQuery = new URLSearchParams();
    let _lastItems = [];

    const COLS_KEY = 'pw_cols_visible_v2';
    const DEFAULT_COLS = [
      {key:'naam', label:'Naam', filterable:false, visible:true},
      {key:'wetenschappelijke_naam', label:'Wetenschappelijke naam', filterable:false, visible:true},
      {key:'standplaats_licht', label:'Licht', filterable:true, visible:true},
      {key:'vocht', label:'Vocht', filterable:true, visible:true},
      {key:'bodem', label:'Bodem', filterable:true, visible:true},
      {key:'winterhardheidszone', label:'WHZ', filterable:true, visible:true},
      {key:'grondsoorten', label:'Grondsoorten', filterable:true, visible:false},
      {key:'inheems', label:'Inheems', filterable:true, visible:false},
      {key:'invasief', label:'Invasief', filterable:true, visible:false},
    ];
    let COLS = JSON.parse(localStorage.getItem(COLS_KEY) || 'null') || DEFAULT_COLS;

    const headerFilters = new Map();

    function html(s){ return (s==null?'':String(s)).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;') }
    function getChecked(name){ return Array.from(document.querySelectorAll('input[name="'+name+'"]:checked')).map(x=>x.value) }
    function tokSplit(val){ return String(val??'').split(/[/|;,]+/).map(s=>s.trim()).filter(Boolean); }

    function computeUsage(){
      const chosenL = getChecked('licht');
      const chosenV = getChecked('vocht');
      const chosenB = getChecked('bodem');
      const useL = chosenL.length > 0;
      const useV = chosenV.length > 0 || !!(ui.ctx && ui.ctx.vocht);
      const useB = chosenB.length > 0 || !!(ui.ctx && ui.ctx.bodem);
      return { useL, useV, useB, chosenL, chosenV, chosenB };
    }

    (function themeInit(){
      const key = 'pw_theme';
      const apply = t => { document.body.classList.toggle('light', t === 'light'); };
      const saved = localStorage.getItem(key) || 'dark';
      apply(saved);
      document.getElementById('btnTheme')?.addEventListener('click', ()=>{
        const now = document.body.classList.contains('light') ? 'dark' : 'light';
        localStorage.setItem(key, now); apply(now);
      });
    })();

    const LocateCtl = L.Control.extend({
      options:{ position:'bottomright' },
      onAdd: function() {
        const div = L.DomUtil.create('div', 'leaflet-control pw-locate');
        const btn = L.DomUtil.create('button', 'pw-locate-btn', div);
        btn.type = 'button'; btn.title = 'Mijn locatie'; btn.textContent = '📍';
        L.DomEvent.on(btn, 'click', (e)=>{
          L.DomEvent.stop(e);
          if(!navigator.geolocation){ alert('Geolocatie niet ondersteund.'); return; }
          navigator.geolocation.getCurrentPosition(pos=>{
            const lat = pos.coords.latitude, lon = pos.coords.longitude;
            map.setView([lat,lon], 14);
            if(window._marker) window._marker.remove();
            window._marker = L.marker([lat,lon]).addTo(map);
            map.fire('click', { latlng:{ lat, lng:lon } });
          }, err=>{ alert('Kon locatie niet ophalen'); });
        });
        return div;
      }
    });

    // ───────────────────────────────── PDOK Locatieserver zoek-control (topleft, boven zoom)
    const PDOKSearch = L.Control.extend({
      options: { position: 'topleft' },
      onAdd: function(map) {
        const div = L.DomUtil.create('div', 'pw-search');
        div.innerHTML = `
          <input id="pwSearchInput" type="text" placeholder="Zoek adres of plaats…" autocomplete="off">
          <div id="pwSugg" class="pw-sugg"></div>
        `;
        const inp = div.querySelector('#pwSearchInput');
        const box = div.querySelector('#pwSugg');

        // Officiële nieuwe endpoint met CORS
        const PDOK_BASE = 'https://api.pdok.nl/bzk/locatieserver/search/v3_1';

        // Klein abort/time-out mechanisme zodat oude requests worden afgebroken
        let lastCtrl = null;
        function fetchJSON(url){
          if(lastCtrl) lastCtrl.abort();
          lastCtrl = new AbortController();
          const id = setTimeout(()=> lastCtrl.abort(), 8000);
          return fetch(url, { mode:'cors', headers:{ 'Accept':'application/json' }, signal:lastCtrl.signal })
            .finally(()=> clearTimeout(id))
            .then(r => {
              if(!r.ok) throw new Error('HTTP '+r.status);
              return r.json();
            });
        }

        let t = null;

        function labelFromDoc(d){
          const s = (d.weergavenaam || d.weergaveNaam || '').replace(/, Nederland$/,'');
          return s || (d.type || d.typeGebied || d.bron || '');
        }

        async function suggest(q){
          if(!q || q.length < 3){ box.innerHTML=''; return; }
          try{
            const url = `${PDOK_BASE}/suggest?rows=10&q=${encodeURIComponent(q)}`;
            const j = await fetchJSON(url);
            const docs = (j.response && j.response.docs) ? j.response.docs : [];
            if(!docs.length){ box.innerHTML = `<div class="muted">Geen resultaten</div>`; return; }
            box.innerHTML = docs.map(d=>`<div data-id="${d.id}">${html(labelFromDoc(d))}</div>`).join('');
            box.querySelectorAll('div[data-id]').forEach(el=>{
              el.addEventListener('click', ()=> selectById(el.getAttribute('data-id'), el.textContent));
            });
          }catch(e){
            box.innerHTML = `<div class="muted">Zoeken mislukt</div>`;
            console.error('[PDOK] suggest error', e);
          }
        }

        async function selectById(id, displayText){
          try{
            const url = `${PDOK_BASE}/lookup?id=${encodeURIComponent(id)}`;
            const j = await fetchJSON(url);
            const doc = (j.response && j.response.docs && j.response.docs[0]) ? j.response.docs[0] : null;
            if(doc && doc.centroide_ll){
              const m = /POINT\\(([-0-9.]+)\\s+([-0-9.]+)\\)/.exec(doc.centroide_ll);
              if(m){
                const lon = parseFloat(m[1]), lat = parseFloat(m[2]);
                map.setView([lat,lon], 15);
                if(window._marker) window._marker.remove();
                window._marker = L.marker([lat,lon]).addTo(map);
                map.fire('click', { latlng:{ lat, lng:lon } }); // triggert je advies-flow
                box.innerHTML=''; inp.value = displayText || labelFromDoc(doc);
              }
            }
          }catch(e){
            console.error('[PDOK] lookup error', e);
          }
        }

        async function freeSearch(q){
          if(!q) return;
          try{
            const url = `${PDOK_BASE}/free?rows=1&q=${encodeURIComponent(q)}`;
            const j = await fetchJSON(url);
            const doc = (j.response && j.response.docs && j.response.docs[0]) ? j.response.docs[0] : null;
            if(doc && doc.centroide_ll){
              const m = /POINT\\(([-0-9.]+)\\s+([-0-9.]+)\\)/.exec(doc.centroide_ll);
              if(m){
                const lon = parseFloat(m[1]), lat = parseFloat(m[2]);
                map.setView([lat,lon], 15);
                if(window._marker) window._marker.remove();
                window._marker = L.marker([lat,lon]).addTo(map);
                map.fire('click', { latlng:{ lat, lng:lon } });
                box.innerHTML=''; inp.value = labelFromDoc(doc) || q;
              }
            }else{
              box.innerHTML = `<div class="muted">Geen resultaten</div>`;
            }
          }catch(e){
            console.error('[PDOK] free error', e);
            box.innerHTML = `<div class="muted">Zoekfout</div>`;
          }
        }

        // Typen → suggest (met debounce)
        inp.addEventListener('input', ()=>{
          clearTimeout(t);
          const q = inp.value.trim();
          t = setTimeout(()=> suggest(q), 250);
        });

        // Enter → pak 1e suggestie, anders free search
        inp.addEventListener('keydown', (ev)=>{
          if(ev.key === 'Enter'){
            ev.preventDefault();
            const first = box.querySelector('div[data-id]');
            if(first){ first.click(); }
            else{ freeSearch(inp.value.trim()); }
          }
        });

        // Houd focus (klik in suggesties sluit input niet)
        box.addEventListener('mousedown', e => e.preventDefault());

        // Leaflet: niet de kaart laten pannen bij interactie met deze control
        L.DomEvent.disableClickPropagation(div);

        // Plaats boven de zoomknoppen
        setTimeout(()=>{
          const corner = map._controlCorners && map._controlCorners['topleft'];
          const zoom = map.zoomControl && map.zoomControl.getContainer ? map.zoomControl.getContainer() : (corner?.querySelector('.leaflet-control-zoom'));
          if(corner && zoom && div.parentNode === corner){
            corner.insertBefore(div, zoom);
          }
        }, 0);

        return div;
      }
    });

    map.addControl(new LocateCtl());
    map.addControl(new PDOKSearch());

    const InfoCtl = L.Control.extend({
      onAdd: function() {
        const div = L.DomUtil.create('div', 'pw-ctl');
        div.innerHTML = `
          <h3>Legenda & info</h3>
          <div class="sec" id="clickInfo">
            <div id="uiF" class="muted">Fysisch Geografische Regio's: —</div>
            <div id="uiB" class="muted">Bodem: —</div>
            <div id="uiG" class="muted">Gt: —</div>
            <div id="uiH" class="muted">AHN (m): —</div>
            <div id="uiM" class="muted">Geomorfologie (GMM): —</div>
          </div>
        `;
        L.DomEvent.disableClickPropagation(div);
        return div;
      }
    });
    const infoCtl = new InfoCtl({ position: IS_MOBILE ? 'bottomright' : 'topright' }).addTo(map);

  function setClickInfo({fgr, bodem, bodem_bron, gt, vocht, ahn, gmm}) {
  const tF = "Fysisch Geografische Regio's: " + (fgr || '—');
  const tB = 'Bodem: ' + ((bodem || '—') + (bodem_bron ? ` (${bodem_bron})` : ''));
  const tG = 'Gt: ' + (gt || '—') + (vocht ? ` → ${vocht}` : ' (onbekend)');
  const tH = 'AHN (m): ' + ((ahn !== null && ahn !== undefined && ahn !== '') ? ahn : '—');
  const tM = 'Geomorfologie (GMM): ' + ((gmm !== null && gmm !== undefined && gmm !== '') ? gmm : '—');

  const set = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  };

  // legenda in de kaart (desktop)
  set('uiF', tF);
  set('uiB', tB);
  set('uiG', tG);
  set('uiH', tH);
  set('uiM', tM);

  // mobiele legenda onder de kaart
  set('uiF2', tF);
  set('uiB2', tB);
  set('uiG2', tG);
  set('uiH2', tH);
  set('uiM2', tM);
}

    async function loadWms(){
      ui.meta = await (await fetch('/api/wms_meta')).json();
      const make = (m, opacity)=> L.tileLayer.wms(m.url, { layers: m.layer, format:'image/png', transparent: true, opacity: opacity, version:'1.3.0', crs: L.CRS.EPSG3857 });
      overlays['BRO Bodemkaart (Bodemvlakken)'] = make(ui.meta.bodem, 0.55).addTo(map);
      overlays['BRO Grondwatertrappen (Gt)']    = make(ui.meta.gt,    0.45).addTo(map);
      overlays["Fysisch Geografische Regio's"]  = make(ui.meta.fgr,   0.45).addTo(map);
      overlays['AHN (hoogte, DTM 0.5m)']        = make(ui.meta.ahn,   0.50).addTo(map);
      overlays['BRO Geomorfologische kaart (GMM)'] = make(ui.meta.gmm,   0.45).addTo(map);

const ctlLayers = L.control.layers({}, overlays, { collapsed:true, position:'bottomleft' }).addTo(map);



      const cont = ctlLayers.getContainer();
      cont.classList.remove('leaflet-control-layers-expanded');
      const baseList = cont.querySelector('.leaflet-control-layers-base'); if(baseList) baseList.remove();
      const sep = cont.querySelector('.leaflet-control-layers-separator'); if(sep) sep.remove();
      const overlaysList = cont.querySelector('.leaflet-control-layers-overlays');
      const title = document.createElement('div');
      title.textContent = 'Kaartlagen';
      title.style.fontWeight = '700'; title.style.fontSize = '15px';
      title.style.margin = '6px 10px'; title.style.color = 'var(--fg)';
      overlaysList.parentNode.insertBefore(title, overlaysList);
    }

    async function fetchList(){
      const url = new URL(location.origin + '/api/plants');
      const inh = document.getElementById('inhOnly');
      const inv = document.getElementById('exInv');
      if(inh && inh.checked) url.searchParams.set('inheems_only','true');
      if(inv && inv.checked) url.searchParams.set('exclude_invasief','true');

      const chosenL = getChecked('licht');
      const chosenV = getChecked('vocht');
      const chosenB = getChecked('bodem');

      for (const v of chosenL) url.searchParams.append('licht', v);
      for (const v of chosenV) url.searchParams.append('vocht', v);
      for (const v of chosenB) url.searchParams.append('bodem', v);

      if (!chosenV.length && ui.ctx && ui.ctx.vocht) url.searchParams.append('vocht', ui.ctx.vocht);
      if (!chosenB.length && ui.ctx && ui.ctx.bodem) url.searchParams.append('bodem', ui.ctx.bodem);

      url.searchParams.set('limit','1000');

      window._lastQuery = new URLSearchParams(url.searchParams);

      const r = await fetch(url);
      return r.json();
    }

    function positionPopup(el, anchor){
      const r = anchor.getBoundingClientRect();
      el.style.left = (r.left) + 'px';
      el.style.top  = (r.bottom + 6) + 'px';
    }

    function openColsMenu(anchor){
      const box = document.getElementById('colMenu');
      box.innerHTML = '<h4 style="margin:0 0 6px;font-size:13px;">Kolommen</h4>' +
        COLS.map((c,i)=>`<label class="opt"><input type="checkbox" data-col="${c.key}" ${c.visible?'checked':''}> ${html(c.label)}</label>`).join('') +
        '<div class="actions" style="margin-top:10px;"><button id="colAll" class="btn">Alles</button><button id="colNone" class="btn">Niets</button></div>';
      positionPopup(box, anchor);
      box.classList.add('show');

      box.querySelectorAll('input[type=checkbox]').forEach(chk=>{
        chk.addEventListener('change', (e)=>{
          const key = chk.getAttribute('data-col');
          const idx = COLS.findIndex(c=>c.key===key);
          if(idx>=0){ COLS[idx].visible = !!chk.checked; saveCols(); buildTableHeader(); renderFromCache(); }
        });
      });
      box.querySelector('#colAll').onclick = ()=>{ COLS.forEach(c=>c.visible=true); saveCols(); buildTableHeader(); renderFromCache(); };
      box.querySelector('#colNone').onclick = ()=>{
        COLS.forEach(c=>c.visible=false);
        (COLS.find(c=>c.key==='naam')||{}).visible = true;
        saveCols(); buildTableHeader(); renderFromCache();
      };
    }
    function saveCols(){ localStorage.setItem(COLS_KEY, JSON.stringify(COLS)); }

    function getVisibleCols(){ return COLS.filter(c=>c.visible).map(c=>c.key); }

    function uniqueTokensFor(items, key){
      const set = new Set();
      for(const row of items||[]){
        if(key==='winterhardheidszone'){
          const v = String(row[key]??'').trim();
          if(v) set.add(v);
        }else if(key==='bodem'){
          for(const t of tokSplit(row['bodem'])) set.add(t);
          for(const t of tokSplit(row['grondsoorten'])) set.add(t);
        }else{
          for(const t of tokSplit(row[key])) set.add(t);
        }
      }
      return Array.from(set).sort((a,b)=> a.localeCompare(b,'nl',{numeric:true}));
    }

    function openHeaderFilterMenu(anchor, key, label){
      const menu = document.getElementById('colFilterMenu');
      const current = headerFilters.get(key) || new Set();
      const options = uniqueTokensFor(_lastItems, key);
      const optsHtml = options.map(val=>{
        const checked = current.has(val) ? 'checked' : '';
        return `<label class="opt"><input type="checkbox" data-key="${key}" value="${html(val)}" ${checked}> ${html(val)}</label>`;
      }).join('') || `<div class="muted">Geen waarden beschikbaar</div>`;
      menu.innerHTML = `<h4>${html(label)}</h4>${optsHtml}<div class="actions"><button class="btn" id="cfApply">Toepassen</button><button class="btn-ghost" id="cfClear">Leegmaken</button></div>`;
      positionPopup(menu, anchor);
      menu.classList.add('show');

      menu.querySelector('#cfApply').onclick = ()=>{
        const sel = new Set(Array.from(menu.querySelectorAll('input[type=checkbox]:checked')).map(i=>i.value));
        if(sel.size) headerFilters.set(key, sel); else headerFilters.delete(key);
        menu.classList.remove('show');
        renderFromCache();
      };
      menu.querySelector('#cfClear').onclick = ()=>{
        headerFilters.delete(key);
        menu.classList.remove('show');
        renderFromCache();
      };
    }

    document.addEventListener('click', (e)=>{
      const m1 = document.getElementById('colFilterMenu');
      const m2 = document.getElementById('colMenu');
      if(m1.classList.contains('show') && !m1.contains(e.target) && !e.target.closest('th.col-filter')){
        m1.classList.remove('show');
      }
      if(m2.classList.contains('show') && !m2.contains(e.target) && e.target.id!=='btnCols'){
        m2.classList.remove('show');
      }
    });

    function applyHeaderFilters(items){
      if(!items || !items.length) return items;
      const active = Array.from(headerFilters.entries());
      if(!active.length) return items;
      return items.filter(row=>{
        for(const [key, selSet] of active){
          if(!selSet || !selSet.size) continue;
          if(key==='winterhardheidszone'){
            const v = String(row[key]??'').trim();
            if(!selSet.has(v)) return false;
          }else if(key==='bodem'){
            const toks = new Set([...tokSplit(row['bodem']), ...tokSplit(row['grondsoorten'])].map(s=>s.toLowerCase()));
            const any = Array.from(selSet).some(s=> toks.has(String(s).toLowerCase()));
            if(!any) return false;
          }else{
            const toks = new Set(tokSplit(row[key]).map(s=>s.toLowerCase()));
            const any = Array.from(selSet).some(s=> toks.has(String(s).toLowerCase()));
            if(!any) return false;
          }
        }
        return true;
      });
    }

    function renderRows(items){
      const tb = document.querySelector('#tbl tbody');
      const vis = getVisibleCols();
      tb.innerHTML = items.map(r=>{
        const tds = vis.map(k=>{
          let v = r[k];
          if(k==='bodem' && !v) v = r['grondsoorten'] || '';
          return `<td>${html(v||'')}</td>`;
        }).join('');
        return `<tr>${tds}</tr>`;
      }).join('');
    }

    function updateCountDisplay(n){ document.getElementById('count').textContent = `${n} resultaten`; }

    function setFilterStatus({useLicht, useVocht, useBodem, sourceCtx=null}){
      const box = document.getElementById('filterStatus');
      const missing = [];
      if(!useLicht){
        missing.push("Er is geen lichtniveau geselecteerd; dit filter wordt niet toegepast.");
      }
      if(!useVocht){
        if(sourceCtx && !sourceCtx.vocht && (!sourceCtx.chosenVocht || sourceCtx.chosenVocht.length===0)){
          missing.push("Er is geen grondwatertrap gevonden op de geselecteerde locatie; er wordt niet op vocht gefilterd.");
        }else{
          missing.push("Er is geen vochtklasse geselecteerd; dit filter wordt niet toegepast.");
        }
      }
      if(!useBodem){
        if(sourceCtx && !sourceCtx.bodem && (!sourceCtx.chosenBodem || sourceCtx.chosenBodem.length===0)){
          missing.push("Er is geen bodemtype gevonden op de geselecteerde locatie; er wordt niet op bodem gefilterd.");
        }else{
          missing.push("Er is geen bodemtype geselecteerd; dit filter wordt niet toegepast.");
        }
      }

      if(missing.length===0){
        box.innerHTML = `<div class="flag ok"><span class="icon">✔</span><span class="text">Alle filters actief</span></div>`;
      }else{
        box.innerHTML = `<div class="flag warn"><span class="icon">⚠</span><span class="text">${missing.join("<br>")}</span></div>`;
      }
    }

    async function refresh(){
      const data = await fetchList();
      _lastItems = data.items||[];
      buildTableHeader();

      const filtered = applyHeaderFilters(_lastItems);
      updateCountDisplay(filtered.length);
      renderRows(filtered);

      const u = computeUsage();
      setFilterStatus({
        useLicht: u.useL,
        useVocht: u.useV,
        useBodem: u.useB,
        sourceCtx: {
          vocht: ui.ctx ? ui.ctx.vocht : null,
          bodem: ui.ctx ? ui.ctx.bodem : null,
          chosenVocht: u.chosenV,
          chosenBodem: u.chosenB
        }
      });
    }

    function renderFromCache(){
      const filtered = applyHeaderFilters(_lastItems);
      updateCountDisplay(filtered.length);
      renderRows(filtered);
    }

    // Klik op de kaart → context + lijst
    map.on('click', async (e)=>{
      if(window._marker) window._marker.remove();
      window._marker = L.marker(e.latlng).addTo(map);

      const urlCtx = new URL(location.origin + '/advies/geo');
      urlCtx.searchParams.set('lat', e.latlng.lat);
      urlCtx.searchParams.set('lon', e.latlng.lng);
      const inh = document.getElementById('inhOnly');
      const inv = document.getElementById('exInv');
      if(inh) urlCtx.searchParams.set('inheems_only', !!inh.checked);
      if(inv) urlCtx.searchParams.set('exclude_invasief', !!inv.checked);

      const j = await (await fetch(urlCtx)).json();

      setClickInfo({ fgr:j.fgr, bodem:j.bodem, bodem_bron:j.bodem_bron, gt:j.gt_code, vocht:j.vocht, ahn:j.ahn, gmm:j.gmm });

      // bewaar context (gebruikt door refresh / filters)
      ui.ctx = { vocht: j.vocht || null, bodem: j.bodem || null };

      refresh();
    });

    // Bouw de kolomkoppen; de titels zelf zijn de filter-triggers
    function buildTableHeader(){
      const tr = document.getElementById('theadRow');
      tr.innerHTML = '';
      for(const c of COLS.filter(c=>c.visible)){
        const th = document.createElement('th');
        if(c.filterable){ th.classList.add('col-filter'); th.dataset.key = c.key; th.dataset.label = c.label; }
        const wrap = document.createElement('div');
        wrap.className = 'th-wrap';
        const lbl = document.createElement('span');
        lbl.className = 'th-text';
        lbl.textContent = c.label;
        wrap.appendChild(lbl);
        th.appendChild(wrap);
        tr.appendChild(th);
      }
    }

    // Klik op kolomtitel → filtermenu
    document.getElementById('theadRow').addEventListener('click', (e)=>{
      const th = e.target.closest('th.col-filter');
      if(!th) return;
      const key = th.dataset.key;
      const label = th.dataset.label || th.textContent.trim();
      openHeaderFilterMenu(th, key, label);
    });

    (function(){
      const bar = document.getElementById('moreBar');
      const box = document.getElementById('moreFilters');
      const arrow = bar.querySelector('.arrow');
      box.classList.remove('open'); box.style.display='none'; arrow.textContent = '▾';
      bar.addEventListener('click', ()=>{
        const open = box.style.display !== 'none';
        if(open){ box.style.display='none'; box.classList.remove('open'); arrow.textContent='▾'; }
        else    { box.style.display='block'; box.classList.add('open'); arrow.textContent='▴'; }
      });
    })();

    function bindFilterEvents(){
      for(const sel of ['input[name="licht"]','input[name="vocht"]','input[name="bodem"]','#inhOnly','#exInv']){
        document.querySelectorAll(sel).forEach(el=> el.addEventListener('change', refresh));
      }
    }
    bindFilterEvents();

    document.getElementById('btnCSV')?.addEventListener('click', ()=>{
      const qp = window._lastQuery ? window._lastQuery.toString() : '';
      const href = '/export/csv' + (qp ? ('?'+qp) : '');
      window.open(href, '_blank');
    });
    document.getElementById('btnXLSX')?.addEventListener('click', ()=>{
      const qp = window._lastQuery ? window._lastQuery.toString() : '';
      const href = '/export/xlsx' + (qp ? ('?'+qp) : '');
      window.open(href, '_blank');
    });

    document.getElementById('btnCols')?.addEventListener('click', (e)=>{
      openColsMenu(e.currentTarget);
    });

    loadWms().then(refresh);
  </script>
</body>
</html>
'''
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store, max-age=0, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
