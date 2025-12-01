# PlantWijs API — v3.9.7 (met AHN + vocht-correctie)
# - PDOK Locatieserver → nieuwe endpoint (api.pdok.nl … /search/v3_1) met CORS
# - UI: Kolomtitel opent filter; kolommen tonen/verbergen; sticky header; thema toggle; CSV/XLSX export
# - AHN: hoogte uit AHN DTM 0,5m + eenvoudige relief-interpretatie en vocht-correctie
#
# Starten (lokaal):
#   cd C:/Rob/Beplantingswijzer/PlantWijs
#   .venv/Scripts/uvicorn api:app --reload --port 9000

from __future__ import annotations

import io
import math
import os
import re
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pyproj import Transformer

# ───────────────────── PDOK endpoints / WMS-config

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

# WMS AHN (DTM 0.5m)
AHN_WMS = "https://service.pdok.nl/rws/ahn/wms/v1_0"

# ───────────────────── Proj

TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ───────────────────── Dataset cache

DATA_PATHS = [
    "out/plantwijs_full_semicolon.csv",
    "out/plantwijs_full.csv",
]

# Online CSV fallback (GitHub raw)
ONLINE_CSV_URLS = [
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full.csv",
    "https://raw.githubusercontent.com/Rroobbbb/plantwijs/main/out/plantwijs_full_semicolon.csv",
]

_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None, "source": None}

# ───────────────────── CSV helpers

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

    # 3) Niets gevonden
    raise FileNotFoundError(
        "Geen dataset gevonden. Lokaal ontbreekt out/plantwijs_full.csv én online CSV kon niet worden opgehaald."
    )

# ───────────────────── HTTP / WMS utils

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
    cand: List[Tuple[str, str]] = []
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


_WMSMETA: Dict[str, Dict[str, str]] = {}


def _resolve_layers() -> None:
    global _WMSMETA
    meta: Dict[str, Dict[str, str]] = {}
    fgr = _find_layer_name(FGR_WMS, ["fysisch", "fgr"]) or ("fysischgeografischeregios", "FGR")
    bodem = _find_layer_name(BODEM_WMS, ["bodemvlakken", "bodem"]) or ("soilarea", "Bodemvlakken")
    gt = _find_layer_name(GWD_WMS, ["grondwatertrappen", "gt"]) or (
        "bro-grondwaterspiegeldieptemetingen-GT",
        "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt",
    )
    ghg = _find_layer_name(GWD_WMS, ["ghg"]) or ("bro-grondwaterspiegeldiepte-metingen-GHG", "GHG")
    glg = _find_layer_name(GWD_WMS, ["glg"]) or ("bro-grondwaterspiegeldiepte-metingen-GLG", "GLG")
    ahn = _find_layer_name(AHN_WMS, ["dtm"]) or ("dtm_05m", "Digital Terrain Model (DTM) 0,5m")

    meta["fgr"] = {"url": FGR_WMS, "layer": fgr[0], "title": fgr[1]}
    meta["bodem"] = {"url": BODEM_WMS, "layer": bodem[0], "title": bodem[1]}
    meta["gt"] = {"url": GWD_WMS, "layer": gt[0], "title": gt[1]}
    meta["ghg"] = {"url": GWD_WMS, "layer": ghg[0], "title": ghg[1]}
    meta["glg"] = {"url": GWD_WMS, "layer": glg[0], "title": glg[1]}
    meta["ahn"] = {"url": AHN_WMS, "layer": ahn[0], "title": ahn[1]}
    _WMSMETA = meta
    print("[WMS] resolved:", _WMSMETA)


def _bbox_around(lat: float, lon: float, size_m: float = 25.0) -> Tuple[float, float, float, float]:
    x, y = TX_WGS84_WEB.transform(lon, lat)
    half = size_m
    return (x - half, y - half, x + half, y + half)


def _wms_getfeatureinfo(
    base_url: str, layer: str, lat: float, lon: float, epsg: int = 4326
) -> Dict[str, Any]:
    if not layer:
        return {}
    bbox = _bbox_around(lat, lon, 25.0)
    params = {
        "service": "WMS",
        "request": "GetFeatureInfo",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "crs": "EPSG:3857",
        "bbox": ",".join(str(v) for v in bbox),
        "width": 101,
        "height": 101,
        "format": "image/png",
        "query_layers": layer,
        "info_format": "application/vnd.ogc.gml",
        "i": 50,
        "j": 50,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        r = _get(url)
        if r.status_code != 200:
            return {}
        xml = r.text
        root = ET.fromstring(xml)
        fields = root.findall(".//{*}FIELDS")
        if not fields:
            return {}
        props: Dict[str, Any] = {}
        for f in fields:
            props.update(f.attrib)
        return props
    except Exception as e:
        print("[WMS FI] fout voor", layer, "→", e)
        return {}

# ───────────────────── FGR / bodem / vocht

def fgr_from_point(lat: float, lon: float) -> Optional[str]:
    x, y = TX_WGS84_RD.transform(lon, lat)
    bbox = (x - 1, y - 1, x + 1, y + 1)
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": "fysischgeografischeregios",
        "srsName": "EPSG:28992",
        "bbox": ",".join(str(v) for v in bbox) + ",EPSG:28992",
        "outputFormat": FMT_JSON,
    }
    url = f"{PDOK_FGR_WFS}&{urllib.parse.urlencode(params)}"
    try:
        r = _get(url)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features") or []
        if not feats:
            return None
        props = feats[0].get("properties") or {}
        for key in ("NAAM", "naam", "fgr_naam", "fysisch_geografische_regio"):
            if key in props:
                return str(props[key])
        return None
    except Exception as e:
        print("[FGR] fout:", e)
        return None


def _vochtklasse_from_gt_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip().upper()
    mapping = {
        "IA": "zeer droog",
        "IB": "zeer droog",
        "IIA": "droog",
        "IIB": "droog",
        "IIIA": "droog / vochtig",
        "IIIB": "droog / vochtig",
        "IIIC": "droog / vochtig",
        "IVA": "vochtig",
        "IVB": "vochtig",
        "IVC": "vochtig",
        "IVD": "vochtig / nat",
        "VA": "nat",
        "VB": "nat",
        "VC": "nat / zeer nat",
        "VIA": "zeer nat",
        "VIB": "zeer nat",
        "VIIA": "zeer nat",
        "VIIB": "zeer nat",
        "VIIIA": "zeer nat",
        "VIIIB": "zeer nat",
        "VIIIC": "zeer nat",
        "VIIID": "zeer nat",
    }
    # sommige services geven 'IVu' etc → uppercase zonder 'U'
    c = c.replace("U", "")
    return mapping.get(c, None)


def bodem_from_bodemkaart(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    meta = _WMSMETA.get("bodem", {})
    layer = meta.get("layer") or "soilarea"
    props = _wms_getfeatureinfo(BODEM_WMS, layer, lat, lon)
    if not props:
        return None, None
    # Codes en omschrijvingen in attribuutnamen kunnen verschillen
    naam = props.get("NAAM") or props.get("naam") or props.get("LEGEND") or props.get("omschrijving")
    return (str(naam) if naam else None, "BRO Bodemkaart WMS")


def vocht_from_gwt(lat: float, lon: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    meta = _WMSMETA.get("gt", {})
    layer = meta.get("layer") or "bro-grondwaterspiegeldiepte-metingen-GT"
    props = _wms_getfeatureinfo(GWD_WMS, layer, lat, lon)
    if not props:
        return None, None, None
    # PDOK Gt heeft meestal value_list = "4" en/of code = "IVu"
    gt_code = props.get("code") or props.get("GT_CODE") or props.get("gt_code")
    if not gt_code and "value_list" in props:
        # oude variant met 1..19 → simpele mapping naar code
        try:
            v = int(str(props["value_list"]).split(",")[0].strip())
        except Exception:
            v = None
        map_idx = {
            1: "Ia",
            2: "Ib",
            3: "IIa",
            4: "IIb",
            5: "IIIa",
            6: "IIIb",
            7: "IIIc",
            8: "IVa",
            9: "IVb",
            10: "IVc",
            11: "IVd",
            12: "Va",
            13: "Vb",
            14: "Vc",
            15: "VIa",
            16: "VIb",
            17: "VIIa",
            18: "VIIb",
            19: "VIII",
        }
        if v in map_idx:
            gt_code = map_idx[v]
    vocht = _vochtklasse_from_gt_code(gt_code)
    return vocht, (gt_code if gt_code else None), "BRO Gt/GLG WMS"

# ───────────────────── AHN helpers

def _parse_ahn_value(props: Dict[str, Any]) -> Optional[float]:
    """
    PDOK AHN-WMS geeft meestal een attribuut 'value_list' met één float (meters t.o.v. NAP).
    """
    v = props.get("value_list") or props.get("VALUE_LIST") or props.get("value")
    if v is None:
        return None
    try:
        txt = str(v).split(",")[0].strip()
        if not txt:
            return None
        val = float(txt)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None


def ahn_from_point(lat: float, lon: float) -> Optional[float]:
    meta = _WMSMETA.get("ahn", {})
    layer = meta.get("layer") or "dtm_05m"
    props = _wms_getfeatureinfo(AHN_WMS, layer, lat, lon)
    if not props:
        return None
    return _parse_ahn_value(props)


def classify_relief(hoogte_m: Optional[float]) -> Optional[str]:
    """
    Eenvoudige relief-indicatie op basis van absolute hoogte.
    Dit is natuurlijk grof, maar geeft voor NL al een gevoel:
    - onder -1 m → 'laagte'
    - tussen -1 en +1 → 'vlak'
    - boven +1 → 'hoogte'
    """
    if hoogte_m is None:
        return None
    if hoogte_m < -1.0:
        return "laagte"
    if hoogte_m > 1.0:
        return "hoogte"
    return "vlak"


def vocht_met_ahn_correctie(vocht_basis: Optional[str], relief: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Simpele 'slimme' correctie:
    - op een 'hoogte' wordt het één stap droger
    - in een 'laagte' wordt het één stap natter
    """
    if not vocht_basis or not relief or relief == "vlak":
        return vocht_basis, False

    volgorde = ["zeer droog", "droog", "droog / vochtig", "vochtig", "vochtig / nat", "nat", "nat / zeer nat", "zeer nat"]

    def _norm(v: str) -> str:
        return v.strip().lower()

    vb = _norm(vocht_basis)
    # Zoek dichtstbijzijnde categorie
    try:
        idx = min(range(len(volgorde)), key=lambda i: abs(i - volgorde.index(vb)))  # type: ignore[arg-type]
    except Exception:
        # fallback: exact match of normalised list
        try:
            idx = [_norm(x) for x in volgorde].index(vb)
        except Exception:
            return vocht_basis, False

    if relief == "hoogte" and idx > 0:
        idx2 = idx - 1
    elif relief == "laagte" and idx < len(volgorde) - 1:
        idx2 = idx + 1
    else:
        return vocht_basis, False

    return volgorde[idx2], True

# ───────────────────── planten-filter logica

_RE_ALL = re.compile(r"\balles?\b|\balle grondsoorten\b", re.IGNORECASE)
_SOIL_CANON = ["zand", "klei", "leem", "veen"]


def _contains_ci(val: Any, q: str) -> bool:
    if not q:
        return True
    v = str(val or "").lower()
    return q.lower() in v


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

app = FastAPI(title="PlantWijs API v3.9.7 (met AHN)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    try:
        if pd.isna(o):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return o

# ───────────────────── API: diagnose/meta

@app.get("/api/wms_meta")
def api_wms_meta():
    return JSONResponse(_clean(_WMSMETA))


@app.get("/api/diag/featureinfo")
def api_diag(
    service: str = Query(..., pattern="^(bodem|gt|ghg|glg|fgr|ahn)$"),
    lat: float = Query(...),
    lon: float = Query(...),
):
    if service == "fgr":
        return JSONResponse({"fgr": fgr_from_point(lat, lon)})
    base = {
        "bodem": BODEM_WMS,
        "gt": GWD_WMS,
        "ghg": GWD_WMS,
        "glg": GWD_WMS,
        "ahn": AHN_WMS,
    }[service]
    layer = _WMSMETA.get(service, {}).get("layer")
    props = _wms_getfeatureinfo(base, layer or "", lat, lon)
    return JSONResponse(_clean({"base": base, "layer": layer, "props": props}))

# ───────────────────── API: plantenlijst + export

@app.get("/api/plants")
def api_plants(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    limit: int = Query(200, ge=1, le=1000),
    sort: str = Query("naam"),
    desc: bool = Query(False),
):
    df = get_df()
    if q:
        df = df[
            df.apply(
                lambda r: _contains_ci(r.get("naam"), q)
                or _contains_ci(r.get("wetenschappelijke_naam"), q),
                axis=1,
            )
        ]
    if inheems_only:
        df = df[df["inheems"].str.lower().eq("ja")]
    if exclude_invasief:
        df = df[~df["invasief"].str.lower().eq("ja")]

    if licht:
        low = [l.strip().lower() for l in licht if l.strip()]
        df = df[
            df["standplaats_licht"]
            .astype(str)
            .str.lower()
            .apply(lambda v: any(l in v for l in low))
        ]

    if vocht:
        low = [v.strip().lower() for v in vocht if v.strip()]
        df = df[
            df["vocht"].astype(str).str.lower().apply(lambda v: any(x in v for x in low))
        ]

    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]

    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)

    total = len(df)
    df = df.head(limit)
    items = df.to_dict(orient="records")
    return JSONResponse(_clean({"count": total, "items": items}))


@app.get("/api/plants/export")
def api_plants_export(
    fmt: str = Query("csv", pattern="^(csv|xlsx)$"),
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
):
    # zelfde filter als /api/plants, maar zonder limit
    df = get_df()
    if q:
        df = df[
            df.apply(
                lambda r: _contains_ci(r.get("naam"), q)
                or _contains_ci(r.get("wetenschappelijke_naam"), q),
                axis=1,
            )
        ]
    if inheems_only:
        df = df[df["inheems"].str.lower().eq("ja")]
    if exclude_invasief:
        df = df[~df["invasief"].str.lower().eq("ja")]
    if licht:
        low = [l.strip().lower() for l in licht if l.strip()]
        df = df[
            df["standplaats_licht"]
            .astype(str)
            .str.lower()
            .apply(lambda v: any(l in v for l in low))
        ]
    if vocht:
        low = [v.strip().lower() for v in vocht if v.strip()]
        df = df[
            df["vocht"].astype(str).str.lower().apply(lambda v: any(x in v for x in low))
        ]
    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]

    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep=";")
        data = buf.getvalue().encode("utf-8")
        return StreamingResponse(
            io.BytesIO(data),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="plantwijs_export.csv"'},
        )
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            data = tmp.read()
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="plantwijs_export.xlsx"'},
        )

# ───────────────────── API: advies /geo (met AHN)

@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    limit: int = Query(150, ge=1, le=500),
):
    fgr = fgr_from_point(lat, lon)
    bodem_naam, bodem_bron = bodem_from_bodemkaart(lat, lon)
    vocht_basis, gt_code, vocht_bron = vocht_from_gwt(lat, lon)

    # AHN
    ahn_h = ahn_from_point(lat, lon)
    ahn_relief = classify_relief(ahn_h)
    vocht_corr, corr_used = vocht_met_ahn_correctie(vocht_basis, ahn_relief)

    moist_for_filter = vocht_corr or vocht_basis

    df = get_df()
    if inheems_only:
        df = df[df["inheems"].str.lower().eq("ja")]
    if exclude_invasief:
        df = df[~df["invasief"].str.lower().eq("ja")]
    if licht:
        low = [l.strip().lower() for l in licht if l.strip()]
        df = df[
            df["standplaats_licht"]
            .astype(str)
            .str.lower()
            .apply(lambda v: any(l in v for l in low))
        ]

    if moist_for_filter:
        m = moist_for_filter.lower()
        df = df[
            df["vocht"]
            .astype(str)
            .str.lower()
            .apply(lambda v: m in v or "alle grondsoorten" in v)
        ]

    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]

    df = df.head(limit)
    advies = df.to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_naam,
        "bodem_bron": bodem_bron,
        "gt_code": gt_code,
        "vocht_basis": vocht_basis,
        "vocht": moist_for_filter,
        "vocht_bron": vocht_bron,
        "vocht_ahn_correctie": bool(corr_used),
        "ahn_hoogte_m": ahn_h,
        "ahn_relief": ahn_relief,
        "ahn_delta_m": None,  # hier kun je later lokale delta op basis van buurpixels toevoegen
        "advies": advies,
    }
    return JSONResponse(_clean(out))

# ───────────────────── API: PDOK Locatieserver zoek

@app.get("/api/search")
def api_search(q: str = Query(..., min_length=3)):
    """
    Simpele proxy naar PDOK Locatieserver v3.1 met CORS, zodat je in de frontend gewoon /api/search kunt aanroepen.
    """
    base = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
    params = {"q": q, "rows": 10, "fl": "weergavenaam,centroide_ll"}
    url = f"{base}?{urllib.parse.urlencode(params)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        r.raise_for_status()
        js = r.json()
        docs = js.get("response", {}).get("docs", [])
        out = []
        for d in docs:
            naam = d.get("weergavenaam")
            ll = d.get("centroide_ll")
            latlon = None
            if ll and ll.startswith("POINT("):
                try:
                    # "POINT(4.123 52.123)"
                    _, rest = ll.split("POINT(")
                    rest = rest.strip(" )")
                    lon_s, lat_s = rest.split()
                    latlon = (float(lat_s), float(lon_s))
                except Exception:
                    latlon = None
            out.append({"label": naam, "latlon": latlon})
        return JSONResponse(_clean({"items": out}))
    except Exception as e:
        print("[SEARCH] fout:", e)
        return JSONResponse({"items": []})

# ───────────────────── HTML frontend

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # kleine inline HTML zodat Render gewoon één bestand nodig heeft
    html = r"""<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <title>PlantWijs v3.9.7 (met AHN)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css"
    integrity="sha256-sA+e2atLYB9wH0uCjISjghDUFxu05mZ8gX+S/5u0mXU="
    crossorigin=""
  />
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f172a;
      --fg: #e5e7eb;
      --fg-muted: #9ca3af;
      --accent: #22c55e;
      --accent-soft: rgba(34,197,94,0.12);
      --border: #1f2937;
      --chip-bg: #111827;
    }
    body {
      margin:0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--fg);
      display:flex;
      flex-direction:column;
      height:100vh;
    }
    header {
      padding:8px 16px;
      border-bottom:1px solid var(--border);
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      background:#020617;
      position:sticky;
      top:0;
      z-index:50;
    }
    header h1 {
      font-size:16px;
      margin:0;
      font-weight:600;
    }
    header small {
      color:var(--fg-muted);
      font-size:11px;
    }
    .layout {
      flex:1;
      display:flex;
      min-height:0;
    }
    #map {
      flex:0 0 40%;
      min-width:260px;
      border-right:1px solid var(--border);
    }
    .right {
      flex:1;
      display:flex;
      flex-direction:column;
      min-width:0;
    }
    .filters {
      padding:8px 12px;
      border-bottom:1px solid var(--border);
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      align-items:center;
    }
    .filters label {
      font-size:11px;
      color:var(--fg-muted);
      display:flex;
      gap:4px;
      align-items:center;
    }
    .filters input[type="search"] {
      background:#020617;
      border:1px solid var(--border);
      border-radius:999px;
      padding:4px 10px;
      color:var(--fg);
      font-size:12px;
      min-width:180px;
    }
    .filters .chip {
      border-radius:999px;
      border:1px solid var(--border);
      background:var(--chip-bg);
      padding:2px 8px;
      font-size:11px;
      display:inline-flex;
      align-items:center;
      gap:4px;
      cursor:pointer;
    }
    .filters .chip span.key {
      color:var(--fg-muted);
    }
    .filters .chip span.val {
      color:var(--accent);
    }
    .filters button {
      background:var(--accent-soft);
      border:1px solid var(--accent);
      color:var(--accent);
      border-radius:999px;
      padding:3px 10px;
      font-size:11px;
      cursor:pointer;
    }
    .legend {
      padding:6px 12px;
      border-bottom:1px solid var(--border);
      font-size:11px;
      display:flex;
      flex-wrap:wrap;
      gap:6px 16px;
      align-items:center;
    }
    .legend span.label {
      color:var(--fg-muted);
    }
    .legend span.value {
      color:var(--fg);
      font-weight:500;
    }
    .legend span.badge {
      border-radius:999px;
      padding:1px 6px;
      border:1px solid var(--border);
      background:#020617;
      color:var(--fg-muted);
      font-size:10px;
    }
    .results {
      flex:1;
      min-height:0;
      overflow:auto;
    }
    table {
      border-collapse:collapse;
      width:100%;
      font-size:12px;
    }
    thead th {
      position:sticky;
      top:0;
      background:#020617;
      z-index:10;
      border-bottom:1px solid var(--border);
      padding:6px 8px;
      text-align:left;
      font-weight:500;
      cursor:pointer;
      white-space:nowrap;
    }
    thead th span.colname {
      border-bottom:1px dashed var(--fg-muted);
    }
    tbody td {
      padding:6px 8px;
      border-bottom:1px solid #111827;
      vertical-align:top;
    }
    tbody tr:nth-child(2n) td {
      background:#020617;
    }
    .countline {
      padding:4px 12px;
      font-size:11px;
      color:var(--fg-muted);
      border-bottom:1px solid var(--border);
    }
    .popup {
      position:absolute;
      background:#020617;
      border:1px solid var(--border);
      border-radius:8px;
      padding:8px;
      font-size:11px;
      box-shadow:0 10px 30px rgba(0,0,0,0.5);
      z-index:1000;
      display:none;
      min-width:160px;
    }
    .popup h4 {
      margin:0 0 6px;
      font-weight:500;
      font-size:11px;
    }
    .popup label {
      display:block;
      margin-bottom:3px;
    }
    .popup-footer {
      margin-top:6px;
      display:flex;
      justify-content:flex-end;
      gap:6px;
    }
    .popup-footer button {
      border-radius:999px;
      border:1px solid var(--border);
      background:#020617;
      color:var(--fg-muted);
      font-size:10px;
      padding:2px 8px;
      cursor:pointer;
    }
    .popup-footer button.primary {
      border-color:var(--accent);
      color:var(--accent);
    }
    @media (max-width:900px){
      .layout { flex-direction:column; }
      #map { flex:0 0 280px; border-right:none; border-bottom:1px solid var(--border); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>PlantWijs</h1>
      <small>Inheemse soorten per plek – met bodem, Gt én AHN</small>
    </div>
    <div style="display:flex;gap:8px;align-items:center;font-size:11px;">
      <button id="btnLocate" style="border-radius:999px;border:1px solid var(--border);background:#020617;color:var(--fg-muted);padding:3px 8px;cursor:pointer;">Gebruik mijn locatie</button>
    </div>
  </header>
  <div class="layout">
    <div id="map"></div>
    <div class="right">
      <div class="filters">
        <input id="q" type="search" placeholder="Zoek op naam / wetenschappelijke naam…">
        <label><input type="checkbox" id="inhOnly" checked> Inheems</label>
        <label><input type="checkbox" id="exInv" checked> Invasieve soorten verbergen</label>
        <span class="chip" id="chipL"><span class="key">Licht:</span> <span class="val" id="chipLval">alle</span></span>
        <span class="chip" id="chipV"><span class="key">Vocht:</span> <span class="val" id="chipVval">auto</span></span>
        <span class="chip" id="chipB"><span class="key">Bodem:</span> <span class="val" id="chipBval">auto</span></span>
        <button id="btnExportCsv">Export CSV</button>
        <button id="btnExportXlsx">Export XLSX</button>
      </div>
      <div class="legend">
        <span><span class="label">FGR:</span> <span class="value" id="uiF">–</span></span>
        <span><span class="label">Bodem:</span> <span class="value" id="uiB">–</span></span>
        <span><span class="label">Gt / vocht:</span> <span class="value" id="uiG">–</span></span>
        <span><span class="label">AHN:</span> <span class="value" id="uiA">–</span></span>
        <span class="badge" id="uiCorr" style="display:none;">AHN-correctie actief</span>
      </div>
      <div class="countline" id="count">0 resultaten</div>
      <div class="results">
        <table>
          <thead>
            <tr>
              <th data-col="naam"><span class="colname">Naam</span></th>
              <th data-col="wetenschappelijke_naam"><span class="colname">Wetenschappelijke naam</span></th>
              <th data-col="standplaats_licht"><span class="colname">Licht</span></th>
              <th data-col="vocht"><span class="colname">Vocht</span></th>
              <th data-col="bodem"><span class="colname">Bodem / grondsoorten</span></th>
              <th data-col="winterhardheidszone"><span class="colname">Zone</span></th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="popup" id="popup">
    <h4 id="popupTitle"></h4>
    <div id="popupBody"></div>
    <div class="popup-footer">
      <button id="btnPopupReset">Reset</button>
      <button id="btnPopupApply" class="primary">Toepassen</button>
    </div>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"
    integrity="sha256-o9N1j7kGEXG1+bLLpCsoPJjt1Iu0p3pM9v+1p5wQ+7Q="
    crossorigin=""
  ></script>
  <script>
    const map = L.map('map').setView([52.1,5.3], 8);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
      maxZoom:19,
      attribution:'&copy; OpenStreetMap &copy; CartoDB'
    }).addTo(map);

    let WMS_META = null;

    async function loadWms(){
      const r = await fetch('/api/wms_meta');
      WMS_META = await r.json();
      if(!WMS_META || !WMS_META.bodem) return;

      const bodem = WMS_META.bodem;
      const gt = WMS_META.gt;
      const fgr = WMS_META.fgr;
      const ahn = WMS_META.ahn;

      L.tileLayer.wms(bodem.url, {
        layers: bodem.layer,
        format: 'image/png',
        transparent: true,
        opacity: 0.5
      }).addTo(map);

      L.tileLayer.wms(gt.url, {
        layers: gt.layer,
        format: 'image/png',
        transparent: true,
        opacity: 0.5
      }).addTo(map);

      L.tileLayer.wms(fgr.url, {
        layers: fgr.layer,
        format: 'image/png',
        transparent: true,
        opacity: 0.4
      }).addTo(map);

      L.tileLayer.wms(ahn.url, {
        layers: ahn.layer,
        format: 'image/png',
        transparent: true,
        opacity: 0.3
      }).addTo(map);
    }

    function setClickInfo(obj){
      const { fgr, bodem, bodem_bron, gt_code, vocht_basis, vocht, vocht_bron, ahn_hoogte_m, ahn_relief, vocht_ahn_correctie } = obj || {};
      document.getElementById('uiF').textContent = fgr || '–';

      let btxt = bodem || '–';
      if(bodem_bron) btxt += ' ('+bodem_bron+')';
      document.getElementById('uiB').textContent = btxt;

      let gtxt = '';
      if(gt_code) gtxt += 'Gt '+gt_code;
      if(vocht_basis) {
        if(gtxt) gtxt += ' · ';
        gtxt += 'vocht: '+vocht_basis;
      }
      if(vocht && vocht !== vocht_basis) {
        gtxt += ' → '+vocht+' (AHN)';
      }
      if(!gtxt) gtxt = '–';
      document.getElementById('uiG').textContent = gtxt;

      let atxt = '–';
      if(ahn_hoogte_m != null){
        const h = Number(ahn_hoogte_m).toFixed(2);
        atxt = h+' m NAP';
        if(ahn_relief){
          atxt += ' ('+ahn_relief+')';
        }
      }
      document.getElementById('uiA').textContent = atxt;

      const badge = document.getElementById('uiCorr');
      if(vocht_ahn_correctie){
        badge.style.display = 'inline-flex';
      } else {
        badge.style.display = 'none';
      }
    }

    let filters = {
      licht: [],
      vocht: [],
      bodem: []
    };

    const POP = document.getElementById('popup');
    let popupTarget = null;

    function openPopup(target, type){
      popupTarget = type;
      const rect = target.getBoundingClientRect();
      POP.style.left = (rect.left + window.scrollX) + 'px';
      POP.style.top = (rect.bottom + window.scrollY + 4) + 'px';
      POP.style.display = 'block';

      document.getElementById('popupTitle').textContent =
        type === 'licht' ? 'Filter op licht' :
        type === 'vocht' ? 'Filter op vocht' :
        'Filter op bodem';

      const body = document.getElementById('popupBody');
      body.innerHTML = '';
      const options = (type === 'licht'
        ? ['schaduw','halfschaduw','zon']
        : type === 'vocht'
        ? ['zeer droog','droog','droog / vochtig','vochtig','vochtig / nat','nat','nat / zeer nat','zeer nat']
        : ['zand','klei','leem','veen']
      );
      const current = new Set(filters[type] || []);
      for(const opt of options){
        const id = 'pop_' + type + '_' + opt.replace(/\s+/g,'_');
        const label = document.createElement('label');
        label.innerHTML = `<input type="checkbox" id="${id}" ${current.has(opt)?'checked':''}> ${opt}`;
        body.appendChild(label);
      }
    }

    function closePopup(){
      POP.style.display = 'none';
      popupTarget = null;
    }

    document.getElementById('btnPopupReset').addEventListener('click', ()=>{
      if(!popupTarget) return;
      filters[popupTarget] = [];
      closePopup();
      renderChips();
      refresh();
    });

    document.getElementById('btnPopupApply').addEventListener('click', ()=>{
      if(!popupTarget) return;
      const body = document.getElementById('popupBody');
      const checked = [];
      for(const inp of body.querySelectorAll('input[type="checkbox"]')){
        if(inp.checked){
          const label = inp.parentElement.textContent.trim();
          checked.push(label);
        }
      }
      filters[popupTarget] = checked;
      closePopup();
      renderChips();
      refresh();
    });

    document.addEventListener('click', (e)=>{
      if(!POP.contains(e.target) && !e.target.closest('.colname') && !e.target.closest('.chip')){
        closePopup();
      }
    });

    function renderChips(){
      document.getElementById('chipLval').textContent = filters.licht.length ? filters.licht.join(', ') : 'alle';
      document.getElementById('chipVval').textContent = filters.vocht.length ? filters.vocht.join(', ') : 'auto';
      document.getElementById('chipBval').textContent = filters.bodem.length ? filters.bodem.join(', ') : 'auto';
    }

    function html(s){ return (s||'').replace(/[&<>"]/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c])); }

    async function fetchList(){
      const url = new URL(location.origin + '/api/plants');
      const q = document.getElementById('q').value.trim();
      if(q) url.searchParams.set('q', q);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      for(const l of filters.licht) url.searchParams.append('licht', l);
      for(const v of filters.vocht) url.searchParams.append('vocht', v);
      for(const b of filters.bodem) url.searchParams.append('bodem', b);
      const r = await fetch(url);
      return await r.json();
    }

    function renderRows(items){
      const tb = document.getElementById('rows');
      tb.innerHTML = (items||[]).map(r=>`
        <tr>
          <td>${html(r.naam||'')}</td>
          <td><i>${html(r.wetenschappelijke_naam||'')}</i></td>
          <td>${html(r.standplaats_licht||'')}</td>
          <td>${html(r.vocht||'')}</td>
          <td>${html(r.bodem||r.grondsoorten||'')}</td>
          <td>${html(r.winterhardheidszone||'')}</td>
        </tr>
      `).join('');
    }

    async function refresh(){
      const data = await fetchList();
      document.getElementById('count').textContent =
        (data.count || 0) + ' resultaten (eerste ' + (data.items ? data.items.length : 0) + ')';
      renderRows(data.items || []);
    }

    // kaart-click → advies + legenda
    map.on('click', async (e)=>{
      if(window._marker) window._marker.remove();
      window._marker = L.marker(e.latlng).addTo(map);

      const url = new URL(location.origin + '/advies/geo');
      url.searchParams.set('lat', e.latlng.lat);
      url.searchParams.set('lon', e.latlng.lng);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);

      // bodem/vocht-filters in advies niet meegeven; die doen we al via AHN + Gt
      const j = await (await fetch(url)).json();

      setClickInfo(j);

      const data = { items: j.advies || [] };
      document.getElementById('count').textContent =
        (data.items.length || 0) + ' resultaten (auto-filter)';
      renderRows(data.items || []);
    });

    document.getElementById('btnLocate').addEventListener('click', ()=>{
      if(!navigator.geolocation){ alert('Geolocatie niet ondersteund.'); return; }
      navigator.geolocation.getCurrentPosition(pos=>{
        const lat = pos.coords.latitude, lon = pos.coords.longitude;
        map.setView([lat, lon], 14);
        if(window._marker) window._marker.remove();
        window._marker = L.marker([lat,lon]).addTo(map);
        map.fire('click', { latlng:{ lat, lng:lon } });
      }, err=>{ alert('Kon locatie niet ophalen'); });
    });

    // kolomtitel-kliks → zelfde popup als chips
    document.querySelectorAll('thead th').forEach(th=>{
      th.addEventListener('click', ()=>{
        const col = th.getAttribute('data-col');
        if(col === 'standplaats_licht') openPopup(th.querySelector('.colname'), 'licht');
        else if(col === 'vocht') openPopup(th.querySelector('.colname'), 'vocht');
        else if(col === 'bodem') openPopup(th.querySelector('.colname'), 'bodem');
      });
    });

    document.getElementById('chipL').addEventListener('click', e=>openPopup(e.currentTarget, 'licht'));
    document.getElementById('chipV').addEventListener('click', e=>openPopup(e.currentTarget, 'vocht'));
    document.getElementById('chipB').addEventListener('click', e=>openPopup(e.currentTarget, 'bodem'));

    document.getElementById('q').addEventListener('input', ()=>{
      clearTimeout(window._qTimer);
      window._qTimer = setTimeout(refresh, 300);
    });
    document.getElementById('inhOnly').addEventListener('change', refresh);
    document.getElementById('exInv').addEventListener('change', refresh);

    document.getElementById('btnExportCsv').addEventListener('click', ()=>{
      const url = new URL(location.origin + '/api/plants/export');
      url.searchParams.set('fmt','csv');
      const q = document.getElementById('q').value.trim();
      if(q) url.searchParams.set('q', q);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      for(const l of filters.licht) url.searchParams.append('licht', l);
      for(const v of filters.vocht) url.searchParams.append('vocht', v);
      for(const b of filters.bodem) url.searchParams.append('bodem', b);
      location.href = url.toString();
    });

    document.getElementById('btnExportXlsx').addEventListener('click', ()=>{
      const url = new URL(location.origin + '/api/plants/export');
      url.searchParams.set('fmt','xlsx');
      const q = document.getElementById('q').value.trim();
      if(q) url.searchParams.set('q', q);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      for(const l of filters.licht) url.searchParams.append('licht', l);
      for(const v of filters.vocht) url.searchParams.append('vocht', v);
      for(const b of filters.bodem) url.searchParams.append('bodem', b);
      location.href = url.toString();
    });

    loadWms().then(refresh);
  </script>
</body>
</html>
"""
    return html

# ───────────────────── startup

@app.on_event("startup")
def _on_startup() -> None:
    _resolve_layers()
