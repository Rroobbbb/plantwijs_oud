# PlantWijs API — v3.9.2 met AHN & slimme vochtcorrectie
from __future__ import annotations

import io
import math
import os
import time
import csv
import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pyproj import Transformer

# ───────────────────── PDOK endpoints
HEADERS = {"User-Agent": "plantwijs/3.9.2"}
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

# WMS AHN (Actueel Hoogtebestand Nederland)
AHN_WMS = "https://service.pdok.nl/rws/ahn/wms/v1_0"

# ───────────────────── Proj
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ───────────────────── Dataset cache
DATA_PATHS = [
    "out/plantwijs_full_semicolon.csv",
    "out/plantwijs_full.csv",
]
_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None}


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
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
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


def get_df() -> pd.DataFrame:
    for path in DATA_PATHS:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if _CACHE["df"] is not None and _CACHE["path"] == path and _CACHE["mtime"] == mtime:
                return _CACHE["df"]
            df = _load_df(path)
            _CACHE.update({"df": df, "mtime": mtime, "path": path})
            return df
    raise FileNotFoundError("Geen plantwijs CSV gevonden in 'out/'")


# ───────────────────── HTTP / WMS / WFS helpers
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
        name = name_el.text if name_el is not None else ""
        title = title_el.text if title_el is not None else ""
        if not name and not title:
            continue
        cand.append((name, title))
    lwant = [w.lower() for w in want]
    # 1) op Title
    for name, title in cand:
        lt = title.lower()
        if any(w in lt for w in lwant):
            return name, title
    # 2) op Name
    for name, title in cand:
        ln = name.lower()
        if any(w in ln for w in lwant):
            return name, title
    return cand[0] if cand else None


_WMSMETA: Dict[str, Dict[str, str]] = {}


def _resolve_layers() -> None:
    global _WMSMETA
    meta: Dict[str, Dict[str, str]] = {}
    fgr = _find_layer_name(FGR_WMS, ["fysisch geografische regio"])
    bodem = _find_layer_name(BODEM_WMS, ["bodem", "soilarea"])
    gt = _find_layer_name(GWD_WMS, ["Grondwatertrappen", "Gt"])
    ghg = _find_layer_name(GWD_WMS, ["GHG"])
    glg = _find_layer_name(GWD_WMS, ["GLG"])
    ahn = _find_layer_name(AHN_WMS, ["dtm", "0,5m", "ahn"])

    if fgr:
        meta["fgr"] = {"url": FGR_WMS, "layer": fgr[0], "title": fgr[1]}
    if bodem:
        meta["bodem"] = {"url": BODEM_WMS, "layer": bodem[0], "title": bodem[1]}
    if gt:
        meta["gt"] = {"url": GWD_WMS, "layer": gt[0], "title": gt[1]}
    if ghg:
        meta["ghg"] = {"url": GWD_WMS, "layer": ghg[0], "title": ghg[1]}
    if glg:
        meta["glg"] = {"url": GWD_WMS, "layer": glg[0], "title": glg[1]}
    if ahn:
        meta["ahn"] = {"url": AHN_WMS, "layer": ahn[0], "title": ahn[1]}

    _WMSMETA = meta
    print("[WMS] resolved:", meta)


_resolve_layers()

_kv_re = re.compile(r"^\s*([A-Za-z0-9_\-\. ]+?)\s*[:=]\s*(.+?)\s*$")

_DEF_INFO_FORMATS = [
    "application/json",
    "application/geo+json",
    "application/json;subtype=geojson",
    "application/vnd.ogc.gml",
    "text/xml",
    "text/plain",
]


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


def _wms_getfeatureinfo(base_url: str, layer: str, lat: float, lon: float) -> Optional[dict]:
    cx, cy = TX_WGS84_WEB.transform(lon, lat)
    m = 200.0
    bbox = f"{cx-m},{cy-m},{cx+m},{cy+m}"
    params_base = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetFeatureInfo",
        "layers": layer,
        "query_layers": layer,
        "styles": "",
        "crs": "EPSG:3857",
        "width": 101,
        "height": 101,
        "i": 50,
        "j": 50,
        "bbox": bbox,
        "feature_count": 10,
    }
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
        except Exception as e:
            print("[GFI] fout:", e, "fmt=", fmt)
            continue
    return None


# ───────────────────── FGR / bodem / GWT / AHN helpers
def fgr_from_point(lat: float, lon: float) -> Optional[str]:
    try:
        x, y = TX_WGS84_RD.transform(lon, lat)
        bbox = f"{x-1},{y-1},{x+1},{y+1}"
        url = (
            PDOK_FGR_WFS
            + "&request=GetFeature&typename=fysischgeografischeregios"
            + "&srsName=EPSG:28992"
            + "&bbox=" + bbox
            + ",EPSG:28992"
            + "&outputFormat=" + FMT_JSON
        )
        r = _get(url)
        if not r.ok:
            return None
        data = r.json()
        feats = data.get("features") or []
        if not feats:
            return None
        props = feats[0].get("properties") or {}
        return props.get("NAAM") or props.get("naam") or None
    except Exception as e:
        print("[FGR] fout:", e)
        return None


def bodem_from_bodemkaart(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    layer = _WMSMETA.get("bodem", {}).get("layer") or "soilarea"
    props = _wms_getfeatureinfo(BODEM_WMS, layer, lat, lon) or {}
    code = None
    for key in ("bodemtype", "soiltype", "SOILTYPE"):
        if key in props and props[key]:
            code = str(props[key])
            break
    if not code and "_text" in props:
        kv = _parse_kv_text(props["_text"])
        for key in ("bodemtype", "soiltype", "SOILTYPE"):
            if key in kv and kv[key]:
                code = str(kv[key])
                break
    bodem = None
    if code:
        c = code.lower()
        if any(k in c for k in ("zand", "sand")):
            bodem = "zand"
        elif "klei" in c or "clay" in c:
            bodem = "klei"
        elif "veen" in c or "peat" in c:
            bodem = "veen"
        elif "leem" in c or "loess" in c or "löss" in c:
            bodem = "leem"
    return bodem, props


_GT_MAPPING = {
    "Ia": "zeer nat",
    "Ib": "zeer nat",
    "II": "nat",
    "III": "nat",
    "IV": "vochtig",
    "IVa": "vochtig",
    "IVb": "vochtig",
    "IVc": "vochtig",
    "V": "droog",
    "VI": "droog",
    "VII": "zeer droog",
    "VIII": "zeer droog",
}
_GT_NUMERIC = {
    "1": "zeer nat",
    "2": "zeer nat",
    "3": "nat",
    "4": "nat",
    "5": "vochtig",
    "6": "vochtig",
    "7": "droog",
    "8": "droog",
    "9": "zeer droog",
    "10": "zeer droog",
}


def _gt_pretty(gt: Optional[str]) -> Optional[str]:
    if not gt:
        return None
    s = gt.strip()
    # Romeins + letter
    if s.upper().startswith(("I", "V", "X")):
        return s.upper()
    return s


def _vochtklasse_from_gt_code(gt: Optional[str]) -> Optional[str]:
    if not gt:
        return None
    s = gt.strip()
    if not s:
        return None
    if s.upper() in _GT_MAPPING:
        return _GT_MAPPING[s.upper()]
    if s in _GT_NUMERIC:
        return _GT_NUMERIC[s]
    m = re.match(r"([IVX]+)([a-z]?)", s, re.I)
    if m:
        base = m.group(1).upper()
        return _GT_MAPPING.get(base, None)
    return None


def vocht_from_gwt(lat: float, lon: float) -> Tuple[Optional[str], dict, Optional[str]]:
    gt_layer = _WMSMETA.get("gt", {}).get("layer") or "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt"
    props = _wms_getfeatureinfo(GWD_WMS, gt_layer, lat, lon) or {}

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

    if not gt_raw and "value_list" in props and str(props["value_list"]).strip():
        gt_raw = str(props["value_list"]).strip()

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
                if depth < 25:
                    klass = "zeer nat"
                elif depth < 40:
                    klass = "nat"
                elif depth < 80:
                    klass = "vochtig"
                elif depth < 120:
                    klass = "droog"
                else:
                    klass = "zeer droog"
                return klass, p2, _gt_pretty(gt_raw)

    return klass, props, _gt_pretty(gt_raw)


def ahn_from_point(lat: float, lon: float) -> Tuple[Optional[float], dict]:
    """
    Haal AHN-hoogte (maaiveld t.o.v. NAP, in meters) op via PDOK AHN WMS.
    """
    layer = _WMSMETA.get("ahn", {}).get("layer") or "dtm_05m"
    props = _wms_getfeatureinfo(AHN_WMS, layer, lat, lon) or {}

    hoogte: Optional[float] = None
    for key in ("value_list", "value", "GRAY_INDEX", "GRAYINDEX", "Band1", "band_1"):
        if key in props and props[key] not in (None, ""):
            try:
                hoogte = float(str(props[key]).split(",")[0].strip())
                break
            except (TypeError, ValueError):
                continue

    if hoogte is None and "_text" in props:
        m = re.search(r"(-?\d+(?:\.\d+)?)", str(props["_text"]))
        if m:
            try:
                hoogte = float(m.group(1))
            except ValueError:
                pass

    return hoogte, props


def ahn_relief_from_point(lat: float, lon: float) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """
    Bepaal lokale relatieve positie (hoogte/laagte/vlak) uit AHN rondom het punt.
    """
    center_h, _ = ahn_from_point(lat, lon)
    if center_h is None:
        return None, None, None

    dlat = 100.0 / 111_000.0
    dlon = 100.0 / 71_000.0

    neighbors: List[float] = []
    for dlat_i, dlon_i in ((dlat, 0.0), (-dlat, 0.0), (0.0, dlon), (0.0, -dlon)):
        h, _ = ahn_from_point(lat + dlat_i, lon + dlon_i)
        if h is not None:
            neighbors.append(h)

    if not neighbors:
        return center_h, None, None

    avg = sum(neighbors) / len(neighbors)
    delta = center_h - avg  # positief = hoger dan omgeving

    relief: Optional[str]
    if delta > 0.4:
        relief = "hoogte"
    elif delta < -0.4:
        relief = "laagte"
    else:
        relief = "vlak"

    return center_h, relief, delta


_VOCHT_SCALE = ["zeer nat", "nat", "vochtig", "droog", "zeer droog"]


def _vocht_met_ahn_correctie(vocht: Optional[str], relief: Optional[str]) -> Optional[str]:
    """
    Corrigeer vochtklasse 1 stap natter/droger op basis van lokaal reliëf.
    """
    if not vocht or not relief or relief not in ("hoogte", "laagte"):
        return vocht
    v = str(vocht).strip().lower()
    try:
        idx = _VOCHT_SCALE.index(v)
    except ValueError:
        return vocht
    if relief == "hoogte" and idx < len(_VOCHT_SCALE) - 1:
        idx += 1
    elif relief == "laagte" and idx > 0:
        idx -= 1
    return _VOCHT_SCALE[idx]


# ───────────────────── filtering helpers
def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()


def _split_tokens(cell: Any) -> List[str]:
    return [
        t.strip().lower()
        for t in str(cell or "").replace("/", ";").replace("|", ";").split(";")
        if t.strip()
    ]


def _match_multival(cell: Any, choices: List[str]) -> bool:
    if not choices:
        return True
    tokens = set(_split_tokens(cell))
    want = set(w.strip().lower() for w in choices if w.strip())
    return bool(tokens.intersection(want))


def _match_bodem_row(row: pd.Series, keuzes: List[str]) -> bool:
    if not keuzes:
        return True
    low = [k.lower() for k in keuzes]
    if "bodem" in row and _match_multival(row.get("bodem"), low):
        return True
    gs = str(row.get("grondsoorten", "")).lower()
    cats = set()
    if "zand" in gs:
        cats.add("zand")
    if "klei" in gs:
        cats.add("klei")
    if any(w in gs for w in ("leem", "löss", "loess")):
        cats.add("leem")
    if "veen" in gs:
        cats.add("veen")
    return bool(set(low).intersection(cats))


# ───────────────────── app + cleaners
app = FastAPI(title="PlantWijs API v3.9.2")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)


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
def api_diag_featureinfo(
    lat: float = Query(...),
    lon: float = Query(...),
    layer: str = Query(..., description="fgr|bodem|gt|ghg|glg|ahn"),
):
    meta = _WMSMETA.get(layer)
    if not meta:
        return JSONResponse({"error": "onbekende layer"}, status_code=400)
    props = _wms_getfeatureinfo(meta["url"], meta["layer"], lat, lon) or {}
    return JSONResponse(_clean({"layer": layer, "props": props}))


# ───────────────────── API: data-export & search
@app.get("/api/columns")
def api_columns():
    df = get_df()
    return JSONResponse(list(df.columns))


@app.get("/api/export")
def api_export():
    df = get_df()
    buf = io.StringIO()
    df.to_csv(buf, sep=";", index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="plantwijs_export.csv"'},
    )


@app.get("/api/search")
def api_search(
    q: str = Query("", description="vrij zoeken in naam / wetenschappelijke_naam"),
    licht: List[str] = Query([], alias="licht[]"),
    vocht: List[str] = Query([], alias="vocht[]"),
    bodem: List[str] = Query([], alias="bodem[]"),
    inheems_only: bool = True,
    exclude_invasief: bool = True,
    limit: int = Query(200, ge=1, le=2000),
    sort: str = Query("ecowaarde"),
    desc: bool = True,
):
    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[
            (df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())
        ]
    if q:
        df = df[
            df["naam"].map(lambda v: _contains_ci(v, q))
            | df["wetenschappelijke_naam"].map(lambda v: _contains_ci(v, q))
        ]
    if licht:
        df = df[df["standplaats_licht"].map(lambda v: _match_multival(v, licht))]
    if vocht:
        df = df[df["vocht"].map(lambda v: _match_multival(v, vocht))]
    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)
    cols = [
        c
        for c in (
            "naam",
            "wetenschappelijke_naam",
            "inheems",
            "invasief",
            "standplaats_licht",
            "vocht",
            "bodem",
            "ellenberg_l",
            "ellenberg_f",
            "ellenberg_t",
            "ellenberg_n",
            "ellenberg_r",
            "ellenberg_s",
            "ellenberg_l_min",
            "ellenberg_l_max",
            "ellenberg_f_min",
            "ellenberg_f_max",
            "ellenberg_t_min",
            "ellenberg_t_max",
            "ellenberg_n_min",
            "ellenberg_n_max",
            "ellenberg_r_min",
            "ellenberg_r_max",
            "ellenberg_s_min",
            "ellenberg_s_max",
            "hoogte",
            "breedte",
            "winterhardheidszone",
            "grondsoorten",
            "ecowaarde",
        )
        if c in df.columns
    ]
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
    ahn_hoogte, ahn_relief, ahn_delta = ahn_relief_from_point(lat, lon)

    bodem_val = bodem_raw or "leem"
    vocht_corr = _vocht_met_ahn_correctie(vocht_raw, ahn_relief)
    vocht_val = vocht_corr or vocht_raw

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[
            (df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())
        ]

    if vocht_val:
        df = df[df["vocht"].map(lambda v: _match_multival(v, [vocht_val]))]
    df = df[
        df.apply(
            lambda r: _match_bodem_row(r, [bodem_val]) if bodem_val else True, axis=1
        )
    ]

    cols = [
        c
        for c in (
            "naam",
            "wetenschappelijke_naam",
            "inheems",
            "invasief",
            "standplaats_licht",
            "vocht",
            "bodem",
            "ellenberg_l",
            "ellenberg_f",
            "ellenberg_t",
            "ellenberg_n",
            "ellenberg_r",
            "ellenberg_s",
            "hoogte",
            "breedte",
            "winterhardheidszone",
            "grondsoorten",
            "ecowaarde",
        )
        if c in df.columns
    ]
    items = df[cols].head(limit).to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_val,
        "bodem_bron": "BRO Bodemkaart WMS" if bodem_raw else "onbekend",
        "gt_code": gt_code,
        "vocht": vocht_val,
        "vocht_basis": vocht_raw,
        "vocht_bron": "BRO Gt/GLG WMS" if vocht_raw else "onbekend",
        "vocht_ahn_correctie": bool(vocht_corr and vocht_corr != vocht_raw),
        "ahn_hoogte_m": ahn_hoogte,
        "ahn_relief": ahn_relief,
        "ahn_delta_m": ahn_delta,
        "advies": items,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    return JSONResponse(_clean(out))


# ───────────────────── UI
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
    :root { --bg:#0b1321; --panel:#0f192e; --muted:#9aa4b2; --fg:#e6edf3; --border:#1c2a42;
            --accent:#4ade80; --accent-soft:rgba(74,222,128,.08); --danger:#fb7185;
            --chip-bg:#111827; --chip-border:#1f2937; --chip-muted:#6b7280; }

    * { box-sizing:border-box; }
    body {
      margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:radial-gradient(circle at top,#111827 0,#020617 55%,#000 100%);
      color:var(--fg); height:100vh; display:flex; align-items:stretch; justify-content:stretch;
      overflow:hidden;
    }
    body.light {
      background:#f3f4f6; color:#020617;
      --bg:#f3f4f6; --panel:#ffffff; --muted:#6b7280; --fg:#020617; --border:#e5e7eb;
      --accent:#16a34a; --accent-soft:rgba(22,163,74,.08); --chip-bg:#f9fafb; --chip-border:#e5e7eb;
      --chip-muted:#9ca3af;
    }

    #root { display:flex; flex:1; max-width:1600px; margin:0 auto; padding:10px; gap:10px; }

    #mapWrap { flex:1 1 50%; min-width:0; border-radius:18px; overflow:hidden;
               box-shadow:0 18px 40px rgba(0,0,0,.45); border:1px solid rgba(148,163,184,.25);}
    #map { width:100%; height:100%; min-height:400px; }

    #side {
      flex:1 1 50%; min-width:0; display:flex; flex-direction:column;
      border-radius:18px; background:linear-gradient(145deg,#020617 0,#020617 0,#020617 30%,#0b1120 100%);
      border:1px solid rgba(148,163,184,.3); box-shadow:0 18px 40px rgba(15,23,42,.8);
      padding:14px 16px; gap:8px; position:relative; overflow:hidden;
    }
    body.light #side {
      background:linear-gradient(145deg,#f9fafb 0,#f9fafb 0,#f3f4f6 40%,#e5e7eb 100%);
      box-shadow:0 8px 25px rgba(15,23,42,.08);
    }

    #sideHeader { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    #sideTitle { display:flex; flex-direction:column; gap:2px; }
    #sideTitle h1 {
      margin:0; font-size:18px; letter-spacing:.05em; text-transform:uppercase;
      color:var(--muted);
    }
    #sideTitle .sub {
      font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.12em;
    }

    #btnTheme {
      border-radius:999px; border:1px solid rgba(148,163,184,.35); background:#020617;
      color:#e5e7eb; padding:4px 10px; font-size:11px; text-transform:uppercase;
      letter-spacing:.15em; cursor:pointer; display:flex; align-items:center; gap:6px;
    }
    #btnTheme span.ic { font-size:13px; }
    body.light #btnTheme { background:#f9fafb; color:#0f172a; }

    #chips {
      margin-top:4px; display:flex; flex-wrap:wrap; gap:6px;
    }
    .chip {
      font-size:11px; border-radius:999px; padding:4px 9px;
      background:var(--chip-bg); border:1px solid var(--chip-border); color:var(--chip-muted);
    }

    #filters {
      display:grid; grid-template-columns:1.1fr 1.1fr 1.1fr; gap:8px;
      margin-top:6px; align-items:start;
    }
    .filterBox {
      background:rgba(15,23,42,.96); border-radius:12px; padding:8px 9px;
      border:1px solid rgba(148,163,184,.35); box-shadow:0 10px 25px rgba(15,23,42,.75);
    }
    body.light .filterBox {
      background:#f9fafb; box-shadow:none; border-color:#d1d5db;
    }
    .filterBox h3 {
      margin:0 0 4px; font-size:12px; text-transform:uppercase; letter-spacing:.15em;
      color:var(--muted);
    }
    .filterBox label {
      display:inline-flex; align-items:center; gap:4px;
      font-size:11px; color:var(--muted); margin:2px 4px 2px 0;
      padding:3px 6px; border-radius:999px; cursor:pointer;
      border:1px solid transparent;
    }
    .filterBox input[type=checkbox] { accent-color:#22c55e; }
    .filterBox label:hover { border-color:rgba(148,163,184,.7); }

    #legendRow {
      display:flex; gap:12px; margin-top:8px; font-size:11px; color:var(--muted);
      align-items:center; flex-wrap:wrap;
    }
    #legendRow span.key {
      display:inline-flex; align-items:center; gap:4px; padding:3px 6px;
      border-radius:999px; background:rgba(15,23,42,.7); border:1px solid rgba(148,163,184,.4);
    }
    body.light #legendRow span.key {
      background:#eef2ff; border-color:#c7d2fe;
    }
    #legendRow span.key b { font-weight:600; color:var(--fg); }

    #warning {
      margin-top:4px; font-size:11px; color:var(--danger); display:none;
    }

    #results {
      margin-top:10px; flex:1 1 auto; min-height:0;
      background:rgba(15,23,42,.96); border-radius:12px;
      border:1px solid rgba(148,163,184,.4); padding:8px 0; overflow:hidden;
      display:flex; flex-direction:column;
    }
    body.light #results { background:#f9fafb; border-color:#d1d5db; }

    #resultsHeader {
      padding:0 10px 6px; display:flex; justify-content:space-between;
      align-items:center; gap:8px; font-size:11px; color:var(--muted);
    }
    #resultsHeader strong { color:var(--fg); font-weight:600; }

    #tblWrap { flex:1 1 auto; overflow:auto; }
    table {
      width:100%; border-collapse:collapse; font-size:11px; min-width:100%;
    }
    thead th {
      position:sticky; top:0; background:rgba(15,23,42,.98);
      color:#9ca3af; font-weight:500; padding:7px 10px; border-bottom:1px solid #1f2937;
      text-align:left; font-size:11px;
    }
    body.light thead th {
      background:#e5e7eb; border-bottom-color:#d1d5db; color:#4b5563;
    }
    tbody tr:nth-child(even) { background:rgba(15,23,42,.88); }
    body.light tbody tr:nth-child(even) { background:#f3f4f6; }
    tbody tr:nth-child(odd) { background:transparent; }
    tbody td {
      padding:6px 10px; border-bottom:1px solid rgba(31,41,55,.85);
      color:var(--fg); font-size:11px; vertical-align:top;
    }
    body.light tbody td { border-bottom-color:#e5e7eb; }

    tbody tr:hover { background:rgba(56,189,248,.12); }

    .badge {
      display:inline-block; font-size:10px; padding:2px 6px; border-radius:999px;
      border:1px solid rgba(148,163,184,.35); margin-right:4px; color:var(--muted);
    }
    .badge.inheems { border-color:var(--accent); color:var(--accent); }
    .badge.invasief { border-color:var(--danger); color:var(--danger); }

    @media (max-width:1024px) {
      #root { flex-direction:column; height:auto; }
      #side { order:-1; max-height:none; }
      #mapWrap { min-height:340px; }
    }

  </style>
</head>
<body>
  <div id="root">
    <div id="mapWrap"><div id="map"></div></div>

    <div id="side">
      <div id="sideHeader">
        <div id="sideTitle">
          <h1>PlantWijs</h1>
          <div class="sub">Inheemse beplanting op maat</div>
          <div id="chips"></div>
        </div>
        <button id="btnTheme"><span class="ic">☾</span> THEMA</button>
      </div>

      <div id="filters">
        <div class="filterBox">
          <h3>Licht</h3>
          <label><input type="checkbox" name="licht" value="schaduw">schaduw</label>
          <label><input type="checkbox" name="licht" value="halfschaduw">halfschaduw</label>
          <label><input type="checkbox" name="licht" value="zon">zon</label>
        </div>
        <div class="filterBox">
          <h3>Vocht</h3>
          <label><input type="checkbox" name="vocht" value="zeer nat">zeer nat</label>
          <label><input type="checkbox" name="vocht" value="nat">nat</label>
          <label><input type="checkbox" name="vocht" value="vochtig">vochtig</label>
          <label><input type="checkbox" name="vocht" value="droog">droog</label>
          <label><input type="checkbox" name="vocht" value="zeer droog">zeer droog</label>
        </div>
        <div class="filterBox">
          <h3>Bodem</h3>
          <label><input type="checkbox" name="bodem" value="zand">zand</label>
          <label><input type="checkbox" name="bodem" value="klei">klei</label>
          <label><input type="checkbox" name="bodem" value="leem">leem / löss</label>
          <label><input type="checkbox" name="bodem" value="veen">veen</label>
        </div>
      </div>

      <div id="legendRow">
        <span class="key"><b>FGR</b> <span id="lgF">—</span></span>
        <span class="key"><b>Bodem</b> <span id="lgB">—</span></span>
        <span class="key"><b>Gt → Vocht</b> <span id="lgG">—</span></span>
        <span class="key"><b>AHN</b> <span id="lgA">—</span></span>
      </div>
      <div id="warning"></div>

      <div id="results">
        <div id="resultsHeader">
          <div><strong id="resCount">0 soorten</strong> <span id="resInfo"></span></div>
        </div>
        <div id="tblWrap">
          <table id="tbl">
            <thead>
              <tr>
                <th>Naam</th>
                <th>Wetenschappelijke naam</th>
                <th>Licht</th>
                <th>Vocht</th>
                <th>Bodem</th>
                <th>WHZ</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

    </div>
  </div>

  <script>
    const map = L.map('map').setView([52.1, 5.3], 8);
    const base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    let overlays = {};
    let ui = { meta:null, ctx:{vocht:null,bodem:null} };

    function html(s){ return (s==null?'':String(s)).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;') }
    function getChecked(name){
      return Array.from(document.querySelectorAll('input[name="'+name+'"]:checked')).map(x=>x.value);
    }
    function renderChips(){
      const ctn = document.getElementById('chips');
      const chips = [];
      for(const [lbl,name] of [['licht','licht'],['vocht','vocht'],['bodem','bodem']]){
        const vals = getChecked(name);
        if(vals.length){ chips.push(`<span class="chip">${lbl}: ${vals.map(html).join(' / ')}</span>`); }
      }
      ctn.innerHTML = chips.join(' ');
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

    const InfoCtl = L.Control.extend({
      onAdd: function() {
        const div = L.DomUtil.create('div', 'pw-ctl');
        div.style.background = 'rgba(15,23,42,.9)';
        div.style.color = '#e5e7eb';
        div.style.padding = '8px 10px';
        div.style.borderRadius = '10px';
        div.style.fontSize = '11px';
        div.style.border = '1px solid rgba(148,163,184,.5)';
        div.innerHTML = `
          <div style="font-size:11px;font-weight:600;margin-bottom:4px;">Klik-info</div>
          <div id="uiF">FGR: —</div>
          <div id="uiB">Bodem: —</div>
          <div id="uiG">Gt → Vocht: —</div>
          <div id="uiA">AHN: —</div>
        `;
        L.DomEvent.disableClickPropagation(div);
        return div;
      }
    });
    const infoCtl = new InfoCtl({ position:'topright' }).addTo(map);

    function setLegend(resp){
      document.getElementById('lgF').textContent = resp.fgr || '—';
      let bodemTxt = resp.bodem || '—';
      if(resp.bodem_bron && resp.bodem_bron !== 'onbekend'){
        bodemTxt += ' ('+resp.bodem_bron+')';
      }
      document.getElementById('lgB').textContent = bodemTxt;

      let gtTxt = resp.gt_code || '—';
      if(resp.vocht_basis){
        gtTxt += ' → '+resp.vocht_basis;
      }
      if(resp.vocht && resp.vocht !== resp.vocht_basis){
        gtTxt += ' → '+resp.vocht+' (met AHN)';
      }
      document.getElementById('lgG').textContent = gtTxt;

      if(resp.ahn_hoogte_m == null){
        document.getElementById('lgA').textContent = 'onbekend';
      } else {
        const h = resp.ahn_hoogte_m.toFixed(2).replace('.',',');
        let txt = h+' m NAP';
        if(resp.ahn_relief){
          txt += ' · '+resp.ahn_relief;
        }
        if(typeof resp.ahn_delta_m === 'number'){
          const d = resp.ahn_delta_m;
          txt += ' (Δ '+(d>=0?'+':'')+d.toFixed(2).replace('.',',')+' m)';
        }
        document.getElementById('lgA').textContent = txt;
      }

      const ciF = document.getElementById('uiF');
      const ciB = document.getElementById('uiB');
      const ciG = document.getElementById('uiG');
      const ciA = document.getElementById('uiA');
      if(ciF){
        ciF.textContent = 'FGR: '+(resp.fgr || '—');
      }
      if(ciB){
        ciB.textContent = 'Bodem: '+bodemTxt;
      }
      if(ciG){
        ciG.textContent = 'Gt → Vocht: '+gtTxt;
      }
      if(ciA){
        ciA.textContent = document.getElementById('lgA').textContent;
      }
    }

    function renderTable(items){
      const tbody = document.querySelector('#tbl tbody');
      tbody.innerHTML = '';
      for(const row of items){
        const tr = document.createElement('tr');
        const badges = [];
        if(String(row.inheems||'').toLowerCase()==='ja'){
          badges.push('<span class="badge inheems">inheems</span>');
        }
        if(String(row.invasief||'').toLowerCase()==='ja'){
          badges.push('<span class="badge invasief">invasief</span>');
        }
        tr.innerHTML = `
          <td>${badges.join('')}${html(row.naam)}</td>
          <td><span style="font-style:italic;">${html(row.wetenschappelijke_naam)}</span></td>
          <td>${html(row.standplaats_licht)}</td>
          <td>${html(row.vocht)}</td>
          <td>${html(row.bodem)}</td>
          <td>${html(row.winterhardheidszone)}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    async function fetchAdvies(lat, lon){
      const params = new URLSearchParams({
        lat: String(lat),
        lon: String(lon),
        inheems_only: 'true',
        exclude_invasief: 'true',
        limit: '150',
      });
      const res = await fetch('/advies/geo?'+params.toString());
      if(!res.ok) throw new Error('Advies-fout');
      return await res.json();
    }

    function updateWarning(ctx){
      const el = document.getElementById('warning');
      const useV = ctx && ctx.vocht;
      const useB = ctx && ctx.bodem;
      const chosenV = getChecked('vocht');
      const chosenB = getChecked('bodem');

      const msgs = [];
      if(useV && chosenV.length){
        msgs.push('Let op: vocht wordt bepaald door Gt/AHN én handmatige filter.');
      }
      if(useB && chosenB.length){
        msgs.push('Let op: bodem wordt bepaald door BRO én handmatige filter.');
      }
      el.textContent = msgs.join(' ');
      el.style.display = msgs.length ? 'block' : 'none';
    }

    map.on('click', async (e)=>{
      const {lat, lng} = e.latlng;
      try{
        const resp = await fetchAdvies(lat, lng);
        renderTable(resp.advies || []);
        setLegend(resp);
        document.getElementById('resCount').textContent = (resp.advies||[]).length+' soorten';
        document.getElementById('resInfo').textContent =
          'Klik op de kaart om advies te vernieuwen. Tijdsduur: '+resp.elapsed_ms+' ms';

        ui.ctx = {
          vocht: resp.vocht || resp.vocht_basis || null,
          bodem: resp.bodem || null
        };
        updateWarning(ui.ctx);
      }catch(err){
        console.error(err);
      }
    });

    for(const name of ['licht','vocht','bodem']){
      document.querySelectorAll('input[name="'+name+'"]').forEach(cb=>{
        cb.addEventListener('change', ()=>{
          renderChips();
          updateWarning(ui.ctx);
        });
      });
    }
    renderChips();
  </script>
</body>
</html>
    """
    return HTMLResponse(html)
