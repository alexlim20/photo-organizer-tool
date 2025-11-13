import sys, tempfile, os, re, csv, shutil, subprocess, json
from threading import Thread
import queue
import time
import reverse_geocoder as rg
import pycountry

DMS_PAIR = re.compile(
    r"""^\s*
        (?P<lat_deg>\d+(?:\.\d+)?)\s*deg\s*(?P<lat_min>\d+(?:\.\d+)?)'\s*(?P<lat_sec>\d+(?:\.\d+)?)"\s*(?P<lat_hem>[NS])
        \s*[, ]\s*
        (?P<lon_deg>\d+(?:\.\d+)?)\s*deg\s*(?P<lon_min>\d+(?:\.\d+)?)'\s*(?P<lon_sec>\d+(?:\.\d+)?)"\s*(?P<lon_hem>[EW])
        \s*$""",
    re.IGNORECASE | re.VERBOSE
)

# ---------- Configuration for AAE Files ----------
import xml.etree.ElementTree as ET
import base64
import zlib
import json
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

# ---------- Safety helpers: hashing, atomic copy, manifest ----------
from datetime import datetime
try:
    from send2trash import send2trash
except Exception:
    send2trash = None

def file_quick_hash(path, chunk=1024 * 1024):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_copy_atomic(src, dst_dir):
    ensure_dir(dst_dir)
    src = os.path.normpath(src)
    dst_dir = os.path.normpath(dst_dir)
    final_path = os.path.join(dst_dir, os.path.basename(src))
    final_path = unique_path(final_path)

    tmp_path = final_path + ".partial"
    with open(src, "rb") as r, open(tmp_path, "wb") as w:
        shutil.copyfileobj(r, w, length=1024 * 1024)
        try:
            w.flush()
            os.fsync(w.fileno())
        except Exception:
            pass
    os.replace(tmp_path, final_path)
    return final_path

# ------- Session/manifest (JSON Lines) -------
def _session_id_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def pick_incoming_session():
    """Pick newest session dir inside INCOMING_ROOT, or None."""
    if not os.path.isdir(INCOMING_ROOT):
        return None
    subs = [os.path.join(INCOMING_ROOT, d) for d in os.listdir(INCOMING_ROOT)]
    subs = [p for p in subs if os.path.isdir(p)]
    if not subs:
        return None
    subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subs[0]

def ensure_session_dir():
    """Create a new session dir if none exists (helps CLI/manual use)."""
    sess = pick_incoming_session()
    if sess:
        return sess
    sess = os.path.join(INCOMING_ROOT, _session_id_now())
    os.makedirs(sess, exist_ok=True)
    return sess

def manifest_paths(session_dir):
    # Keep manifest in a hidden subfolder to avoid UI/AV locks on Windows
    state_dir = os.path.join(session_dir, "._state")
    os.makedirs(state_dir, exist_ok=True)
    return (
        os.path.join(state_dir, "session.meta.json"),
        os.path.join(state_dir, "session.jsonl"),
    )

def manifest_load_meta(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"phase": "staging", "total": 0, "planned": 0, "copied": 0, "verified": 0, "duplicate": 0, "error": 0}

def manifest_save_meta(meta_path, meta):
    # ensure parent exists (._state may be created lazily)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    # Windows: retry a few times if something has a transient lock
    for i in range(6):
        try:
            os.replace(tmp, meta_path)
            return
        except PermissionError:
            time.sleep(0.15)
        except FileNotFoundError:
            # parent or target got removed (e.g., after auto-empty) – stop quietly
            return
    os.replace(tmp, meta_path)

def manifest_append(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def manifest_load_status_by_src(jsonl_path):
    """
    Read session.jsonl and build a map:
        abs(source_path) -> last known status string
    Used for crash-safe resume: we skip files already verified/duplicate.
    """
    status = {}
    if not os.path.exists(jsonl_path):
        return status
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                src = rec.get("from")
                st  = rec.get("status")
                if not src or not st:
                    continue
                status[os.path.abspath(src)] = st
    except Exception as e:
        print(f"(warn) could not read manifest jsonl for resume: {e}")
    return status

LIB_HASH_INDEX = {}  # global: digest -> [paths]

def build_library_hash_index(lib_root):
    """
    Build an in-memory hash index of existing media in the library.
    NOTE: This can be expensive on huge libraries (one hash per file),
    but is simple and safe for now.
    """
    index = {}
    for root, dirs, files in os.walk(lib_root):
        # Skip staging + trash-like internals
        rel = os.path.relpath(root, lib_root)
        parts = rel.split(os.sep)
        if parts and parts[0].startswith("_incoming"):
            # don't consider _incoming as "library"
            dirs[:] = []
            continue
        if parts and parts[0].startswith("._"):
            # skip hidden internal dirs (._state, ._trash, etc.)
            continue

        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                continue
            path = os.path.join(root, fn)
            try:
                h = file_quick_hash(path)
            except Exception as e:
                print(f"(warn) failed to hash library file {path}: {e}")
                continue
            index.setdefault(h, []).append(path)
    return index

def lib_contains_hash(lib_root, digest, stop_after_first=True):
    """
    Check if the library hash index already contains this digest.
    The lib_root param is kept for signature compatibility but not used.
    """
    hits = LIB_HASH_INDEX.get(digest)
    return bool(hits)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXIFTOOL_EXE = os.path.join(SCRIPT_DIR, "exiftool.exe")
if not os.path.exists(EXIFTOOL_EXE):
    alt = os.path.join(SCRIPT_DIR, "exiftool(-k).exe")
    if os.path.exists(alt):
        EXIFTOOL_EXE = alt

# --- Base folder setup (user chooses or default = script folder) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FOLDER = os.path.join(SCRIPT_DIR, "photos_to_organize")
os.makedirs(BASE_FOLDER, exist_ok=True)

# Library/staging roots (backend-first refactor)
LIBRARY_ROOT = BASE_FOLDER
INCOMING_ROOT = os.path.join(LIBRARY_ROOT, "_incoming")
os.makedirs(INCOMING_ROOT, exist_ok=True)

# Behavior flags (can be toggled later from UI)
DRY_RUN = False          # backend supports it now; default off for CLI use
AUTO_EMPTY = True        # auto-empty incoming session after verified run
DUPLICATE_POLICY = "skip"   # "skip" or "copy_rename" (set by UI)

LOG_CSV = os.path.join(LIBRARY_ROOT, "organize_fast_log.csv")

# --- File types we care about ---
IMAGE_EXTS = {".jpg",".jpeg",".png",".heic",".webp",".tif",".tiff", ".dng", ".arw", ".gif"}
VIDEO_EXTS = {".mov",".mp4"}

def dms_to_decimal(d, m, s, hem):
    val = float(d) + float(m)/60.0 + float(s)/3600.0
    if hem.upper() in ("S", "W"):
        val = -val
    return val

def parse_gps_string(s):
    if not isinstance(s, str):
        return None
    t = s.strip()
    # DMS like: 49 deg 0' 41.40" N, 12 deg 5' 57.84" E
    m = DMS_PAIR.match(t)
    if m:
        return (
            dms_to_decimal(m.group("lat_deg"), m.group("lat_min"), m.group("lat_sec"), m.group("lat_hem")),
            dms_to_decimal(m.group("lon_deg"), m.group("lon_min"), m.group("lon_sec"), m.group("lon_hem")),
        )
    # ISO-6709 like: +49.0115+12.0994/
    v = parse_iso6709(t)
    if v:
        return v
    # plain decimal "lat, lon" or "lat lon"
    t2 = t.replace(",", " ")
    parts = [p for p in t2.split() if p]
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except:
            pass
    return None

# ---------- Make a folder if it does not exist. ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- Get a unique file path (avoid overwrites) ----------
def unique_path(dest_path):
    if not os.path.exists(dest_path): return dest_path
    base, ext = os.path.splitext(dest_path)
    i = 1
    while True:
        cand = f"{base}-{i}{ext}"
        if not os.path.exists(cand): return cand
        i += 1

def assert_inside_library(dest_dir):
    """
    Ensure destination directory is inside the library root.
    Raises RuntimeError if something tries to write outside LIBRARY_ROOT.
    """
    real_dest = os.path.realpath(dest_dir)
    real_lib  = os.path.realpath(LIBRARY_ROOT)
    if not real_dest.startswith(real_lib):
        raise RuntimeError(f"Destination outside library root: {dest_dir}")

def assert_inside_library(dest_dir):
    """
    Ensure destination directory is inside the library root.
    Raises RuntimeError if something tries to write outside LIBRARY_ROOT.
    """
    real_dest = os.path.realpath(dest_dir)
    real_lib = os.path.realpath(LIBRARY_ROOT)
    if not real_dest.startswith(real_lib):
        raise RuntimeError(f"Destination outside library root: {dest_dir}")

# ---------- Moves a file safely to a new folder ----------
def safe_move(src, dst_dir):
    ensure_dir(dst_dir)
    # Normalize Unicode (fixes CJK, accents, etc.)
    src = os.path.normpath(src)
    dst_dir = os.path.normpath(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src))
    dst_path = unique_path(dst_path)
    shutil.move(src, dst_path)
    return dst_path

# ---------- ExifTool bulk (one pass) ----------
def run_exiftool_tree(root):
    # collect ONLY top-level files we care about
    files = []
    for fn in os.listdir(root):
        p = os.path.join(root, fn)
        if not os.path.isfile(p): 
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
            files.append(p)

    if not files:
        return []

    base_args = [
        EXIFTOOL_EXE,
        "-j", "-n",
        "-G1",                    # show tag groups; helps ensure we see where tags live
        "-api", "requestAll=3",   # surface all known tags
        "-api", "largefilesupport=1",
        "-charset", "filename=utf8",
        "-ee3",                   # dig into embedded/timed metadata in videos

        # times (still needed)
        "-time:all",

        # explicitly ask for all the GPS/location fields we saw in your sample JSON:
        "-GPSLatitude", "-GPSLongitude", "-GPSPosition",
        "-GPSCoordinates", "-Composite:GPSCoordinates",
        "-XMP:GPSLatitude", "-XMP:GPSLongitude",

        # QuickTime / Keys / ItemList variants + Android fields
        "-QuickTime:Location", "-QuickTime:GPSCoordinates",
        "-Keys:Location",
        "-ItemList:Location",
        "-UserData:GPSCoordinates",

        # plain 'location' keys (as seen in your dump)
        "-location", "-location-eng",
        "-progress",
    ]

    results = []
    CHUNK = 600  # reduce if you still hit the limit
    total = len(files)
    for i in range(0, total, CHUNK):
        chunk = files[i:i+CHUNK]

        # write paths to a temporary argfile (one per line)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tf:
            for p in chunk:
                # Normalize path to NFC (important for macOS & Windows Chinese filenames)
                norm_path = os.path.normpath(p)
                tf.write(norm_path + "\n")
            argfile = tf.name

        try:
            args = base_args + ["-@", argfile]   # tell exiftool to read the file list
            # show which overall range we’re in (once)
            chunk_lo, chunk_hi = i+1, min(i+CHUNK, total)
            sys.stdout.write(f"\n  → ExifTool chunk {chunk_lo}-{chunk_hi} / {total}\n")
            sys.stdout.flush()

            # run exiftool and stream its progress from stderr
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # line-buffered so progress appears promptly
            )

            # read stderr lines in a thread so stdout can't block
            def pump_stderr(pipe, q):
                for line in iter(pipe.readline, ''):
                    q.put(line)
                pipe.close()

            q = queue.Queue()
            t_err = Thread(target=pump_stderr, args=(proc.stderr, q), daemon=True)
            t_err.start()

            # --- pump stdout (drain JSON to avoid blocking) ---
            stdout_chunks = []

            def pump_stdout(pipe, sink_list):
                # read in chunks (JSON may not be line-delimited)
                for chunk in iter(lambda: pipe.read(65536), ''):
                    sink_list.append(chunk)
                pipe.close()

            t_out = Thread(target=pump_stdout, args=(proc.stdout, stdout_chunks), daemon=True)
            t_out.start()

            # live progress loop (stderr shows lines like "123/600 files processed")
            while True:
                try:
                    line = q.get(timeout=0.05)
                    msg = line.strip()
                    sys.stdout.write("\r     " + msg + " " * 20)  # update one status line
                    sys.stdout.flush()
                except queue.Empty:
                    if proc.poll() is not None and q.empty():
                        break
                    time.sleep(0.02)

            # finalize the progress line for this chunk
            sys.stdout.write("\n")
            sys.stdout.flush()

            # wait for threads to finish and get exit code
            t_err.join(timeout=1.0)
            t_out.join(timeout=1.0)
            ret = proc.wait()
            
            if ret != 0:
                print("  (warn) exiftool chunk failed")
            else:
                out = "".join(stdout_chunks)
                results.extend(json.loads(out))

            if results:
                # pick that exact file to confirm we see GPS keys now
                for rec in results:
                    src = rec.get("SourceFile") or rec.get("SourceFilePath") or ""
                    if src.endswith("PXL_20240831_150314486.mp4"):
                        gpsish = {k: rec[k] for k in rec if ("GPS" in k or "Location" in k or "location" in k or "Position" in k)}
                        print("CHECK GPS KEYS:", gpsish)
                        break
        finally:
            try: os.remove(argfile)
            except: pass

    return results


# ---------- Metadata helpers ----------
ISO6709 = re.compile(r"^([+\-]\d+(?:\.\d+)?)([+\-]\d+(?:\.\d+)?)(?:[+\-]\d+(?:\.\d+)?/)?$")

# ---------- Parse ISO-6709 string(e.g. +49.0115+12.0994) ----------
def parse_iso6709(s):
    m = ISO6709.match(s.strip().replace(" ", ""))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

# ---------- Extract lat/lon from metadata dict ----------
def extract_latlon(meta: dict):
    if not meta:
        return None

    # 1) Common explicit pairs (may be numeric with -n, or DMS strings)
    for klat, klon in [
        ("Composite:GPSLatitude", "Composite:GPSLongitude"),
        ("GPSLatitude", "GPSLongitude"),
        ("QuickTime:GPSLatitude", "QuickTime:GPSLongitude"),
        ("Keys:GPSLatitude", "Keys:GPSLongitude"),
        ("XMP:GPSLatitude", "XMP:GPSLongitude"),
    ]:
        lat = meta.get(klat)
        lon = meta.get(klon)
        if lat is not None and lon is not None:
            # Try numeric first
            try:
                return float(lat), float(lon)
            except Exception:
                # Then try DMS/decimal strings
                v = parse_gps_string(f"{lat}, {lon}")
                if v:
                    return v

    # 2) Single-string fields frequently used by videos
    for k in [
        "Composite:GPSPosition",
        "GPSPosition",
        "GPSCoordinates",
        "Composite:GPSCoordinates",
        "QuickTime:GPSCoordinates",
        "UserData:GPSCoordinates",
    ]:
        v = meta.get(k)
        if isinstance(v, str):
            out = parse_gps_string(v)
            if out:
                return out
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    out = parse_gps_string(item)
                    if out:
                        return out

    # 3) QuickTime/Keys and plain 'location' (often ISO-6709 like +49.0115+12.0994/)
    for k in ["QuickTime:Location", "Keys:Location", "location", "location-eng", "Location"]:
        v = meta.get(k)
        if isinstance(v, str):
            out = parse_iso6709(v)
            if out:
                return out

    # 4) Generic suffix fallback (…Latitude / …Longitude)
    lat_key = next((k for k in meta.keys() if k.lower().endswith("latitude")), None)
    lon_key = next((k for k in meta.keys() if k.lower().endswith("longitude")), None)
    if lat_key and lon_key:
        v = parse_gps_string(f"{meta.get(lat_key)}, {meta.get(lon_key)}")
        if v:
            return v

    # 5) FINAL SWEEP: scan all stringy metadata for anything that looks like coords
    for k, v in meta.items():
        if isinstance(v, str):
            out = parse_gps_string(v) or parse_iso6709(v)
            if out:
                return out
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    out = parse_gps_string(item) or parse_iso6709(item)
                    if out:
                        return out

    # 6) Google Photos Takeout sidecar fallback
    # If no GPS in metadata, try "<image>.json" next to the photo
    if "SourceFile" in meta:
        sidecar = meta["SourceFile"] + ".json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    j = json.load(f)
                for key in ("geoData", "geoDataExif"):
                    g = j.get(key) or {}
                    lat = g.get("latitude")
                    lon = g.get("longitude")
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        if lat != 0.0 or lon != 0.0:  # skip null coords
                            return float(lat), float(lon)
            except Exception as e:
                print(f"(warn) could not read sidecar for GPS: {sidecar} ({e})")

    return None

# 2) Helper: get a tag with or without a group prefix
def get_any(meta, *keys):
    for k in keys:
        # try unprefixed, e.g., "DateTimeOriginal"
        if k in meta and meta[k]:
            return meta[k]
        # try common groups, e.g., "EXIF:DateTimeOriginal", "QuickTime:CreateDate"
        for grp in ("EXIF", "QuickTime", "XMP", "Composite"):
            gk = f"{grp}:{k}"
            if gk in meta and meta[gk]:
                return meta[gk]
    return None

DATE_PAT = re.compile(r"(\d{4})[:\-\/.](\d{2})[:\-\/.](\d{2})")

# ---------- Extract YYYY-MM from metadata dict ----------
def extract_month(meta, abs_path=None):
    """
    Return 'YYYY-MM' from the best capture/create timestamp in `meta`.
    - Tries un/prefixed keys (EXIF:, XMP:, QuickTime:, Composite:).
    - Accepts strings OR lists from exiftool.
    - Parses common formats if DATE_PAT fails.
    - Keeps your filename/sidecar fallbacks.
    - Returns '' if nothing found.
    """
    base_tags = ["DateTimeOriginal", "CreateDate", "DateTimeDigitized", "ModifyDate"]
    groups = ("", "EXIF", "XMP", "QuickTime", "Composite")  # intentionally NOT 'File'

    # Build ordered key list (unprefixed + common groups)
    preferred_keys, seen = [], set()
    for tag in base_tags:
        for grp in groups:
            k = f"{grp + ':' if grp else ''}{tag}"
            if k not in seen:
                seen.add(k)
                preferred_keys.append(k)
    
    # extra DJI/EXIF fallbacks
    preferred_keys.extend([
        "IFD0:DateTime", "IFD0:ModifyDate",
        "Composite:DateTimeCreated", "Composite:DateTimeOriginal"
    ])

    # helper: try to parse a single value into YYYY-MM
    def _as_month(val):
        if val is None:
            return None
        # exiftool sometimes returns arrays; take first non-empty
        if isinstance(val, list):
            val = next((x for x in val if x), None)
        if val is None:
            return None

        # normalize to string
        s = str(val).strip()

        # 1) your original regex path (DATE_PAT) if you have one in scope
        try:
            m = DATE_PAT.search(s)  # uses your existing compiled regex
            if m:
                return f"{m.group(1)}-{m.group(2)}"
        except NameError:
            pass  # DATE_PAT not defined here; we’ll use formats below

        # 2) common datetime formats seen in EXIF/XMP/QuickTime
        from datetime import datetime
        for fmt in (
            "%Y:%m:%d %H:%M:%S",   # EXIF classic (DJI often this)
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y:%m:%d %H:%M:%S%z",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                return f"{dt.year:04d}-{dt.month:02d}"
            except Exception:
                pass

        # 3) numbers as epoch seconds (rare)
        try:
            iv = int(float(s))
            if iv > 0:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(iv, tz=timezone.utc)
                return f"{dt.year:04d}-{dt.month:02d}"
        except Exception:
            pass
        return None

    # Try metadata keys first (ignoring File:*)
    for k in preferred_keys:
        if k.startswith("File:"):
            continue
        if k in meta:
            month = _as_month(meta[k])
            if month:
                return month

    # Fallbacks: filename pattern (IMG_20240926_...), then Google Takeout sidecar
    import os, re, json
    if abs_path:
        fn = os.path.basename(abs_path)
        m = re.search(r"(20\d{2})(\d{2})(\d{2})[_-]", fn)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        sidecar = abs_path + ".json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    j = json.load(f)
                ts = (j.get("photoTakenTime") or {}).get("timestamp")
                if ts:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    return f"{dt.year:04d}-{dt.month:02d}"
            except Exception:
                pass

    dbg_keys = [
        "DateTimeOriginal","EXIF:DateTimeOriginal","Composite:SubSecDateTimeOriginal",
        "CreateDate","EXIF:CreateDate","XMP:CreateDate","XMP:DateCreated",
        "DateTimeDigitized","EXIF:ModifyDate","ModifyDate",
        "IFD0:DateTime","IFD0:ModifyDate","Composite:DateTimeCreated","Composite:DateTimeOriginal",
        "QuickTime:CreateDate","MediaCreateDate","TrackCreateDate"
    ]
    print("DEBUG no_date:", abs_path, {k: meta.get(k) for k in dbg_keys if k in meta})
    return ""


# ---------- Get stem variants for IMG_E#### <-> IMG_#### pairing ----------
def stem_for_pair(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(IMG_)E(\d+)$", name, re.IGNORECASE)
    if m:
        return {name, f"{m.group(1)}{m.group(2)}"}  # {"IMG_E2592", "IMG_2592"}
    return {name}

# ---------- Reverse geocode with caching & batch ----------
def coords_to_country_batch(latlon_list):
    # round for caching (3 decimals ~ 110m)
    rounded = [(round(lat,3), round(lon,3)) for (lat,lon) in latlon_list]
    unique = list(dict.fromkeys(rounded))
    if not unique:
        return {}
    results = rg.search(unique, mode=1)  # batch lookup
    out = {}
    for (latlon, rec) in zip(unique, results):
        cc = rec.get("cc","")
        try:
            name = pycountry.countries.get(alpha_2=cc).name if cc else ""
        except Exception:
            name = cc or ""
        clean = re.sub(r"[^A-Za-z0-9 _\-]", "", name).strip().replace(" ", "_") or "Unknown"
        out[latlon] = clean
    return out  # map (lat_r,lon_r) -> "Country_Name"

def month_from_video_meta(meta):
    for k in [
        "EXIF:DateTimeOriginal","DateTimeOriginal",
        "QuickTime:CreateDate","CreateDate","XMP:CreateDate",
        "MediaCreateDate","TrackCreateDate","Composite:SubSecDateTimeOriginal"
    ]:
        val = meta.get(k)
        if not val:
            continue
        m = re.search(r"(\d{4})[:\-](\d{2})[:\-](\d{2})", str(val))
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return ""

def apply_aae_edits(image_path, aae_path=None):
    """
    NO-OP: Do not parse/apply AAE. 
    Leave the original photo untouched and do not create an edited copy.
    AAE sidecar files are left in place and are not moved.
    """
    return image_path


# ---------- Main ----------
def main():

    print("\n=== FAST ORGANIZE: one-pass exiftool + batched geocode ===\n")
    
    # Resolve roots (UI may have overridden BASE_FOLDER)
    global LIBRARY_ROOT, INCOMING_ROOT, LIB_HASH_INDEX
    LIBRARY_ROOT = BASE_FOLDER
    INCOMING_ROOT = os.path.join(LIBRARY_ROOT, "_incoming")
    ensure_dir(LIBRARY_ROOT)
    ensure_dir(INCOMING_ROOT)

    # Pick the newest incoming session; if none, create one (empty) to keep flow simple
    session_dir = pick_incoming_session()
    if not session_dir:
        session_dir = ensure_session_dir()

    meta_path, jsonl_path = manifest_paths(session_dir)
    meta = manifest_load_meta(meta_path)

    # Crash-safe resume: load previous per-file statuses from session.jsonl
    processed_status = manifest_load_status_by_src(jsonl_path)

    # Build library hash index for duplicate detection
    print("Building library hash index for duplicate detection…")
    LIB_HASH_INDEX = build_library_hash_index(LIBRARY_ROOT)
    print(f"Library hash index contains {len(LIB_HASH_INDEX)} distinct hashes.")

    meta["phase"] = "planning"
    manifest_save_meta(meta_path, meta)

    print(f"🧺 Using incoming session: {session_dir}")
    no_date_dir = os.path.join(LIBRARY_ROOT, "no_date")
    ensure_dir(no_date_dir)

    # 1) One exiftool pass for staged files only (top-level scan, like before)
    print("Scanning metadata with ExifTool... (staging session)")
    all_meta = run_exiftool_tree(session_dir)

    # 2) Index by absolute path, and map stems to file paths
    meta_by_path = {}
    images = []
    videos = []
    stem_to_videos = {}
    for m in all_meta:
        path = m.get("SourceFile") or m.get("SourceFilePath") or m.get("File:FileName")
        if not path: continue
        path = os.path.abspath(path)
        meta_by_path[path] = m
        ext = os.path.splitext(path)[1].lower()
        stem = os.path.splitext(os.path.basename(path))[0]
        # build IMG_E#### <-> IMG_#### pairing
        for s in stem_for_pair(path):
            if ext in VIDEO_EXTS:
                stem_to_videos.setdefault(s.lower(), []).append(path)
        if ext in IMAGE_EXTS:
            images.append(path)
        elif ext in VIDEO_EXTS:
            videos.append(path)

    # 3) Precompute lat/lon for videos (many photos will borrow these)
    video_latlon = {}
    for vp in videos:
        latlon = extract_latlon(meta_by_path[vp])
        if latlon: video_latlon[vp] = latlon

    # 4) Walk images: get month; get GPS (from self or paired video)
    actions = []
    coord_set = set()
    file_moves = []  # (src, dest_dir)
    for ip in images:
        m = meta_by_path.get(ip, {})
        month = extract_month(m, abs_path=ip)
        latlon = extract_latlon(m)

        # 🆕 try to borrow date and GPS from companion video
        stem = os.path.splitext(os.path.basename(ip))[0]
        if not month or not latlon:
            for s in stem_for_pair(ip):
                for vp in stem_to_videos.get(s.lower(), []):
                    vmeta = meta_by_path.get(vp, {})
                    if not month:
                        month_try = month_from_video_meta(vmeta)
                        if month_try:
                            month = month_try
                    if not latlon:
                        latlon = extract_latlon(vmeta)
                    if month and latlon:
                        break
                if month and latlon:
                    break

        if not latlon:
            # try companion video(s) by stem variants
            stem = os.path.splitext(os.path.basename(ip))[0]
            latlon = None
            for s in stem_for_pair(ip):
                for vp in stem_to_videos.get(s.lower(), []):
                    latlon = video_latlon.get(vp)
                    if latlon:
                        break
                if latlon:
                    break

        if not month:
            dest_dir = no_date_dir
            # AAE for no-date too (new)
            stem_name = os.path.splitext(os.path.basename(ip))[0]
            aae_path = os.path.join(os.path.dirname(ip), f"{stem_name}.AAE")
            edited_ip = apply_aae_edits(ip, aae_path)
            final_ip = edited_ip if os.path.exists(edited_ip) and edited_ip != ip else ip
            final_basename = os.path.basename(final_ip)
            file_moves.append((ip, dest_dir))
            actions.append({
                "file": final_basename, "from": ip, "to_dir": dest_dir,
                "month": "", "gps": "", "country": "", "note": "no_date",
                "original_file": os.path.basename(ip),
                "edited": edited_ip != ip,
                "final_path": final_ip
            })
            print(f"✔ {final_basename} -> no_date/ (no date)")
            continue

        # AAE edit step (new)
        stem_name = os.path.splitext(os.path.basename(ip))[0]
        aae_path = os.path.join(os.path.dirname(ip), f"{stem_name}.AAE")
        edited_ip = apply_aae_edits(ip, aae_path)
        final_ip = edited_ip if os.path.exists(edited_ip) and edited_ip != ip else ip
        final_basename = os.path.basename(final_ip)

        if latlon:
            lat_r, lon_r = round(latlon[0],3), round(latlon[1],3)
            coord_set.add((lat_r, lon_r))
            actions.append({
                "file": final_basename, "from": ip, "to_dir": f"{month}/country_?", 
                "month": month, "gps": True, "lat": lat_r, "lon": lon_r,
                "original_file": os.path.basename(ip),
                "edited": edited_ip != ip,
                "final_path": final_ip
            })
        else:
            actions.append({
                "file": final_basename, "from": ip, "to_dir": month, 
                "month": month, "gps": False,
                "original_file": os.path.basename(ip),
                "edited": edited_ip != ip,
                "final_path": final_ip
            })
    
    # --- Handle standalone videos (e.g., when there are no images) ---
    for vp in videos:
        vm = meta_by_path.get(vp, {})
        month = month_from_video_meta(vm) or extract_month(vm, abs_path=vp)
        latlon = extract_latlon(vm)
        print(f"[GPSDEBUG] {os.path.basename(vp)}  month={month}  latlon={latlon}")
        if not latlon:
            bad = {k: vm.get(k) for k in vm.keys() if "GPS" in k or "Location" in k or "position" in k.lower()}
            print(f"DEBUG GPS? {os.path.basename(vp)} -> candidates: {bad}")
        ext = os.path.splitext(vp)[1].lower()
        fname = os.path.basename(vp)
        if not month:
            dest_dir = no_date_dir
            actions.append({
                "file": fname, "from": vp, "to_dir": dest_dir,
                "month": "", "gps": "", "country": "", "note": "no_date",
                "original_file": fname, "edited": False,
                "final_path": vp, "is_video": True, "ext": ext
            })
            print(f"✔ {fname} -> no_date/ (no date)")
            continue
        
        if latlon:
            lat_r, lon_r = round(latlon[0], 3), round(latlon[1], 3)
            coord_set.add((lat_r, lon_r))
            actions.append({
                "file": fname, "from": vp, "to_dir": f"{month}/country_?",
                "month": month, "gps": True, "lat": lat_r, "lon": lon_r,
                "original_file": fname, "edited": False,
                "final_path": vp, "is_video": True, "ext": ext
            })
        else:
            actions.append({
                "file": fname, "from": vp, "to_dir": month,
                "month": month, "gps": False,
                "original_file": fname, "edited": False,
                "final_path": vp, "is_video": True, "ext": ext
            })

    # --- Fallback: files ExifTool missed (often Unicode / Chinese filenames) ---
    actual_files = []
    for fn in os.listdir(session_dir):
        p = os.path.join(session_dir, fn)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
            actual_files.append(os.path.abspath(p))

    # what ExifTool actually returned
    processed_paths = set(meta_by_path.keys())

    for p in actual_files:
        if p not in processed_paths:
            fname = os.path.basename(p)
            ext = os.path.splitext(p)[1].lower()
            actions.append({
                "file": fname,
                "from": p,
                "to_dir": no_date_dir,
                "month": "",
                "gps": "",
                "note": "exiftool_miss_unicode",
                "original_file": fname,
                "edited": False,
                "final_path": p,
                "is_video": ext in VIDEO_EXTS,
                "ext": ext,
            })
            print(f"✔ {fname} -> no_date/ (ExifTool missed it, likely unicode filename)")

    # 5) Batch reverse geocode unique coords -> countries
    coord_to_country = coords_to_country_batch(list(coord_set))

    # 6) Plan final destinations and include companion video moves
    final_actions = []
    for a in actions:
        month = a.get("month","")
        if not month:
            dest_dir = os.path.join(LIBRARY_ROOT, "no_date")
        else:
            if a.get("gps"):
                key = (a["lat"], a["lon"])
                country = coord_to_country.get(key, "Unknown")
                dest_dir = os.path.join(LIBRARY_ROOT, month, f"country_{country}")
                a["country"] = country
            else:
                dest_dir = os.path.join(LIBRARY_ROOT, month)
        
        if a.get("is_video") and a.get("ext") == ".mp4":
            dest_dir = os.path.join(dest_dir, "video")

        # 🔒 Path sandbox: ensure we never write outside the library
        assert_inside_library(dest_dir)

        a["dest_dir"] = dest_dir
        final_actions.append(a)

    # Rebuild by_basename with final paths
    by_basename = {}
    for a in final_actions:
        final_p = a.get("final_path", a["from"])
        if final_p and os.path.exists(final_p):
            by_basename[a["file"]] = final_p

    print(f"DEBUG: by_basename has {len(by_basename)} entries, sample keys: {list(by_basename.keys())[:3]}")

    # Single copy+verify loop (handles everything)
    print(f"DEBUG: Starting copy loop with {len(final_actions)} actions")
    moves = []
    for a in final_actions:
        fname = a["file"]
        src = by_basename.get(fname)
        print(f"DEBUG: For {fname}, src path = {src} (exists? {os.path.exists(src) if src else 'N/A'})")
        if not src or not os.path.exists(src):
            continue

        src_abs = os.path.abspath(src)

        # Crash-safe resume: skip files that were already verified/duplicate in a previous run
        prev_status = processed_status.get(src_abs)
        if prev_status in ("verified", "duplicate", "duplicate_skipped", "duplicate_copied"):
            print(f"RESUME: skipping already-processed file {fname} ({prev_status})")
            continue

        # DRY-RUN: record the plan and continue
        if DRY_RUN:
            rec = {
                "file": fname, "from": src,
                "dest_dir": a["dest_dir"], "month": a.get("month",""),
                "gps": bool(a.get("gps")), "note": a.get("note",""),
                "status": "planned", "ts": datetime.now().isoformat()
            }
            manifest_append(jsonl_path, rec)
            meta["planned"] = meta.get("planned", 0) + 1
            continue

        # Compute source hash once (used for duplicate detection & verification)
        h_src = file_quick_hash(src)

        # Duplicate check against whole library (including previous runs)
        if lib_contains_hash(LIBRARY_ROOT, h_src):
            if DUPLICATE_POLICY == "skip":
                # Mark as duplicate and skip copying
                rec = {
                    "file": fname,
                    "from": src,
                    "hash": h_src,
                    "status": "duplicate_skipped",
                    "ts": datetime.now().isoformat()
                }
                manifest_append(jsonl_path, rec)
                meta["duplicate"] = meta.get("duplicate", 0) + 1
                print(f"(duplicate-skip) {fname} already in library; skipping copy.")
                continue
            elif DUPLICATE_POLICY == "copy_rename":
                # We will still copy this file (with a unique name), but remember it's a duplicate
                a["is_duplicate"] = True
                print(f"(duplicate-copy) {fname} already in library; copying as new file.")
            else:
                # Fallback: treat unknown policy as "skip" for safety
                rec = {
                    "file": fname,
                    "from": src,
                    "hash": h_src,
                    "status": "duplicate_skipped",
                    "ts": datetime.now().isoformat()
                }
                manifest_append(jsonl_path, rec)
                meta["duplicate"] = meta.get("duplicate", 0) + 1
                print(f"(duplicate-skip/fallback) {fname} already in library; skipping copy.")
                continue

        # If edited, copy original first (your AAE path logic stayed above)
        original_from = a["from"]
        if a.get("edited") and os.path.exists(original_from) and original_from != src:
            _ = safe_copy_atomic(original_from, a["dest_dir"])
            print(f"  ↳ also copied original: {os.path.basename(original_from)}")

        # Copy final file
        dest_final = safe_copy_atomic(src, a["dest_dir"])
        a["to"] = dest_final
        a["from"] = src
        moves.append((src, dest_final))

        # Verify hash
        h_dst = file_quick_hash(dest_final)
        if h_dst != h_src:
            rec = {
                "file": fname, "from": src, "to": dest_final,
                "hash_src": h_src, "hash_dst": h_dst,
                "status": "error", "ts": datetime.now().isoformat()
            }
            manifest_append(jsonl_path, rec)
            meta["error"] = meta.get("error", 0) + 1
            print(f"(error) hash mismatch: {fname}")
            continue

        # Update library hash index to include this newly copied file
        LIB_HASH_INDEX.setdefault(h_src, []).append(dest_final)

        # Copy sidecar (.json) if present
        sidecar = src + ".json"
        if os.path.exists(sidecar):
            _ = safe_copy_atomic(sidecar, a["dest_dir"])
            print(f"  ↳ copied sidecar: {os.path.basename(sidecar)}")

        # Record success in manifest
        rec = {
            "file": fname,
            "from": src,
            "to": dest_final,
            "hash": h_src,
            "status": "verified",
            "is_duplicate": bool(a.get("is_duplicate")),
            "ts": datetime.now().isoformat(),
        }
        manifest_append(jsonl_path, rec)
        meta["verified"] = meta.get("verified", 0) + 1

        # Print success line (same UX as before)
        tail = a['dest_dir'].replace(LIBRARY_ROOT + os.sep, '')
        tag = '(GPS)' if a.get('gps') else '(date only)' if a.get('month') else '(no date)'
        print(f"✔ {fname} -> {tail} {tag}")

    # (Delete the second moves=[] loop and its for loop entirely—no more duplicates)

    # Update manifest meta and maybe auto-empty
    # (save BEFORE any deletion to avoid ENOENT)
    if not DRY_RUN:
        total = len(final_actions)
        if meta.get("planned", 0) == 0:
            meta["planned"] = total
    
        ok = (meta.get("verified", 0) + meta.get("duplicate", 0)) >= total and meta.get("error", 0) == 0
        if AUTO_EMPTY and ok:
            meta["phase"] = "done"
            try:
                manifest_save_meta(meta_path, meta)        # save final state first
            except Exception:
                pass
            try:
                if send2trash:
                    send2trash(session_dir)
                else:
                    trash_dir = os.path.join(INCOMING_ROOT, "._trash")
                    ensure_dir(trash_dir)
                    target = os.path.join(trash_dir, os.path.basename(session_dir))
                    target = unique_path(target)
                    shutil.move(session_dir, target)
                print(f"🧹 Auto-emptied staging session: {session_dir}")
            except Exception as e:
                print(f"(warn) failed to auto-empty staging: {e}")
                meta["phase"] = "attention_required"
                try: manifest_save_meta(meta_path, meta)
                except Exception: pass
        else:
            meta["phase"] = "attention_required" if meta.get("error", 0) else "staged"
            try: manifest_save_meta(meta_path, meta)
            except Exception: pass
    else:
        # dry-run just saves 'planned'
        try: manifest_save_meta(meta_path, meta)
        except Exception: pass

    # 7) Log CSV
    if final_actions:
        fields = sorted({k for a in final_actions for k in a.keys()})
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for a in final_actions:
                w.writerow({k: a.get(k, "") for k in fields})

    print(f"\nDone. Log: {LOG_CSV}\n")

if __name__ == "__main__":
    main()
