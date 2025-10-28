import sys, tempfile, os, re, csv, shutil, subprocess, json
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

# Create a "photos_to_organize" folder next to the script if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)
print(f"📁 Base folder: {BASE_FOLDER}")
print("👉 Place your photos and videos inside this folder before running the script.")

LOG_CSV = os.path.join(BASE_FOLDER, "organize_fast_log.csv")

# --- File types we care about ---
IMAGE_EXTS = {".jpg",".jpeg",".png",".heic",".webp",".tif",".tiff"}
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

# ---------- Moves a file safely to a new folder ----------
def safe_move(src, dst_dir):
    ensure_dir(dst_dir)
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
    ]

    results = []
    CHUNK = 600  # reduce if you still hit the limit
    total = len(files)
    for i in range(0, total, CHUNK):
        chunk = files[i:i+CHUNK]

        # write paths to a temporary argfile (one per line)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tf:
            for p in chunk:
                tf.write(p + "\n")
            argfile = tf.name

        try:
            args = base_args + ["-@", argfile]   # tell exiftool to read the file list
            print(f"  → ExifTool: {i+1}-{min(i+CHUNK, total)} / {total}")
            res = subprocess.run(args, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  (warn) exiftool chunk failed: {res.stderr.strip()[:200]}")
                continue
            results.extend(json.loads(res.stdout))

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

    return None


DATE_PAT = re.compile(r"(\d{4})[:\-\/.](\d{2})[:\-\/.](\d{2})")

# ---------- Extract YYYY-MM from metadata dict ----------
def extract_month(meta, abs_path=None):
    # Prefer real capture/create timestamps; ignore File:* and generic ModifyDate
    preferred_keys = [
        "EXIF:DateTimeOriginal", "DateTimeOriginal",
        "QuickTime:CreateDate", "CreateDate", "XMP:CreateDate",
        "MediaCreateDate", "TrackCreateDate", "Composite:SubSecDateTimeOriginal"
    ]
    for k in preferred_keys:
        v = meta.get(k)
        if isinstance(v, str):
            m = DATE_PAT.search(v)
            if m:
                return f"{m.group(1)}-{m.group(2)}"

    # Fallbacks: filename pattern (IMG_20240926_...), or Google Takeout sidecar
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
            except:
                pass
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
    """Parse AAE and apply basic crop to image; save as _edited.jpg"""
    if not aae_path or not os.path.exists(aae_path):
        return image_path  # No AAE? Return original
    
    # Parse AAE (fixed for standard ElementTree)
    tree = ET.parse(aae_path)
    root = tree.getroot()

    # Plist structure: <plist><dict><key>...</key><data>...</data>...</dict></plist>
    dict_elem = root.find('dict')  # Safer: find the <dict> child
    if dict_elem is None:
        return image_path

    adjustment_data_b64 = None
    children = list(dict_elem)  # List of child elements
    for i, child in enumerate(children):
        if child.tag == 'key' and child.text == 'adjustmentData':
            if i + 1 < len(children) and children[i + 1].tag == 'data':
                adjustment_data_b64 = children[i + 1].text.strip()
                break

    if not adjustment_data_b64:
        return image_path
    
    try:
        decoded = base64.b64decode(adjustment_data_b64)
        decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)
        adjustments = json.loads(decompressed)
        
        # Find crop adjustment
        crop_adjust = next((adj for adj in adjustments.get('adjustments', []) 
                            if adj.get('identifier') == 'Crop' and adj.get('enabled')), None)
        if not crop_adjust:
            return image_path  # No crop? Skip
        
        settings = crop_adjust['settings']
        img = Image.open(image_path)
        
        # Apply crop (adjust coords if needed for orientation)
        box = (settings['xOrigin'], settings['yOrigin'], 
               settings['xOrigin'] + settings['width'], 
               settings['yOrigin'] + settings['height'])
        cropped = img.crop(box)
        
        # Save edited version
        base, ext = os.path.splitext(image_path)
        edited_path = f"{base}_edited.jpg"
        cropped.convert('RGB').save(edited_path, 'JPEG', quality=95)
        
        print(f"  → Applied crop to {os.path.basename(image_path)} → {os.path.basename(edited_path)}")
        return edited_path  # Return edited path for moving
    except Exception as e:
        print(f"  (warn) Failed to apply AAE to {os.path.basename(image_path)}: {e}")
        return image_path

def stem_for_pair_from_base(base_name):
    """Like stem_for_pair but takes basename without ext."""
    m = re.match(r"^(IMG_)E(\d+)$", base_name, re.IGNORECASE)
    if m:
        return {base_name, f"{m.group(1)}{m.group(2)}"}
    return {base_name}

# ---------- Main ----------
def main():

    print("\n=== FAST ORGANIZE: one-pass exiftool + batched geocode ===\n")
    
    # Check for files in BASE_FOLDER
    if not any(os.scandir(BASE_FOLDER)):
        print("\n⚠️  No photos found in 'photos_to_organize' folder.")
        print("Please place your images or videos there and run the script again.")
        input("Press Enter to exit...")
        sys,exit(0)

    no_date_dir = os.path.join(BASE_FOLDER, "no_date")
    ensure_dir(no_date_dir)

    # 1) One exiftool pass for all files (images + videos)
    print("Scanning metadata with ExifTool... (one pass)")
    all_meta = run_exiftool_tree(BASE_FOLDER)

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

    # 5) Batch reverse geocode unique coords -> countries
    coord_to_country = coords_to_country_batch(list(coord_set))

    # 6) Plan final destinations and include companion video moves
    final_actions = []
    for a in actions:
        month = a.get("month","")
        if not month:
            dest_dir = os.path.join(BASE_FOLDER, "no_date")
        else:
            if a.get("gps"):
                key = (a["lat"], a["lon"])
                country = coord_to_country.get(key, "Unknown")
                dest_dir = os.path.join(BASE_FOLDER, month, f"country_{country}")
                a["country"] = country
            else:
                dest_dir = os.path.join(BASE_FOLDER, month)
        
        if a.get("is_video") and a.get("ext") == ".mp4":
            dest_dir = os.path.join(dest_dir, "video")

        a["dest_dir"] = dest_dir
        final_actions.append(a)

    # Rebuild by_basename with final paths
    by_basename = {}
    for a in final_actions:
        final_p = a.get("final_path", a["from"])
        if final_p and os.path.exists(final_p):
            by_basename[a["file"]] = final_p

    print(f"DEBUG: by_basename has {len(by_basename)} entries, sample keys: {list(by_basename.keys())[:3]}")

    # Single move loop (handles everything)
    print(f"DEBUG: Starting move loop with {len(final_actions)} actions")
    moves = []
    for a in final_actions:
        fname = a["file"]
        src = by_basename.get(fname)
        print(f"DEBUG: For {fname}, src path = {src} (exists? {os.path.exists(src) if src else 'N/A'})")
        if not src or not os.path.exists(src):
            continue
        
        # If edited, move original FIRST (before overwriting a["from"])
        original_from = a["from"]  # Save true original path
        if a.get("edited") and os.path.exists(original_from) and original_from != src:
            original_new = safe_move(original_from, a["dest_dir"])
            print(f"  ↳ also moved original: {os.path.basename(original_from)}")

        # Now move the final file (edited or original)
        new_path = safe_move(src, a["dest_dir"])
        a["to"] = new_path
        a["from"] = src  # Now safe to overwrite for log
        moves.append((src, new_path))

        # Move companion video(s) if GPS (original stem)
        #if a.get("gps") and not a.get("is_video"):
        #    original_stem_base = os.path.splitext(a['original_file'])[0]
        #    for s in stem_for_pair_from_base(original_stem_base):
        #        for vp in stem_to_videos.get(s.lower(), []):
        #            if os.path.exists(vp):
        #                ext = os.path.splitext(vp)[1].lower()
        #                target_dir = os.path.join(a["dest_dir"], "video") if ext == ".mp4" else a["dest_dir"]
        #                vp_new = safe_move(vp, target_dir)
        #                print(f"  ↳ also moved {'MP4' if ext=='.mp4' else 'MOV'}: {os.path.basename(vp)}")

        # Print success for all
        print(f"✔ {fname} -> {a['dest_dir'].replace(BASE_FOLDER + os.sep,'')} {'(GPS)' if a.get('gps') else '(date only)' if a.get('month') else '(no date)'}")

    # (Delete the second moves=[] loop and its for loop entirely—no more duplicates)

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
