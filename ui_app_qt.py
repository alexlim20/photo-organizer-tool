# ui_app_qt.py
import os, sys
import main as organizer
import hashlib, json
import sqlite3
from typing import List, Dict, Tuple
import threading
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "error")
import cv2
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import (
    QDir, Qt, QSize, QObject, QThread, 
    Signal, QTimer, QEvent, QStandardPaths,
    QRunnable, QThreadPool, QItemSelectionModel, QModelIndex,
    QMutex, QMutexLocker,
    QUrl,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QListView,
    QAbstractItemView, QStatusBar, QStyleFactory, QMessageBox,
    QFileSystemModel, QListView, QSplitterHandle, QFrame, QFileIconProvider,
    QStackedWidget,
)
from PySide6.QtGui import (
    QKeySequence, QImageReader, QPixmap, 
    QIcon, QImage, QShortcut
)
from collections import OrderedDict

from PIL import Image
from pillow_heif import register_heif_opener # enable HEIC/HEIF reading in Pillow

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None

register_heif_opener() 

DEBUG_THUMBS = True

def dprint(*args):
    if DEBUG_THUMBS:
        try:
            print(*args, flush=True)
        except Exception:
            pass

IMAGE_EXTS = [
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tif", "*.tiff", "*.heic", "*.heif", "*.webp"
]
VIDEO_EXTS = [
    "*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"
]

# ============================================================================
# 1. GPS Cache Database (persistent, fast lookups)
# ============================================================================

class GPSCache:
    """
    Optimized SQLite-backed cache.
    """
    def __init__(self, library_path: str):
        cache_dir = os.path.join(library_path, "._cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "gps_index.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        # WAL mode allows faster concurrent reads/writes
        conn.execute("PRAGMA journal_mode=WAL;") 
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gps_points (
                filepath TEXT PRIMARY KEY,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                country TEXT,
                mtime INTEGER,
                last_checked INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_latlon ON gps_points(lat, lon)")
        conn.commit()
        conn.close()

    def get_known_mtimes(self) -> Dict[str, int]:
        """
        Returns a dict of {filepath: mtime} for ALL files currently in DB.
        Used for instant 'is this file changed?' checks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT filepath, mtime FROM gps_points")
        data = {row[0]: row[1] for row in cursor}
        conn.close()
        return data

    def get_all_points(self, limit: int = 100000) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        # Only fetch valid coordinates
        cursor = conn.execute(
            "SELECT lat, lon, country, filepath FROM gps_points WHERE lat != 0 AND lon != 0 LIMIT ?", 
            (limit,)
        )
        points = []
        for row in cursor:
            lat, lon, country, filepath = row
            # Fallback label logic
            label = country if country else "Photo"
            points.append({
                "lat": lat, "lon": lon, 
                "label": label, "country": country
            })
        conn.close()
        return points
    
    def get_clustered_points(self, zoom_level: int = 5) -> List[Dict]:
        # Reuse your existing clustering logic here if you wish, 
        # or just use get_all_points if using the CesiumJS client-side clustering.
        return []

    def save_batch(self, records: List[Tuple]):
        """
        Bulk insert/update to minimize transaction overhead.
        records: list of (filepath, lat, lon, country, mtime, now_ts)
        """
        if not records: return
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO gps_points
                (filepath, lat, lon, country, mtime, last_checked)
                VALUES (?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        except Exception as e:
            print(f"[GPS Cache] Save batch error: {e}")
        finally:
            conn.close()

# ============================================================================
# 2. Background GPS Indexer (non-blocking)
# ============================================================================

class GPSIndexerThread(QThread):
    """
    High-Performance Background Indexer
    Uses 'exiftool -@ argfile' to scan thousands of files in a single process.
    """
    progress = Signal(int)            # processed count
    intermediate_points = Signal(list) # batches for live UI update
    finished_with_count = Signal(int)
    error = Signal(str)

    def __init__(self, library_path: str, parent=None):
        super().__init__(parent)
        self.library_path = library_path
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        import subprocess
        import tempfile
        import time
        
        try:
            cache = GPSCache(self.library_path)
            
            # 1. LOAD STATE: Get all existing mtimes from DB (Instant)
            print("[GPS Indexer] Loading database state...")
            raw_known = cache.get_known_mtimes()
            known_files = {k.lower(): v for k, v in raw_known.items()}

            print(f"[DEBUG] Database has {len(known_files)} entries")
            if len(known_files) > 0:
                sample_key = list(known_files.keys())[0]
                print(f"[DEBUG] Sample DB key: '{sample_key}'")
            
            # 2. FAST SCAN: Walk filesystem to find changed/new files
            print("[GPS Indexer] Scanning directory structure...")
            to_scan = []       # Files that need ExifTool analysis
            files_checked = 0
            
            img_exts = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff", ".dng", ".cr2", ".arw"}
            vid_exts = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
            valid_exts = img_exts | vid_exts

            # Pre-compiled regex for speed if needed, but endswith is fast enough
            
            for root, dirs, files in os.walk(self.library_path):
                if self._stop_flag: break
                
                # Skip internal folders
                if "_incoming" in root or "._" in root:
                    continue
                
                # Try to guess country from folder path (fast fallback)
                country_guess = None
                parts = root.split(os.sep)
                for p in parts:
                    if p.startswith("country_"):
                        country_guess = p.replace("country_", "")
                        break
                
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in valid_exts:
                        continue
                        
                    filepath_raw = os.path.join(root, fname)
                    # 2. Create a "normalized" path (forward slashes) to check against the DB
                    filepath_normalized = filepath_raw.replace("\\", "/")
                    try:
                        mtime = int(os.path.getmtime(filepath_raw))
                    except OSError:
                        continue
                    
                    # 3. Check the DB using the NORMALIZED path
                    if filepath_normalized.lower() in known_files and known_files[filepath_normalized.lower()] == mtime:
                        continue # Skip!

                    # 4. If we must scan, use the raw path
                    to_scan.append((filepath_normalized, mtime, country_guess))
                    
                files_checked += len(files)
                if files_checked % 5000 == 0:
                    print(f"[GPS Indexer] Walked {files_checked} files...")

            total_new = len(to_scan)
            print(f"[GPS Indexer] Analysis done. {total_new} files need GPS extraction.")
            
            if total_new == 0:
                self.finished_with_count.emit(0)
                return

            # 3. BATCH PROCESSING: Send files to ExifTool in chunks of 1000
            # This prevents "Command line too long" errors and gives UI updates
            CHUNK_SIZE = 100
            processed_count = 0
            
            # Find ExifTool path (borrowed from main.py logic)
            exif_exe = "exiftool" # Assume in PATH by default
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_exif = os.path.join(script_dir, "exiftool.exe")
            if os.path.exists(local_exif):
                exif_exe = local_exif

            for i in range(0, total_new, CHUNK_SIZE):
                if self._stop_flag: break
                
                chunk = to_scan[i : i + CHUNK_SIZE]
                
                # Create a temporary file listing all paths for this chunk
                # This is the "-@" argument feature of ExifTool
                with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as arg_file:
                    for item in chunk:
                        path = item[0]
                        # Normalize path separators
                        arg_file.write(path.replace("\\", "/") + "\n")
                    arg_file_path = arg_file.name

                try:
                    # Run ExifTool ONCE for 1000 files
                    # -fast: avoid scanning end of JPEGs 
                    # -n: numeric output (easier parsing)
                    # -p: custom output format allows parsing without heavy JSON overhead
                    # requesting specific GPS tags only
                    cmd = [
                        exif_exe,
                        "-@", arg_file_path,
                        "-n", 
                        "-fast", 
                        "-j",  # JSON output is safest for special characters
                        "-GPSLatitude", "-GPSLongitude", 
                        "-GPSPosition", "-Composite:GPSPosition",
                        "-ignoreMinorErrors"
                    ]
                    
                    # On Windows, suppress console window
                    startupinfo = None
                    if os.name == 'nt':
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    
                    proc = subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.DEVNULL,
                        text=True, 
                        encoding='utf-8',
                        errors='replace',
                        startupinfo=startupinfo
                    )
                    
                    if proc.stdout:
                        data = json.loads(proc.stdout)
                        
                        # Map source file back to our queue data (mtime, country)
                        # We use a dictionary for O(1) lookup of the chunk data
                        # Key: normalized absolute path
                        chunk_map = {}
                        for item in chunk:
                            # item[0] is the raw path from os.walk
                            # Force E:/photos/file.jpg format
                            clean_key = item[0].replace("\\", "/").lower()
                            chunk_map[clean_key] = (item[1], item[2])
                        
                        db_batch = []
                        ui_batch = []
                        now_ts = int(time.time())
                        
                        for record in data:
                            src = record.get("SourceFile", "")
                            if not src: continue
                            
                            # 2. Convert ExifTool path to match our clean key
                            # ExifTool usually gives E:/Photos/File.jpg
                            src_key = src.replace("\\", "/").lower()
                            
                            meta_data = chunk_map.get(src_key)
                            
                            # SAFETY NET: If match fails, try one fallback, then PRINT THE ERROR
                            if not meta_data:
                                # Sometimes Exiftool resolves symlinks, so try the raw src just in case
                                meta_data = chunk_map.get(src.lower())
                                
                            if not meta_data:
                                # THIS IS THE DEBUG LINE THAT WAS MISSING
                                # It will tell us if we are still losing data
                                print(f"[CRITICAL FAILURE] Could not match ExifTool file: {src}")
                                print(f"   -> We looked for key: {src_key}")
                                # print(f"   -> Available keys: {list(chunk_map.keys())[0]}...") # Uncomment if needed
                                continue
                                
                            mtime, country_fallback = meta_data
                            
                            # Extract Lat/Lon
                            lat = record.get("GPSLatitude")
                            lon = record.get("GPSLongitude")
                            
                            # Handle Composite string format "lat lon" if individual tags fail
                            if lat is None and "Composite:GPSPosition" in record:
                                try:
                                    parts = record["Composite:GPSPosition"].split()
                                    if len(parts) >= 2:
                                        lat = float(parts[0])
                                        lon = float(parts[1])
                                except: pass
                                
                            # Final validation
                            try:
                                lat = float(lat)
                                lon = float(lon)
                            except (TypeError, ValueError):
                                lat = 0.0
                                lon = 0.0
                            db_batch.append((src_key, lat, lon, country_fallback, mtime, now_ts))
                            
                            base_name = os.path.basename(src)
                            label = f"{country_fallback} Â· {base_name}" if country_fallback else base_name
                            ui_batch.append({
                                "lat": lat, "lon": lon, 
                                "label": label, "country": country_fallback
                            })

                        # Batch save to DB
                        cache.save_batch(db_batch)
                        
                        # Emit to UI
                        if ui_batch:
                            self.intermediate_points.emit(ui_batch)

                    processed_count += len(chunk)
                    self.progress.emit(processed_count)
                    
                except Exception as e:
                    print(f"[GPS Indexer] Batch Error: {e}")
                finally:
                    # Cleanup temp file
                    if os.path.exists(arg_file_path):
                        os.unlink(arg_file_path)

            self.finished_with_count.emit(processed_count)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class ThumbnailFileSystemModel(QFileSystemModel):
    def __init__(self, thumb_size=QSize(256, 256), parent=None):
        super().__init__(parent)
        self._thumb_size = thumb_size
        self._cache = {}  # key: (path, mtime) -> QIcon
        self._lock = QMutex()
        # Supported extensions
        self._img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"}
        self._vid_exts = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

    def data(self, index, role=Qt.DecorationRole):
        if role != Qt.DecorationRole or not index.isValid():
            return super().data(index, role)

        path = self.filePath(index)
        if not path:
            return super().data(index, role)

        # Only files
        if self.isDir(index):
            return super().data(index, role)

        ext = os.path.splitext(path)[1].lower()
        try:
            stat = os.stat(path)
            key = (path, stat.st_mtime)
        except Exception:
            key = (path, 0)

        with QMutexLocker(self._lock):
            ic = self._cache.get(key)
        if ic:
            return ic

        pix = None
        # Try image first
        if ext in self._img_exts:
            pix = self._read_image_thumb(path)
        # Try video (first frame) if not an image
        if pix is None and ext in self._vid_exts:
            pix = self._read_video_thumb(path)

        if pix is None:
            # Fallback to default icon
            return super().data(index, role)

        ic = QIcon(pix)
        with QMutexLocker(self._lock):
            self._cache[key] = ic
        return ic

    def _read_image_thumb(self, path):
        try:
            reader = QImageReader(path)
            # Ask Qt to scale during decode (faster & low memory)
            s = self._thumb_size
            reader.setAutoTransform(True)
            reader.setScaledSize(s)
            img = reader.read()
            if img.isNull():
                return None
            return QPixmap.fromImage(img)
        except Exception:
            return None

    def _read_video_thumb(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            # Convert BGR->RGB and scale
            h, w = frame.shape[:2]
            target_w, target_h = self._thumb_size.width(), self._thumb_size.height()
            scale = min(target_w / max(1, w), target_h / max(1, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            qimg = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None
        
class GlobeWorker(QThread):
    finished_points = Signal(list)
    error = Signal(str)

    def __init__(self, folder: str, parent=None):
        super().__init__(parent)
        self.folder = folder

    def run(self):
        try:
            points = collect_gps_points(self.folder)
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.finished_points.emit(points)

class WorkerThread(QThread):
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, folder: str, dup_policy: str = "skip", parent=None):
        super().__init__(parent)
        self.folder = folder
        self.dup_policy = dup_policy  # "skip" or "copy_rename"

    def run(self):
        try:
            # Configure backend globals before running
            if hasattr(organizer, "BASE_FOLDER"):
                organizer.BASE_FOLDER = self.folder
                organizer.DRY_RUN = False
            if hasattr(organizer, "LOG_CSV"):
                import os
                organizer.LOG_CSV = os.path.join(self.folder, "organize_fast_log.csv")
                organizer.AUTO_EMPTY = True

            if hasattr(organizer, "DUPLICATE_POLICY"):
                organizer.DUPLICATE_POLICY = self.dup_policy

            organizer.main()  # runs in this QThread
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))

class LockedSplitterHandle(QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setCursor(Qt.ArrowCursor)  # never show resize cursor

    # Block dragging entirely
    def mousePressEvent(self, e):  # no-op
        return
    def mouseMoveEvent(self, e):   # no-op
        return
    def mouseReleaseEvent(self, e):# no-op
        return

class LockedSplitter(QSplitter):
    def createHandle(self):
        return LockedSplitterHandle(self.orientation(), self)
    
class ThumbSignals(QObject):
    ready = Signal(str, int)  # emits absolute file path when a thumb is ready

class ThumbJob(QRunnable):
    def __init__(self, path, size, kind, build_image_fn, build_video_fn, done_cb, gen):
        super().__init__()
        self.path = path
        self.size = size
        self.kind = kind          # "image" or "video"
        self.build_image_fn = build_image_fn
        self.build_video_fn = build_video_fn
        self.done_cb = done_cb    # callback(path) on UI thread
        self.gen = gen

    def run(self):
        ok = False
        try:
            if self.kind == "image":
                ok = self.build_image_fn(self.path, self.size, self.gen)   # <-- pass gen
            elif self.kind == "video":
                ok = self.build_video_fn(self.path, self.size, self.gen)   # <-- pass gen
        except Exception:
            ok = False
        finally:
            self.done_cb(self.path, self.gen)

class ThumbnailIconProvider(QFileIconProvider):
    """
    Returns thumbnails for image files; falls back to default icons for others.
    Caches a small LRU of thumbnails to avoid re-reading.
    """
    IMAGE_SUFFIXES = {
        "jpg","jpeg","png","gif","bmp","tif","tiff","webp","heic","heif"
    }

    VIDEO_SUFFIXES = {"mp4", "mov", "avi", "mkv", "m4v"}

    @staticmethod
    def _norm(path: str) -> str:
        # on Windows: lowercased + forward slashes so everything matches
        try:
            import os
            return os.path.normcase(path).replace("\\", "/")
        except Exception:
            return path.replace("\\", "/")

    def __init__(self, thumb_size=128, cache_max=256, on_ready=None):
        import threading
        from collections import deque
        super().__init__()
        cache_root = QStandardPaths.writableLocation(QStandardPaths.CacheLocation) or os.path.expanduser("~/.cache")
        self.disk_cache_dir = os.path.join(cache_root, "travel_photo_organizer", "thumbs")
        # (debug) print("ðŸ—‚ï¸  Disk thumbnail cache is at:", self.disk_cache_dir)
        os.makedirs(self.disk_cache_dir, exist_ok=True)

        self.thumb_size = thumb_size
        self.cache = OrderedDict()
        self.cache_max = cache_max
        self._fallback = QFileIconProvider()
        self.signals = ThumbSignals()
        # UI callback to request repaint for a given path
        self._on_ready = on_ready or (lambda _p: None)
        self.signals.ready.connect(self._on_job_done)
        self.threadpool = QThreadPool.globalInstance()
        # coalesce duplicate jobs
        self._inflight = set()
        self._video_gate = threading.Semaphore(1)   # only 1 video decode at a time
        self._image_gate = threading.Semaphore(16)  # Increased to 16 for more concurrent image decodes
        self._max_inflight = 300                   # Increased to 300 to handle larger folders
        self._pending = deque()                    # paths waiting to be scheduled when inflight drops
        self.cache_max = 2048                      # larger LRU so going back is instant
        self.gen = 0
        self.current_root = ""
        self.threadpool.setMaxThreadCount(16)      # Increased to 16 threads to reduce delays

    def _stat_key(self, path: str, size: int) -> str:
        try:
            st = os.stat(path)
            payload = f"{self._norm(path)}|{size}|{st.st_mtime_ns}|{st.st_size}"
        except Exception:
            payload = f"{self._norm(path)}|{size}|0|0"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _disk_thumb_path(self, path: str, size: int) -> str:
        return os.path.join(self.disk_cache_dir, self._stat_key(path, size) + ".png")
    

    def _cache_icon(self, path: str, px: QPixmap):
        ico = QIcon(px)
        n = self._norm(path)
        key = (n, self.thumb_size)
        self.cache[key] = ico
        self._inflight.discard(n)
        # NEW: write to disk cache (best-effort)
        try:
            disk_p = self._disk_thumb_path(path, self.thumb_size)
            # PNG is fine for tiny thumbs; itâ€™s fast & supports alpha
            px.save(disk_p, "PNG")
        except Exception:
            pass
        if len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)
        return True

    def _build_image_sync(self, path: str, size: int, expected_gen: int):
        # gen changed â†’ stop
        if expected_gen != self.gen:
            return False
        self._image_gate.acquire()

        # Fast HEIC/HEIF preview path
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".heic", ".heif"):
            try:
                from pillow_heif import open_heif
                heif = open_heif(path)
                # Try embedded thumbnail first
                if heif.has_thumbnails:
                    thumb = heif.thumbnails[0].to_pillow()
                    thumb.thumbnail((size, size))
                    qimg = QImage(thumb.tobytes("raw", "RGBA"), thumb.width, thumb.height, QImage.Format_RGBA8888)
                    return self._cache_icon(path, QPixmap.fromImage(qimg))
                # Else decode primary, but ask Pillow-HEIF to scale down early when possible
                pil_img = heif.to_pillow()  # already orientation-aware
                pil_img.thumbnail((size, size))
                pil_img = pil_img.convert("RGBA")
                qimg = QImage(pil_img.tobytes("raw", "RGBA"), pil_img.width, pil_img.height, QImage.Format_RGBA8888)
                return self._cache_icon(path, QPixmap.fromImage(qimg))
            except Exception:
                pass  # fall back to your existing reader/Pillow path
        
        try:
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            if reader.size().isValid():
                w, h = reader.size().width(), reader.size().height()
                if w > h:
                    reader.setScaledSize(QSize(size, max(1, int(size * h / w))))
                else:
                    reader.setScaledSize(QSize(max(1, int(size * w / h)), size))

            # check again before decode
            if expected_gen != self.gen:
                return False

            img = reader.read()
            if img.isNull():
                # Pillow fallback for HEIC/HEIF (and any formats Qt can't read)
                try:
                    if expected_gen != self.gen:
                        return False
                    pil_img = Image.open(path)
                    pil_img.thumbnail((size, size), resample=Image.NEAREST)  # UPDATED: Use Image.NEAREST for even faster downsampling (trade quality for speed during rapid navigation)
                    pil_img = pil_img.convert("RGBA")
                    if expected_gen != self.gen:
                        return False
                    qimg = QImage(
                        pil_img.tobytes("raw", "RGBA"),
                        pil_img.width, pil_img.height,
                        QImage.Format_RGBA8888,
                    )
                    return self._cache_icon(path, QPixmap.fromImage(qimg))
                except Exception:
                    return False
            else:
                # âœ… Qt read succeeded â€” cache it
                if expected_gen != self.gen:
                    return False
                return self._cache_icon(path, QPixmap.fromImage(img))
        finally:
            self._image_gate.release()

    def _build_video_sync(self, path: str, size: int, expected_gen: int):
        if expected_gen != self.gen:
            return False

        self._video_gate.acquire()
        try:
            if expected_gen != self.gen:
                return False

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return False

            # derive positions (fractions of duration)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
            dur_ms = (1000.0 * total_frames / fps) if total_frames > 0 else 0.0

            # try a few spots to avoid black/intro frames
            fractions = [0.10, 0.35, 0.60]
            frame = None
            for f in fractions:
                if expected_gen != self.gen:
                    return False
                t_ms = dur_ms * f if dur_ms > 0 else (1000.0 * f)
                cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
                ok, fr = cap.read()
                if not ok or fr is None:
                    continue

                # reject very dark/blank frames
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                mean = float(gray.mean())
                std  = float(gray.std())
                if mean < 8 or std < 4:  # heuristics; tweak as needed
                    continue

                frame = fr
                break

            if frame is None:
                # fallback: first readable frame
                cap.set(cv2.CAP_PROP_POS_MSEC, 500)
                ok, frame = cap.read()

            cap.release()
            if frame is None or expected_gen != self.gen:
                return False

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            scale = float(size) / float(max(h, w))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            frame_small = cv2.resize(frame, (nw, nh))

            if expected_gen != self.gen:
                return False
            qimg = QImage(frame_small.data, nw, nh, 3 * nw, QImage.Format_RGB888)
            return self._cache_icon(path, QPixmap.fromImage(qimg))
        finally:
            self._video_gate.release()

    def icon(self, arg):
        """
        Handle both overloads:
          - icon(QFileIconProvider.IconType)
          - icon(QFileInfo)
        Always return a QIcon.
        """
        # Overload 1: IconType (generic icons). Just delegate to base.
        from PySide6.QtWidgets import QFileIconProvider
        if isinstance(arg, QFileIconProvider.IconType):
            return super().icon(arg)

        # Overload 2: QFileInfo for an actual item
        fileInfo = arg  # expected to be QFileInfo
        try:
            # Folders: use default folder icon (fast, no thumbnail job)
            if fileInfo.isDir():
                return super().icon(fileInfo)

            # Files: our thumbnail pipeline
            path = self._norm(fileInfo.filePath())
            suffix = fileInfo.suffix().lower()

            # Not a supported media type -> default icon
            if suffix not in self.IMAGE_SUFFIXES and suffix not in self.VIDEO_SUFFIXES:
                return super().icon(fileInfo)

            # Cache hit -> return immediately
            key = (path, self.thumb_size)
            ico = self.cache.get(key)
            if ico is not None:
                return ico
            
            # NEW: disk cache hit?
            disk_p = self._disk_thumb_path(path, self.thumb_size)
            if os.path.exists(disk_p):
                px = QPixmap(disk_p)
                if not px.isNull():
                    ico = QIcon(px)
                    self.cache[key] = ico
                    return ico

            # Schedule (or queue) a job if below cap; otherwise enqueue
            kind = "image" if suffix in self.IMAGE_SUFFIXES else "video"
            if path not in self._inflight:
                if len(self._inflight) >= self._max_inflight:
                    # queue for later and return a generic icon for now
                    self._pending.append((path, kind))
                    return super().icon(fileInfo)

                self._inflight.add(path)
                job = ThumbJob(
                    path, self.thumb_size, kind,
                    self._build_image_sync, self._build_video_sync,
                    done_cb=self.signals.ready.emit,
                    gen=self.gen
                )
                self.threadpool.start(job)

            # Fallback until the job finishes
            return super().icon(fileInfo)
        except Exception:
            # Never let exceptions bubble into Qt's paint path
            return super().icon(fileInfo)
    
    def _on_job_done(self, path: str, gen: int):
        #if gen != self.gen:
            #if DEBUG_THUMBS:  # Optional - only print stale if debug is on; can set DEBUG_THUMBS=False to suppress
            #    dprint(f"[done/STALE] gen={gen}!=curr={self.gen} :: {path}")
        #    return
        if gen != self.gen:
            # stale completion from a previous folder â€” ignore
            return
        n = self._norm(path)
        self._inflight.discard(n)
        self._maybe_start_pending()
        self._on_ready(n)  # emit normalized path to the UI

    def begin_folder(self, root_path: str):
        # Bump generation so old finishes are ignored.
        self.gen += 1
        self.current_root = self._norm(root_path)

        # Drop any queued jobs that haven't started and clear book-keeping.
        self.threadpool.clear()
        self._inflight.clear()
        self._pending.clear()

        # Prune cache to only keep entries under the new root, freeing memory for large navigations
        to_remove = [k for k in self.cache if not k[0].startswith(self.current_root)]
        for k in to_remove:
            del self.cache[k]

    def _maybe_start_pending(self):
        # Start as many pending as the cap allows
        while self._pending and len(self._inflight) < self._max_inflight:
            path, kind = self._pending.popleft()
            if path in self._inflight:
                continue
            self._inflight.add(path)
            job = ThumbJob(
                path, self.thumb_size, kind,
                self._build_image_sync, self._build_video_sync,
                done_cb=self.signals.ready.emit,
                gen=self.gen
            )
            self.threadpool.start(job)
    
    def cancel_all_and_clear(self):
        """Abort current gen and purge any queued work (keep cache)."""
        self.gen += 1                   # stale-guard everything already enqueued
        self.threadpool.clear()         # drop queued runnables that haven't started
        self._inflight.clear()          # reset in-flight set
        self._pending.clear()           # reset pending queue

class FSModel(QFileSystemModel):
    """Return provider-cached thumbs for DecorationRole; otherwise default behavior."""
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole and index.isValid():
            prov = self.iconProvider()
            if isinstance(prov, ThumbnailIconProvider):
                # use the same normalized key the provider uses
                path = prov._norm(self.filePath(index))
                key = (path, prov.thumb_size)
                ico = prov.cache.get(key)
                if ico is not None:
                    return ico
        return super().data(index, role)
    
class GlobeWidget(QWidget):
    """
    Globe widget with improved debugging and error handling
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._points = []
        self._use_clustering = True
        
        if QWebEngineView is None:
            self.info = QLabel("Globe view requires Qt WebEngine.")
            self.info.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.info)
            self.view = None
            
        else:
            self.info = None
            self.view = QWebEngineView(self)

            # NEW: Hook JS errors globally
            self.view.page().javaScriptConsoleMessage = self._js_console_message
            self.view.urlChanged.connect(lambda url: print(f"[Globe] Nav to: {url.toString()}"))
            self.view.setZoomFactor(1.05)

            # ======= NEW: Add map style toggle button =======
            self.map_style_btn = QPushButton("ðŸ—ºï¸ ", self)
            self.map_style_btn.setFixedSize(40, 40)
            self.map_style_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 2px solid #ccc;
                    border-radius: 20px;
                    font-size: 18px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 1.0);
                    border-color: #666;
                }
                QPushButton:pressed {
                    background-color: rgba(230, 230, 230, 1.0);
                }
            """)
            self.map_style_btn.setToolTip("Toggle: Satellite â†” Street Map")
            self.map_style_btn.clicked.connect(self._toggle_map_style)
            self.map_style_btn.raise_()  # Keep button on top
            self._current_style = "satellite"  # Track current style
            # ======= END NEW CODE =======

            layout.addWidget(self.view)

            self._is_loaded = False
            
            # Enable JS console and debugging
            from PySide6.QtWebEngineCore import QWebEngineSettings
            settings = self.view.settings()
            settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.ErrorPageEnabled, True)
            
            # Connect to console messages for debugging
            self.view.page().javaScriptConsoleMessage = self._js_console_message
            self.view.loadFinished.connect(lambda ok: print(f"[Globe] Load done: {'OK' if ok else 'FAIL'}"))

    def _js_console_message(self, level, message, line, source):
        """Capture JavaScript console messages for debugging"""
        print(f"[JS Console] {message} (line {line})")
    
    def set_points(self, points, use_clustering=True):
        self._points = points or []
        print(f"[Globe] set_points: {len(self._points)} pts")
        
        # 1. IF GLOBE IS ALREADY LOADED: Update data via JS (No Camera Reset)
        if self._is_loaded and self.view:
            valid_points = []
            for p in self._points:
                 if isinstance(p.get('lat'), (int, float)) and isinstance(p.get('lon'), (int, float)):
                     valid_points.append({
                         'lat': float(p['lat']),
                         'lon': float(p['lon']),
                         'label': str(p.get('label', 'Photo')),
                         'country': str(p.get('country', ''))
                     })
            
            import json
            json_str = json.dumps(valid_points)
            
            # Call the new JS function that swaps data without reloading viewer
            js_code = """
            (function() {
                if (typeof replaceAllPoints === 'function') {
                    replaceAllPoints(PERCENT_DATA_PERCENT);
                }
            })();
            """.replace("PERCENT_DATA_PERCENT", json_str)
            
            self.view.page().runJavaScript(js_code)
            print("[Globe] Updated points via JS (Camera preserved)")
            return

        # 2. IF GLOBE IS NOT LOADED: Build it from scratch
        if self.view:
            self._render_globe()
    
class GlobeWidget(QWidget):
    """
    Globe widget: Optimized for 25,000+ points without culling or limits.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._points = []
        
        if QWebEngineView is None:
            self.info = QLabel("Globe view requires Qt WebEngine.")
            self.info.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.info)
            self.view = None
        else:
            self.view = QWebEngineView(self)
            
            # Debugging setup
            self.view.page().javaScriptConsoleMessage = self._js_console_message
            self.view.urlChanged.connect(lambda url: print(f"[Globe] Nav to: {url.toString()}"))
            
            # Map Style Button
            self.map_style_btn = QPushButton("ðŸ—ºï¸ ", self)
            self.map_style_btn.setFixedSize(40, 40)
            self.map_style_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 2px solid #ccc;
                    border-radius: 20px;
                    font-size: 18px;
                }
                QPushButton:hover { background-color: white; border-color: #666; }
            """)
            self.map_style_btn.clicked.connect(self._toggle_map_style)
            self._current_style = "satellite"
            
            layout.addWidget(self.view)
            
            # Performance Settings
            settings = self.view.settings()
            settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.ErrorPageEnabled, True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'map_style_btn'):
            self.map_style_btn.move(self.width() - 50, 10)

    def _js_console_message(self, level, message, line, source):
        if "Blocking XFrameOptions" in message: return
        print(f"[Globe JS] {message} (line {line})")

    def _toggle_map_style(self):
        if not self.view: return
        
        script = ""
        if self._current_style == "satellite":
            # --- FIX: Send 'dark' to match the JavaScript ---
            script = "setMapStyle('dark');"  
            self._current_style = "dark"
            self.map_style_btn.setToolTip("Switch to Satellite")
        else:
            script = "setMapStyle('satellite');"
            self._current_style = "satellite"
            self.map_style_btn.setToolTip("Switch to Dark Map")
            
        self.view.page().runJavaScript(script)

    def set_points(self, points, use_clustering=True):
        self._points = points or []
        print(f"[Globe] set_points: {len(self._points)} pts")
        if self.view:
            self._render_globe()

    def add_batch_js(self, new_points: list):
        """Injects new points safely into the running globe"""
        if not new_points or not self.view: return

        valid_batch = []
        for p in new_points:
             if isinstance(p.get('lat'), (int, float)) and isinstance(p.get('lon'), (int, float)):
                 valid_batch.append({
                     'lat': float(p['lat']),
                     'lon': float(p['lon']),
                     'label': str(p.get('label', 'Photo')),
                     'country': str(p.get('country', ''))
                 })
        
        if not valid_batch: return

        import json
        # We use a placeholder replacement approach even here to be safe
        json_str = json.dumps(valid_batch)
        
        js_code = """
        (function() {
            if (typeof addPointsBatch === 'function') {
                addPointsBatch(PERCENT_DATA_PERCENT);
            }
        })();
        """.replace("PERCENT_DATA_PERCENT", json_str)

        self.view.page().runJavaScript(js_code)

    def _render_globe(self):
        """
        Render globe with:
        1. Google-Style Label Collision.
        2. High-Performance State Borders (Primitives).
        3. SMART VISIBILITY: State borders hide from space to fix lag.
        4. Photo Clustering.
        """
        import json
        from PySide6.QtCore import QUrl
    
        self._has_been_rendered = False
        self._is_loaded = False
        
        valid_points = []
        for p in self._points: 
            if isinstance(p.get('lat'), (int, float)) and isinstance(p.get('lon'), (int, float)):
                valid_points.append({
                    'lat': float(p['lat']),
                    'lon': float(p['lon']),
                    'label': str(p.get('label', 'Photo')),
                    'country': str(p.get('country', ''))
                })
    
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        border_file_abs = os.path.join(assets_dir, "optimized_borders.json").replace("\\", "/")
        border_url = f"file:///{border_file_abs}"
        has_local_borders = os.path.exists(os.path.join(assets_dir, "optimized_borders.json"))
        
        detected_name_key = "name"
        if has_local_borders:
            try:
                with open(os.path.join(assets_dir, "optimized_borders.json"), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "features" in data and len(data["features"]) > 0:
                        props = data["features"][0].get("properties", {})
                        candidates = ["name", "NAME", "NAME_1", "admin_name", "woe_name"]
                        for c in candidates:
                            if c in props:
                                detected_name_key = c
                                break
            except: pass

        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        base_url = QUrl.fromLocalFile(os.path.join(assets_dir, ""))
        
        html_template = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap" rel="stylesheet">
      <style>
        html, body {{ 
            margin: 0; padding: 0; width: 100%; height: 100%; 
            overflow: hidden; background: #000; 
            font-family: 'Roboto', sans-serif; 
        }}
        #cesiumContainer {{ width: 100%; height: 100%; display: block; }}
      </style>
      <script src="Cesium/Cesium.js"></script>
      <link href="Cesium/Widgets/widgets.css" rel="stylesheet" />
    </head>
    <body>
      <div id="cesiumContainer"></div>
    
      <script>
      var viewer = null;
      var allPoints = []; 
      
      var userPointCollection = null;
      var userLabelCollection = null;
      
      var updateTimeout = null;    
      var collisionTimeout = null; 
      var labelEntities = [];      
      
      var stateBordersPrimitive = null; 

      var LOCAL_BORDER_URL = "{border_url}";
      var HAS_LOCAL_BORDERS = {str(has_local_borders).lower()};
      var NAME_KEY = "{detected_name_key}";

      (function initViewer() {{
        if (typeof Cesium === 'undefined') return;
    
        try {{
            Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJmOGVmNmIxYy0yYTkwLTRjN2QtYWEyYy1lNWZmNmE2YmY0YzkiLCJpZCI6MzYxMTAzLCJpYXQiOjE3NjMzNzcyMDd9.6v-mLQxQSwC6r2FaxJWPJATP3Jw977W7VTsuS-ue3Dc';
    
            viewer = new Cesium.Viewer('cesiumContainer', {{
              animation: false, timeline: false, baseLayerPicker: false, geocoder: false,
              sceneModePicker: false, navigationHelpButton: false, homeButton: false,
              fullscreenButton: false, infoBox: false, selectionIndicator: false,
              shouldAnimate: false, requestRenderMode: true, maximumRenderTimeChange: Infinity
            }});

            viewer.scene.requestRenderMode = true;
            viewer.scene.maximumRenderTimeChange = 0.001;
            viewer.shadowMap.size = 1024;
            viewer.scene.highDynamicRange = false;
            viewer.scene.fog.enabled = true;
            viewer.scene.fog.density = 0.0001;
            viewer.scene.globe.enableLighting = false;
            viewer.scene.globe.maximumScreenSpaceError = 2;
            
            viewer.resolutionScale = window.devicePixelRatio || 1.0;
            viewer.scene.postProcessStages.fxaa.enabled = false;
            viewer.scene.screenSpaceCameraController.minimumZoomDistance = 1000;
            
            viewer.imageryLayers.removeAll();
            viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({{
                url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
                credit: 'ESRI'
            }}));

            userPointCollection = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
            userLabelCollection = viewer.scene.primitives.add(new Cesium.LabelCollection());
            
            loadHighDetailLabels();
            loadBoundaries();
    
            viewer.camera.moveEnd.addEventListener(function() {{
                if (collisionTimeout) clearTimeout(collisionTimeout);
                collisionTimeout = setTimeout(updateLabelCollisions, 100);
                
                if (updateTimeout) clearTimeout(updateTimeout);
                updateTimeout = setTimeout(updatePointsForZoom, 150);
            }});
            
            viewer.scene.preUpdate.addEventListener(function() {{
                updateBorderVisibility(); 
            }});
            
            setTimeout(updateLabelCollisions, 1000);
            setTimeout(updatePointsForZoom, 1000);

        }} catch(e) {{ console.error(e); }}
      }})();

      var lastBorderState = null;
      function updateBorderVisibility() {{
          if (stateBordersPrimitive) {{
              var height = viewer.camera.positionCartographic.height;
              var shouldShow = (height < 1000000);
              
              if (lastBorderState !== shouldShow) {{
                  lastBorderState = shouldShow;
                  if (shouldShow) {{
                      stateBordersPrimitive.show = true;
                      viewer.scene.requestRender();
                  }} else {{
                      stateBordersPrimitive.show = false;
                  }}
              }}
          }}
      }}

      var lastCameraHeight = 0;
      function updateLabelCollisions() {{
          if (!viewer || labelEntities.length === 0) return;
          
          var height = viewer.camera.positionCartographic.height;
          if (Math.abs(height - lastCameraHeight) / lastCameraHeight < 0.1) {{
              return;
          }}
          lastCameraHeight = height;

          var scene = viewer.scene;
          var camera = scene.camera;
          var ellipsoid = scene.globe.ellipsoid;
          var canvasHeight = scene.canvas.clientHeight;
          var canvasWidth = scene.canvas.clientWidth;

          if (height > 20000000) {{
              labelEntities.forEach(item => {{ item.entity.label.show = false; }});
              viewer.scene.requestRender();
              return;
          }}

          labelEntities.sort(function(a, b) {{ return a.rank - b.rank; }});
          
          var visibleLabels = labelEntities.filter(item => {{
              var distance = Cesium.Cartesian3.distance(
                  viewer.camera.position,
                  Cesium.Cartesian3.fromDegrees(item.lon, item.lat)
              );
              return distance < height * 2; 
          }});

          var placedBoxes = []; 

          for (var i = 0; i < visibleLabels.length; i++) {{
              var item = visibleLabels[i];
              var entity = item.entity;
              
              var position = Cesium.Cartesian3.fromDegrees(item.lon, item.lat);
              var occluded = new Cesium.EllipsoidalOccluder(ellipsoid, camera.position).isPointVisible(position);
              if (!occluded) {{ entity.label.show = false; continue; }}

              var screenPos = scene.cartesianToCanvasCoordinates(position);
              if (!screenPos) {{ entity.label.show = false; continue; }}

              if (screenPos.x < 0 || screenPos.y < 0 || 
                  screenPos.x > canvasWidth || screenPos.y > canvasHeight) {{
                  entity.label.show = false;
                  continue;
              }}

              var boxW = (item.rank >= 5) ? 50 : 70;
              var boxH = 30;
              var myBox = {{ x: screenPos.x - boxW/2, y: screenPos.y - boxH/2, w: boxW, h: boxH }};

              var collision = false;
              for (var j = 0; j < placedBoxes.length; j++) {{
                  var other = placedBoxes[j];
                  if (myBox.x < other.x + other.w && myBox.x + myBox.w > other.x &&
                      myBox.y < other.y + other.h && myBox.y + myBox.h > other.y) {{
                      collision = true;
                      break;
                  }}
              }}

              if (collision) {{ entity.label.show = false; }} 
              else {{ entity.label.show = true; placedBoxes.push(myBox); }}
          }}
          viewer.scene.requestRender();
      }}

      function loadBoundaries() {{
          console.log("Loading boundaries...");

          Cesium.GeoJsonDataSource.load(
              'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_boundary_lines_land.geojson',
              {{ clampToGround: false }} 
          ).then(function(ds) {{
              var entities = ds.entities.values;
              for (var i = 0; i < entities.length; i++) {{
                  var entity = entities[i];
                  if (entity.polyline) {{
                      entity.polyline.material = Cesium.Color.WHITE.withAlpha(0.6);
                      entity.polyline.width = 1.0;
                      entity.polyline.arcType = Cesium.ArcType.GEODESIC;
                  }}
              }}
              viewer.dataSources.add(ds);
          }});
          
          var stateLinesUrl = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces_lines.geojson';
          
          fetch(stateLinesUrl)
            .then(response => response.json())
            .then(data => {{
                var instances = [];
                function addLine(coordinates) {{
                    var flatCoords = [];
                    for (var i = 0; i < coordinates.length; i++) {{
                        flatCoords.push(coordinates[i][0], coordinates[i][1]);
                    }}
                    if (flatCoords.length < 4) return;

                    instances.push(new Cesium.GeometryInstance({{
                        geometry: new Cesium.PolylineGeometry({{
                            positions: Cesium.Cartesian3.fromDegreesArray(flatCoords),
                            width: 1.5,
                            arcType: Cesium.ArcType.GEODESIC
                        }})
                    }}));
                }}

                if (data.features) {{
                    for (var i = 0; i < data.features.length; i++) {{
                        var geom = data.features[i].geometry;
                        if (!geom) continue;
                        if (geom.type === "LineString") {{
                            addLine(geom.coordinates);
                        }} else if (geom.type === "MultiLineString") {{
                            for (var j = 0; j < geom.coordinates.length; j++) {{
                                addLine(geom.coordinates[j]);
                            }}
                        }}
                    }}
                }}
                
                if (instances.length > 0) {{
                    stateBordersPrimitive = new Cesium.Primitive({{
                        geometryInstances: instances,
                        appearance: new Cesium.PolylineMaterialAppearance({{
                            material: Cesium.Material.fromType('PolylineDash', {{
                                color: Cesium.Color.WHITE.withAlpha(0.3),
                                dashLength: 8.0
                            }})
                        }}),
                        asynchronous: true 
                    }});
                    
                    var height = viewer.camera.positionCartographic.height;
                    stateBordersPrimitive.show = (height < 6000000);
                    viewer.scene.primitives.add(stateBordersPrimitive);
                }}
            }})
            .catch(error => {{
                console.error("Failed to load state borders: " + error);
            }});
          
          if (HAS_LOCAL_BORDERS) {{
               loadLocalLabelsOnly();
          }}
      }}

      function loadLocalLabelsOnly() {{
          Cesium.GeoJsonDataSource.load(LOCAL_BORDER_URL, {{ clampToGround: false }}).then(function(ds) {{
              var entities = ds.entities.values;
              for (var i = 0; i < entities.length; i++) {{
                  var entity = entities[i];
                  if (entity.polygon) entity.polygon.show = false;
                  if (entity.polyline) entity.polyline.show = false;
                  
                  var rawName = entity.properties[NAME_KEY];
                  if (rawName) {{
                      entity.label = {{
                          text: rawName.toString().toUpperCase(),
                          font: '600 10px Roboto, sans-serif',
                          fillColor: Cesium.Color.WHITE.withAlpha(0.7),
                          style: Cesium.LabelStyle.FILL,
                          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 3000000),
                          heightReference: Cesium.HeightReference.NONE,
                          disableDepthTestDistance: Number.POSITIVE_INFINITY,
                          translucencyByDistance: new Cesium.NearFarScalar(1.0e6, 1.0, 3.5e6, 0.0)
                      }};
                      entity.show = true; 
                  }} else {{
                      entity.show = false;
                  }}
              }}
              viewer.dataSources.add(ds);
          }}).catch(function(e){{ console.error(e); }});
      }}

      function loadHighDetailLabels() {{
          var countryUrl = 'https://raw.githubusercontent.com/mledoze/countries/master/dist/countries.json';
          Cesium.Resource.fetchJson(countryUrl).then(function(data) {{
              data.forEach(function(country) {{
                  var lat = country.latlng[0];
                  var lon = country.latlng[1];
                  var name = country.name.common || country.name;
                  var area = country.area || 0; 
                  if (!lat || !lon) return;
                  
                  var rank = 4; 
                  if (area > 2500000) rank = 1;      
                  else if (area > 400000) rank = 2;  
                  else if (area > 50000) rank = 3;   

                  addLabel(name.toUpperCase(), lat, lon, rank);
              }});
          }});
          
          Cesium.Resource.fetchJson('https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_10m_populated_places_simple.geojson').then(function(data) {{
              data.features.forEach(function(feature) {{
                  var coords = feature.geometry.coordinates;
                  var props = feature.properties;
                  var pop = props.pop_max || 0;
                  var name = props.name;
                  
                  if (pop > 1000000) {{
                      addLabel(name, coords[1], coords[0], 5);
                  }}
              }});
          }});
      }}

      function addLabel(text, lat, lon, rank) {{
          var displayScale = 0.5;
          var fontSize = (rank === 1) ? 28 : (rank === 5 ? 20 : 24);
          var fontWeight = (rank === 5) ? '700' : '900';
          var font = fontWeight + ' ' + fontSize + 'px Roboto, sans-serif';
          
          var translucency;
          if (rank <= 2) {{ 
              translucency = new Cesium.NearFarScalar(6.0e6, 1.0, 1.0e7, 0.0);
          }} else if (rank <= 4) {{ 
              translucency = new Cesium.NearFarScalar(4.0e6, 1.0, 7.0e6, 0.0);
          }} else {{ 
              translucency = new Cesium.NearFarScalar(1.5e6, 1.0, 6.0e6, 0.0);
          }}

          var entity = viewer.entities.add({{
              position: Cesium.Cartesian3.fromDegrees(lon, lat),
              label: {{
                  text: text,
                  font: font,
                  scale: displayScale,
                  fillColor: Cesium.Color.WHITE,
                  style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                  outlineColor: Cesium.Color.BLACK.withAlpha(0.8),
                  outlineWidth: 6,
                  horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                  verticalOrigin: Cesium.VerticalOrigin.CENTER,
                  
                  translucencyByDistance: translucency,
                  
                  heightReference: Cesium.HeightReference.NONE,
                  eyeOffset: new Cesium.Cartesian3(0.0, 0.0, -50000.0),
                  distanceDisplayCondition: undefined,
                  scaleByDistance: new Cesium.NearFarScalar(1.0, 1.0, 1.0e8, 1.0),
                  disableDepthTestDistance: Number.POSITIVE_INFINITY
              }}
          }});
          
          labelEntities.push({{ entity: entity, rank: rank, lat: lat, lon: lon }});
      }}
      
      function replaceAllPoints(newSet) {{
          allPoints = newSet;
          if (updatePointsForZoom) updatePointsForZoom();
      }}
      function addPointsBatch(newPts) {{
          allPoints.push(...newPts);
          setTimeout(updatePointsForZoom, 500);
      }}
      
      var lastRenderHeight = 0;
      function updatePointsForZoom() {{
          if (!viewer || !userPointCollection) return;
          
          var height = viewer.camera.positionCartographic.height;
          if (Math.abs(height - lastRenderHeight) / lastRenderHeight < 0.2) {{
              return;
          }}
          lastRenderHeight = height;
          
          requestAnimationFrame(() => {{
              userPointCollection.removeAll();
              userLabelCollection.removeAll();
              
              var CLUSTER_THRESHOLD = 3000000;
              if (height > CLUSTER_THRESHOLD) renderUserClusters(height);
              else renderUserPoints();
              
              viewer.scene.requestRender();
          }});
      }}

      // --- FIX START: Use CullingVolume instead of Frustum ---
      function renderUserPoints() {{
          var cullingVolume = viewer.camera.frustum.computeCullingVolume(
              viewer.camera.position,
              viewer.camera.direction,
              viewer.camera.up
          );
          var culledPoints = 0;

          for (var i = 0; i < allPoints.length; i++) {{
              var p = allPoints[i];
              var position = Cesium.Cartesian3.fromDegrees(p.lon, p.lat);
              
              if (viewer.scene.mode === Cesium.SceneMode.SCENE3D) {{
                  var visible = cullingVolume.computeVisibility(
                      new Cesium.BoundingSphere(position, 500)
                  );
                  if (visible === Cesium.Intersect.OUTSIDE) {{
                      culledPoints++;
                      continue;
                  }}
              }}

              userPointCollection.add({{
                  position: position,
                  color: Cesium.Color.CYAN,
                  pixelSize: 6,
                  outlineColor: Cesium.Color.BLACK,
                  outlineWidth: 1,
                  eyeOffset: new Cesium.Cartesian3(0.0, 0.0, -500.0),
                  distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 10000000)
              }});
          }}
      }}
      // --- FIX END ---

      function renderUserClusters(height) {{
          var grid = (height > 10000000) ? 10.0 : 4.0;
          var clusters = new Map();
          for (var i = 0; i < allPoints.length; i++) {{
              var p = allPoints[i];
              var k = Math.round(p.lat/grid)*grid + ',' + Math.round(p.lon/grid)*grid;
              if (!clusters.has(k)) clusters.set(k, {{lat: Math.round(p.lat/grid)*grid, lon: Math.round(p.lon/grid)*grid, count: 0}});
              clusters.get(k).count++;
          }}
          clusters.forEach(function(c) {{
              var size = Math.min(40, 15 + Math.log(c.count) * 4);
              userPointCollection.add({{
                  position: Cesium.Cartesian3.fromDegrees(c.lon, c.lat),
                  color: Cesium.Color.ORANGE.withAlpha(0.9),
                  pixelSize: size,
                  outlineColor: Cesium.Color.WHITE,
                  outlineWidth: 2,
                  eyeOffset: new Cesium.Cartesian3(0.0, 0.0, -500.0)
              }});
              userLabelCollection.add({{
                  position: Cesium.Cartesian3.fromDegrees(c.lon, c.lat),
                  text: c.count.toString(),
                  font: 'bold 14px sans-serif',
                  fillColor: Cesium.Color.WHITE,
                  style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                  outlineWidth: 2,
                  verticalOrigin: Cesium.VerticalOrigin.CENTER,
                  horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                  pixelOffset: new Cesium.Cartesian2(0, -2),
                  eyeOffset: new Cesium.Cartesian3(0.0, 0.0, -600.0)
              }});
          }});
      }}
      
      function setMapStyle(style) {{
          if (!viewer) return;
          viewer.imageryLayers.removeAll();
          if (style === 'dark') {{
              viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({{
                  url: 'https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}.png',
                  subdomains: 'abcd',
                  credit: 'CartoDB'
              }}));
          }} else {{
              viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({{
                  url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
                  credit: 'ESRI'
              }}));
          }}
          viewer.scene.requestRender();
      }}
      </script>
    </body>
    </html>"""
        
        self.view.setHtml(html_template, base_url)
        
        def inject_data_when_ready():
            def check_ready(is_ready):
                if is_ready:
                    CHUNK_SIZE = 5000
                    for i in range(0, len(valid_points), CHUNK_SIZE):
                        chunk = valid_points[i:i+CHUNK_SIZE]
                        json_chunk = json.dumps(chunk)
                        js = f"if (typeof replaceAllPoints === 'function' && {i}==0) replaceAllPoints({json_chunk}); else if (typeof addPointsBatch === 'function') addPointsBatch({json_chunk});"
                        self.view.page().runJavaScript(js)
                    self._is_loaded = True
                    self._has_been_rendered = True
                else:
                    QTimer.singleShot(500, inject_data_when_ready)
            
            self.view.page().runJavaScript(
                "typeof viewer !== 'undefined' && viewer !== null && typeof replaceAllPoints === 'function'",
                check_ready
            )
        
        QTimer.singleShot(1500, inject_data_when_ready)

class MainWindow(QMainWindow):
    @staticmethod
    def _norm(path: str) -> str:
        try:
            import os
            return os.path.normcase(path).replace("\\", "/")
        except Exception:
            return path.replace("\\", "/")

    def _apply_split_ratio(self):
        # keep 55% : 45% split and recompute columns
        total_w = self.splitter.width() or self.centralWidget().width()
        left_w  = int(total_w * 0.55)
        right_w = max(1, total_w - left_w)
        self.splitter.setSizes([left_w, right_w])
        self._update_grid_columns(right_w)

    def _update_grid_columns(self, _right_w: int):
        # Use the real drawable area but reserve a fixed scrollbar width
        vw = self.grid.viewport().width() or _right_w
        SCROLLBAR_RESERVE = 18  # px (typical Windows scroll width)
        vw = max(0, vw - SCROLLBAR_RESERVE)

        SPACING  = 4
        MAX_COLS = 4
        MIN_CELL = 110  # lower = easier to fit more columns

        # 1) decide how many cols fit at minimum cell width
        cols = max(1, min(MAX_COLS, (vw + SPACING) // (MIN_CELL + SPACING)))

        # 2) recompute cell width to fill the row exactly (no gutter)
        cell_w = (vw + SPACING) // cols - SPACING

        # 3) icon + text heights
        icon_w = max(48, min(96, cell_w - 16))
        text_h = self.grid.fontMetrics().height()
        cell_h = icon_w + text_h + 6

        self.grid.setSpacing(SPACING)
        self.grid.setIconSize(QSize(icon_w, icon_w))
        self.grid.setGridSize(QSize(cell_w, cell_h))
        self.grid.setWordWrap(False)

    def closeEvent(self, event):
        # Stop the GPS thread if it is running
        if hasattr(self, 'gps_indexer_thread') and self.gps_indexer_thread:
            print("[Exit] Stopping GPS Indexer...")
            self.gps_indexer_thread.stop()
            self.gps_indexer_thread.wait(2000) # Wait up to 2 seconds for it to finish cleanly
        
        # Stop the Organizer thread if it is running
        if hasattr(self, 'thread') and self.thread:
             self.thread.terminate()
             self.thread.wait()
             
        event.accept()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Travel Photo Organizer")
        WIN_W, WIN_H = 1200, 800
        self.resize(900, 600)

        # ---- Top controls (left: Choose Folder, right: Run) ----
        top_bar = QHBoxLayout()
        self.choose_btn = QPushButton("Choose Folder")
        self.path_label = QLabel("No folder selected")
        self.path_label.setStyleSheet("color: gray;")
        self.run_btn = QPushButton("Run Organizer")
        self.run_btn.setEnabled(False)  # enable only after a folder is chosen
        self.choose_btn.setFocusPolicy(Qt.NoFocus)
        self.run_btn.setFocusPolicy(Qt.NoFocus)

        top_bar.addWidget(self.choose_btn)
        top_bar.addWidget(self.path_label, 1)   # stretch
        top_bar.addWidget(self.run_btn)

        # ---- Center: Splitter with grid view on the right ----
        self.model = QFileSystemModel(self)

        def _refresh_thumb(p: str):
            # If we switched folders, ignore thumbs from old roots (gen guard already helps)
            if not self.grid.model():
                return

            # NEW: Check if path is still relevant (under current root)
            current_root_norm = self._norm(self.model.filePath(self.grid.rootIndex()))
            if not p.startswith(current_root_norm):
                self._pending_refresh.discard(self._norm(p))
                return

            idx = self.model.index(self._norm(p))
            if not idx.isValid():
                # queue for later and try again soon
                norm_p = self._norm(p)
                if len(self._pending_refresh) < 200:  # NEW: Cap pending to prevent accumulation
                    self._pending_refresh.add(norm_p)
                QTimer.singleShot(100, lambda: _refresh_thumb(p))  # NEW: Increased interval to 100ms
                return

            # valid index: remove from pending, nudge model + repaint one cell
            self._pending_refresh.discard(self._norm(p))
            self.model.dataChanged.emit(idx, idx, [Qt.DecorationRole, Qt.DisplayRole, Qt.SizeHintRole])
            rect = self.grid.visualRect(idx)
            self.grid.viewport().update(rect)
            self.grid.model().fetchMore(self.grid.rootIndex())  # harmless nudge

        provider = ThumbnailIconProvider(thumb_size=96, on_ready=_refresh_thumb)
        self._icon_provider = provider

        self.model = FSModel(self)                 # <-- use our model
        self.model.setIconProvider(provider)       # keep provider attached
        # âžœ ONLY show images & videos (folders still visible)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(IMAGE_EXTS + VIDEO_EXTS)
        self.model.setNameFilterDisables(False)
        self._pending_refresh = set()  # absolute paths waiting for index to exist

        # whenever rows appear or layout changes, flush pending
        self.model.rowsInserted.connect(lambda *_: self._flush_pending_refresh())
        self.model.layoutChanged.connect(self._flush_pending_refresh)
        self.model.directoryLoaded.connect(lambda _p: self._flush_pending_refresh())

        self.grid = QListView()
        self.grid.setModel(self.model)
        #self.grid.setViewMode(QListView.IconMode)      # <-- grid / boxes
        self.grid.setWrapping(True)
        self.grid.setResizeMode(QListView.Adjust)      # adapt layout when window resizes
        self.grid.setMovement(QListView.Static)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # tweak sizes to taste:
        self.grid.setIconSize(QSize(96, 96))        # tile icon size (we can adjust later)
        self.grid.setGridSize(QSize(120, 120))      # tile cell size
        self.grid.setSpacing(8)

        self.grid.setFrameShape(QFrame.NoFrame)
        self.grid.setViewportMargins(0, 0, 0, 0)
        self.grid.setStyleSheet(
            "QListView { padding:0; margin:0; border:0; }"
            "QListView::item { margin:0; padding:0; }"
        )

        self.grid.viewport().installEventFilter(self)
        self.grid.setLayoutMode(QListView.Batched)
        self.grid.setBatchSize(256)  # draw items in batches to avoid stutter

        # Make sure wrapping is on and movement is static (already set, but reaffirm):
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setFlow(QListView.LeftToRight)
        self.grid.setWrapping(True)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)

        # Selection behavior (so the "pointer" is an item)
        self.grid.setSelectionBehavior(QListView.SelectItems)
        self.grid.setSelectionMode(QListView.SingleSelection)
        self.grid.setFocusPolicy(Qt.StrongFocus)

        # Initially, don't show anything in the grid until a folder is chosen
        self.grid.setModel(None)

        self.grid.verticalScrollBar().valueChanged.connect(
            lambda _v: self._icon_provider._maybe_start_pending()
        )

        # --- Keyboard shortcuts ---
        self.short_back     = QShortcut(QKeySequence(Qt.ALT | Qt.Key_Up), self)
        self.short_enter    = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.short_enter2   = QShortcut(QKeySequence(Qt.Key_Enter),  self)

        self.short_back.activated.connect(self.on_nav_up)
        self.short_enter.activated.connect(self.on_open_current)
        self.short_enter2.activated.connect(self.on_open_current)

        # --- Loading overlay (covers the grid while loading) ---
        self.loading = QLabel("Loading folderâ€¦")
        self.loading.setAlignment(Qt.AlignCenter)
        self.loading.setStyleSheet("""
            background: rgba(0,0,0,0.35);
            color: white;
            font-size: 16px;
            padding: 12px;
        """)
        self.loading.setVisible(False)
        # put the overlay on top of the grid
        self.loading.setParent(self.grid)
        self.loading.raise_()

        self._pending_dir = None # path string of a directory we are switching to
        self._want_select_first = False

        # Left pane placeholder (for future stats/filters)
        left_placeholder = QWidget()

        # --- Right panel: header (Back + path) + grid ---
        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(6)

        self.back_btn = QPushButton("â—€ Back")
        self.back_btn.setFocusPolicy(Qt.NoFocus)
        self.path_now = QLabel("â€”")
        self.path_now.setStyleSheet("color: #ccc;")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.back_btn, 2)  # 2:8 ratio
        header.addWidget(self.path_now, 8)
        right_v.addLayout(header)
        right_v.addWidget(self.grid, 1)

        # --- LEFT panel: controls beside the file view ---
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(6, 6, 6, 6)
        left_v.setSpacing(6)

        # --- LEFT: Staging preview (thumbnails of _incoming/<session>) ---
        self.staging_model = ThumbnailFileSystemModel(QSize(256, 256), self)
        self.staging_model.setFilter(QDir.Files | QDir.NoDotAndDotDot)
        self.staging_model.setNameFilters(IMAGE_EXTS + VIDEO_EXTS)
        self.staging_model.setNameFilterDisables(False)

        self.staging_view = QListView(self)
        self.staging_view.setViewMode(QListView.IconMode)
        self.staging_view.setResizeMode(QListView.Adjust)
        self.staging_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.staging_view.setIconSize(QSize(128, 128))
        self.staging_view.setSpacing(8)
        self.staging_view.setUniformItemSizes(True)
        self.staging_view.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.staging_view.setModel(self.staging_model)

        # Header row for the left panel (title + Globe toggle)
        self.staging_hdr = QLabel("Staging â€” _incoming")
        self.staging_hdr.setStyleSheet("font-weight: 600; color: #bbb;")

        self.globe_toggle_btn = QPushButton("Globe")
        self.globe_toggle_btn.setCheckable(True)
        self.globe_toggle_btn.setFocusPolicy(Qt.NoFocus)

        # TEMPORARY: Debug button to test GPS cache
        self.test_gps_btn = QPushButton("Test GPS")
        self.test_gps_btn.setFocusPolicy(Qt.NoFocus)
        self.test_gps_btn.clicked.connect(self.on_test_gps)

        #Add GPS cache reference
        self.gps_cache = None
        self.gps_indexer_thread = None

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        header_row.addWidget(self.staging_hdr, 1)
        header_row.addWidget(self.globe_toggle_btn)
        header_row.addWidget(self.test_gps_btn)

        # Stack: page 0 = staging thumbnails, page 1 = globe
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.staging_view)   # index 0

        self.globe_widget = GlobeWidget()
        self.left_stack.addWidget(self.globe_widget)   # index 1

        controls_row = QHBoxLayout()

        left_v.addLayout(header_row)
        left_v.addWidget(self.left_stack, 1)

        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setEnabled(False)  # enabled after a folder is chosen
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)

        controls_row.addWidget(self.choose_btn)
        controls_row.addWidget(self.upload_btn)
        controls_row.addWidget(self.path_label, 1)
        controls_row.addStretch()
        controls_row.addWidget(self.run_btn)
        self.upload_btn.clicked.connect(self.on_upload_clicked)

        left_v.addLayout(controls_row)

        # --- Splitter with left controls and right file view ---
        splitter = LockedSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setHandleWidth(0)
        splitter.setStyleSheet("QSplitter::handle { image: none; background: transparent; }")
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        
        # ---- Main layout ----
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        QTimer.singleShot(0, self._apply_split_ratio)
        self.setStatusBar(QStatusBar(self))
        # Make sure initial nav state reflects "no folder selected"
        self.base_path = None
        self._nav_stack = []
        self._update_nav_state()

        self.thread = None

        # ---- Signals ----
        self.choose_btn.clicked.connect(self.on_choose_folder)
        self.run_btn.clicked.connect(self.on_run_clicked)  # not wired yet

        if hasattr(self, "globe_toggle_btn"):
            self.globe_toggle_btn.toggled.connect(self.on_globe_toggled)

        # Keep track of the selected folder path
        self.selected_folder = None
        # Worker thread pointer (set when running organizer)
        self.thread = None

        # Optional: a simple filter dropdown for later (commented out to avoid overload)
        # self.filter_combo = QComboBox()
        # self.filter_combo.addItems(["All files", "Images only", "Videos only"])
        # top_bar.insertWidget(1, self.filter_combo)
        # self.filter_combo.currentIndexChanged.connect(self.apply_filter)

        self.model.directoryLoaded.connect(self.on_directory_loaded)
        self.model.layoutChanged.connect(self._try_select_first)
        self.model.rowsInserted.connect(lambda *_: self._try_select_first())

        # nav stack + signals
        self._nav_stack = []
        self.grid.doubleClicked.connect(self.on_item_activated)
        self.back_btn.clicked.connect(self.on_nav_back)

        # Debounced layout scheduler
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(60)  # ms
        self._layout_timer.timeout.connect(self._apply_split_ratio)

    def _force_requery_icons(self):
        """Make the view/model immediately re-ask for DecorationRole icons."""
        root = self.grid.rootIndex()
        if not root.isValid():
            return
        rows = self.model.rowCount(root)
        if rows <= 0:
            return
        first = self.model.index(0, 0, root)
        last  = self.model.index(rows - 1, 0, root)
        # Tell the model the decoration may have changed â†’ provider.icon() runs again
        self.model.dataChanged.emit(first, last, [Qt.DecorationRole, Qt.DisplayRole, Qt.SizeHintRole])
        # Nudge layout so the paint happens right away
        self.grid.doItemsLayout()
        # Kick the thumbnail providerâ€™s pending queue
        self._icon_provider._maybe_start_pending()

    def _touch_visible_icons(self):
        """Force the provider.icon(...) to run for items currently visible."""
        root = self.grid.rootIndex()
        if not root.isValid():
            return
        rows = self.model.rowCount(root)
        if rows <= 0:
            return

        vp = self.grid.viewport().rect()
        # Find rough top/bottom rows on screen, with a small margin
        top_idx = self.grid.indexAt(vp.topLeft())
        bot_idx = self.grid.indexAt(vp.bottomLeft())
        start = top_idx.row() if top_idx.isValid() else 0
        end   = bot_idx.row() if bot_idx.isValid() else min(rows - 1, start + 120)
        start = max(0, start - 20)
        end   = min(rows - 1, end + 40)

        # Touch only the visible slice to keep it light
        for r in range(start, end + 1):
            idx = self.model.index(r, 0, root)
            if not idx.isValid():
                continue
            rect = self.grid.visualRect(idx)
            if not rect.intersects(vp):
                continue
            # Calling icon(...) schedules/queues a job if there is no cache-hit.
            fi = self.model.fileInfo(idx)
            self._icon_provider.icon(fi)

    def resizeEvent(self, event):
        """Keep button in top-right corner when widget resizes"""
        super().resizeEvent(event)
        if hasattr(self, 'map_style_btn'):
            # Position button 10px from top-right corner
            btn_size = self.map_style_btn.size()
            x = self.width() - btn_size.width() - 10
            y = 10
            self.map_style_btn.move(x, y)
    
    def eventFilter(self, obj, event):
        if obj is self.grid.viewport() and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._schedule_layout)
        return super().eventFilter(obj, event)

    def on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to preview")
        if not folder:
            return

        self.selected_folder = folder

        # Clean up old GPS thread
        if hasattr(self, 'gps_indexer_thread') and self.gps_indexer_thread:
            try:
                self.gps_indexer_thread.intermediate_points.disconnect()
                self.gps_indexer_thread.progress.disconnect()
                self.gps_indexer_thread.finished_with_count.disconnect()
                self.gps_indexer_thread.error.disconnect()
            except Exception:
                pass

            self.gps_indexer_thread.stop()
            self.gps_indexer_thread.quit()
            self.gps_indexer_thread.wait(3000)
            print(f"[GPS Switch] Stopped old thread after {folder}")

        self.gps_indexer_thread = None

        # Clear globe if visible
        if hasattr(self, "globe_widget"):
            print(f"[GPS Switch] Resetting globe for new library: {folder}")
            self.globe_widget._has_been_rendered = False  # Force re-render for new library
            if getattr(self, "globe_toggle_btn", None) and self.globe_toggle_btn.isChecked():
                # If currently viewing globe, clear it
                self.globe_widget.set_points([], use_clustering=False)

        self.path_label.setText(folder)
        self._nav_stack.clear()
        self._set_current_path(folder)
        self.path_label.setStyleSheet("")
        self.statusBar().showMessage("Folder selected.", 2000)

        # ... rest of your existing code for staging setup ...
        incoming_root = os.path.join(self.selected_folder, "_incoming")
        os.makedirs(incoming_root, exist_ok=True)
        # ... (keep your existing staging code) ...

        # Show loading overlay
        self.loading.setGeometry(self.grid.rect())
        self.loading.setVisible(True)
        self.grid.setEnabled(False)
        QApplication.processEvents()

        self._icon_provider.cancel_all_and_clear()   
        self._pending_refresh.clear()

        QTimer.singleShot(0, lambda: self._load_folder(folder))
        QTimer.singleShot(50, self._schedule_layout)

        self.run_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        # Initialize GPS cache
        self.gps_cache = GPSCache(folder)

        # **KEY FIX**: Start background indexing immediately (don't wait for globe toggle)
        self._start_background_gps_index()
    
    def _load_folder(self, folder: str):
        # If the grid has no model yet (first time), attach it now
        if self.grid.model() is None:
            self.grid.setModel(self.model)

        # This call may take time on big trees; overlay is already visible
        root_index = self.model.setRootPath(folder)
        self.grid.setRootIndex(root_index)
        self._icon_provider.begin_folder(folder)
        QTimer.singleShot(0, self._prewarm_thumbs_for_current_folder)
        self.base_path = folder
        self.base_root = root_index
        self._update_nav_state()

    def on_directory_loaded(self, path: str):
        # Hide overlay if this is the directory we were navigating to,
        # or if overlay is visible anyway (safe + idempotent)
        if self.loading.isVisible():
            self._hide_loading()
        QTimer.singleShot(0, self._schedule_layout)
        QTimer.singleShot(50, self._schedule_layout)
        self.grid.doItemsLayout() 
        self._icon_provider._maybe_start_pending()
        self._touch_visible_icons() 
        # reflect the actual root path in header
        current_root = self.model.filePath(self.grid.rootIndex())
        self._set_current_path(current_root)
        self._update_nav_state()
        QTimer.singleShot(0, self._request_select_first)
        root = self.grid.rootIndex()
        rows = self.model.rowCount(root)
        if rows > 0:
            first = self.model.index(0, 0, root)
            last  = self.model.index(rows - 1, 0, root)
            self.model.dataChanged.emit(first, last, [Qt.DecorationRole])
            QTimer.singleShot(0, self._prewarm_thumbs_for_current_folder)


    def on_run_clicked(self):
            import os
            if not self.selected_folder or not os.path.isdir(self.selected_folder):
                QMessageBox.warning(self, "No folder", "Please choose a valid folder first.")
                return

            # Prevent double-starts
            if self.thread and self.thread.isRunning():
                QMessageBox.information(self, "Busy", "Organizer is already running.")
                return

            # Build an optional resume message line
            resume = self._scan_resume_state()
            resume_line = ""
            if resume and resume["total"] > 0:
                if resume["processed"] > 0:
                    resume_line = (
                        f"\n\nPrevious progress: {resume['processed']} of {resume['total']} "
                        f"files already processed; {resume['remaining']} will be resumed."
                    )
                else:
                    # Session exists but nothing processed yet; still OK to mention
                    resume_line = f"\n\nStaging session has {resume['total']} file(s) to process."

            # Ask how to handle duplicates (with resume info appended)
            choice = QMessageBox.question(
                self,
                "Duplicate photos/videos",
                (
                    "If some staged files are exact duplicates of items already in the library,\n"
                    "how should they be handled?\n\n"
                    "Yes = Skip duplicates (do not copy them again)\n"
                    "No  = Copy duplicates as new files (with unique names)\n"
                    "Cancel = Do not run the organizer now."
                    + resume_line
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )

            if choice == QMessageBox.Cancel:
                return
            dup_policy = "skip" if choice == QMessageBox.Yes else "copy_rename"

            # Disable controls while running
            self.choose_btn.setEnabled(False)
            self.run_btn.setEnabled(False)
            mode_text = "skipping duplicates" if dup_policy == "skip" else "copying duplicates as new files"
            self.statusBar().showMessage(f"Running organizerâ€¦ ({mode_text})")

            # Start worker thread
            self.thread = WorkerThread(self.selected_folder, dup_policy, parent=self)
            self.thread.finished_ok.connect(self.on_finished)
            self.thread.error.connect(self.on_error)

            # When the thread fully finishes, clear pointer so the next run can start
            self.thread.finished.connect(lambda: setattr(self, "thread", None))

            self.thread.start()

    def on_upload_clicked(self):
        if not self.selected_folder:
            QMessageBox.information(self, "Choose folder", "Pick your library folder first.")
            return

        # choose files anywhere
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select photos/videos",
            "",
            "Media files (*.jpg *.jpeg *.png *.gif *.bmp *.tif *.tiff *.heic *.heif *.webp *.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)"
        )
        if not files:
            return

        # First, validate which ones are real media (magic-bytes + no shortcuts/symlinks)
        valid_files = []
        rejected = 0
        for src in files:
            if self._is_valid_media_file(src):
                valid_files.append(src)
            else:
                rejected += 1

        if not valid_files:
            QMessageBox.warning(
                self,
                "Upload",
                "None of the selected files are valid photos or videos.\n"
                "They may be shortcuts, unsupported types, or corrupt."
            )
            return

        # make _incoming/<session>
        import time, shutil
        incoming_root = os.path.join(self.selected_folder, "_incoming")
        os.makedirs(incoming_root, exist_ok=True)
        session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(incoming_root, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # simple copy loop (copy2 keeps timestamps)
        errors = 0
        copied = 0
        for src in valid_files:
            try:
                base = os.path.basename(src)
                dst = os.path.join(session_dir, base)
                # avoid name collisions inside the session
                i = 1
                root, ext = os.path.splitext(dst)
                while os.path.exists(dst):
                    dst = f"{root}-{i}{ext}"
                    i += 1
                shutil.copy2(src, dst)
                copied += 1

                # copy sidecar if present (e.g., Google Takeout .json)
                if os.path.exists(src + ".json"):
                    sidecar_dst = dst + ".json"
                    shutil.copy2(src + ".json", sidecar_dst)
            except Exception:
                errors += 1

        # Make the LEFT (staging) panel show this session
        stg_idx = self.staging_model.setRootPath(session_dir)
        self.staging_view.setRootIndex(stg_idx)
        self.staging_hdr.setText(f"Staging â€” {os.path.basename(session_dir)}")

        # Build a friendly status message
        if errors:
            QMessageBox.warning(
                self,
                "Upload",
                f"Uploaded {copied} file(s) into _incoming/{session_id}, "
                f"with {errors} copy error(s)."
                + (f"\nSkipped {rejected} non-media/shortcut file(s)." if rejected else "")
            )
        else:
            msg = f"Uploaded {copied} file(s) into _incoming/{session_id}"
            if rejected:
                msg += f" (skipped {rejected} non-media/shortcut file(s))"
            self.statusBar().showMessage(msg, 5000)

        # Library content changed; refresh globe if it's visible
        if getattr(self, "globe_toggle_btn", None) and self.globe_toggle_btn.isChecked():
            self.update_globe_for_library_cached()

    # Optional simple extension filter, if you want to enable later.
    def apply_filter(self):
        idx = self.filter_combo.currentIndex()
        if idx == 0:
            # All files
            self.model.setNameFilters([])  # clear
            self.model.setNameFilterDisables(False)
        elif idx == 1:
            # Images only
            self.model.setNameFilters(IMAGE_EXTS)
            self.model.setNameFilterDisables(False)
        elif idx == 2:
            # Videos only
            self.model.setNameFilters(VIDEO_EXTS)
            self.model.setNameFilterDisables(False)

    def on_finished(self):
        # Thread has finished; clear our pointer so we don't touch a deleted object
        self.thread = None

        self.statusBar().showMessage("Done ✅ - Refreshing Map...", 3000)
        self.choose_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

        # Reset staging view back to _incoming (session was likely auto-emptied)
        if self.selected_folder:
            incoming_root = os.path.join(self.selected_folder, "_incoming")
            idx = self.staging_model.setRootPath(incoming_root)
            self.staging_view.setRootIndex(idx)
            self.staging_hdr.setText("Staging : _incoming (empty)")

            # === FIX START: TRIGGER RE-SCAN ===
            # The files are moved, but the GPS database doesn't know about them yet.
            # We restart the background indexer to find the new files and put them on the map.
            print("[UI] Organization complete. Restarting GPS indexer to map new files.")
            self._start_background_gps_index()

    def on_error(self, msg: str):
        # Thread will be torn down; clear pointer
        self.thread = None

        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Error running organizer", msg)
        self.choose_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

    def _set_current_path(self, path: str):
        fm = self.path_now.fontMetrics()
        width = self.path_now.width() if self.path_now.width() > 0 else 600
        self.path_now.setText(fm.elidedText(path, Qt.ElideMiddle, width))
        self.path_now.setToolTip(path)

    def _scan_resume_state(self):
        """
        Inspect the newest _incoming session and return a small summary dict or None.

        Returns dict with keys:
            session_dir, session_name, total, processed, remaining, duplicates, errors
        """
        if not self.selected_folder:
            return None

        incoming_root = os.path.join(self.selected_folder, "_incoming")
        if not os.path.isdir(incoming_root):
            return None

        # Find newest session dir (ignore internal folders starting with ._)
        try:
            candidates = []
            for name in os.listdir(incoming_root):
                session_path = os.path.join(incoming_root, name)
                if not os.path.isdir(session_path):
                    continue
                if name.startswith("._"):
                    continue
                candidates.append(session_path)
            if not candidates:
                return None
            session_dir = max(candidates, key=os.path.getmtime)
        except Exception:
            return None

        state_dir = os.path.join(session_dir, "._state")
        jsonl_path = os.path.join(state_dir, "session.jsonl")
        meta_path = os.path.join(state_dir, "session.meta.json")
        if not os.path.exists(jsonl_path):
            return None

        # Count media files in the session dir
        exts = {e.lower().lstrip("*.") for e in (IMAGE_EXTS + VIDEO_EXTS)}
        total = 0
        try:
            for name in os.listdir(session_dir):
                full = os.path.join(session_dir, name)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                if ext not in exts:
                    continue
                total += 1
        except Exception:
            total = 0

        processed = 0
        duplicates = 0
        errors = 0
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
                    status = (rec.get("status") or "").lower()
                    if status in ("verified", "duplicate", "duplicate_skipped", "duplicate_copied"):
                        processed += 1
                        if "duplicate" in status:
                            duplicates += 1
                    elif status == "error":
                        errors += 1
        except Exception:
            pass

        remaining = max(total - processed, 0)
        return {
            "session_dir": session_dir,
            "session_name": os.path.basename(session_dir),
            "total": total,
            "processed": processed,
            "remaining": remaining,
            "duplicates": duplicates,
            "errors": errors,
        }

    def on_item_activated(self, index):
        if not index.isValid():
            return
        path = self.model.filePath(index)
        if self.model.isDir(index):
            self._icon_provider.cancel_all_and_clear() 
            self._pending_refresh.clear()  # NEW: Clear pending refresh on nav
            # push current root for Back
            self._nav_stack.append(self.grid.rootIndex())
            # enter folder
            self._pending_dir = path
            self._show_loading()
            self.grid.setRootIndex(index)
            self._icon_provider.begin_folder(path)
            QTimer.singleShot(0, self._prewarm_thumbs_for_current_folder)
            self._set_current_path(path)
            self._update_nav_state()
            QTimer.singleShot(0, self._schedule_layout)
            QTimer.singleShot(50, self._schedule_layout)
            QTimer.singleShot(0, self._request_select_first)
            QTimer.singleShot(0, self._force_requery_icons)
            QTimer.singleShot(0, self._touch_visible_icons)
            # Fallback: if Qt didn't emit directoryLoaded (already cached), hide after 400ms
            QTimer.singleShot(400, lambda: self.loading.isVisible() and self._hide_loading())

        else:
            # open file with default app
            try:
                os.startfile(path)
            except Exception:
                QMessageBox.information(self, "Open file", path)

    def on_nav_back(self):
        if not self._nav_stack:
            return
        self._icon_provider.cancel_all_and_clear()   
        self._pending_refresh.clear()  # NEW: Clear pending refresh on nav
        prev_root = self._nav_stack.pop()
        self._pending_dir = self.model.filePath(prev_root)
        self._show_loading()
        self.grid.setRootIndex(prev_root)
        self._icon_provider.begin_folder(self.model.filePath(prev_root))
        QTimer.singleShot(0, self._prewarm_thumbs_for_current_folder)
        root_path = self.model.filePath(prev_root)
        self._set_current_path(root_path)
        self._update_nav_state()
        QTimer.singleShot(0, self._schedule_layout)
        QTimer.singleShot(50, self._schedule_layout)
        QTimer.singleShot(0, self._request_select_first)
        QTimer.singleShot(0, self._force_requery_icons)
        QTimer.singleShot(0, self._touch_visible_icons)
        QTimer.singleShot(400, lambda: self.loading.isVisible() and self._hide_loading())

    def on_nav_up(self):
        root_path = self.model.filePath(self.grid.rootIndex())
        if root_path == (self.base_path or self.selected_folder):
            return  # already at top; do nothing

        root = self.grid.rootIndex()
        if not root.isValid():
            return
        self._icon_provider.cancel_all_and_clear()   
        self._pending_refresh.clear()  # NEW: Clear pending refresh on nav
        parent_idx = self.model.parent(root)
        if parent_idx.isValid():
            self._pending_dir = self.model.filePath(parent_idx)
            self._show_loading()
            self.grid.setRootIndex(parent_idx)
            self._icon_provider.begin_folder(self.model.filePath(parent_idx))
            QTimer.singleShot(0, self._prewarm_thumbs_for_current_folder)
            self._set_current_path(self.model.filePath(parent_idx))
            self._update_nav_state()
            self._schedule_layout()
            QTimer.singleShot(0, self._request_select_first)
            QTimer.singleShot(0, self._force_requery_icons)
            QTimer.singleShot(0, self._touch_visible_icons)
            QTimer.singleShot(400, lambda: self.loading.isVisible() and self._hide_loading())

    def _schedule_layout(self):
        # Restart a short single-shot timer; collapses bursts of layout calls
        self._layout_timer.start()

    def on_open_current(self):
        idx = self.grid.currentIndex()
        if idx.isValid():
            self.on_item_activated(idx)

    def _select_first(self):
        root = self.grid.rootIndex()
        if not root.isValid():
            return
        first = self.model.index(0, 0, root)
        sm = self.grid.selectionModel()
        if first.isValid() and sm:
            sm.clearSelection()
            sm.setCurrentIndex(first, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Current)
            self.grid.scrollTo(first, QListView.PositionAtTop)
            self.grid.setFocus(Qt.OtherFocusReason)

    def _request_select_first(self):
        # mark intent; weâ€™ll satisfy when rows exist
        self._want_select_first = True
        QTimer.singleShot(0, self._try_select_first)

    def _try_select_first(self):
        if not self._want_select_first:
            return
        root = self.grid.rootIndex()
        if not root.isValid():
            return
        first = self.model.index(0, 0, root)
        if first.isValid():
            sm = self.grid.selectionModel()
            if sm:
                sm.clearSelection()
                sm.setCurrentIndex(first, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Current)
            else:
                self.grid.setCurrentIndex(first)
            self.grid.scrollTo(first, QListView.PositionAtTop)
            self.grid.setFocus(Qt.OtherFocusReason)
            self._want_select_first = False   # satisfied
        else:
            # try again shortlyâ€”rows may not be in yet
            QTimer.singleShot(30, self._try_select_first)

    # -- helpers to show/hide the loading overlay --
    def _show_loading(self):
        self.loading.setGeometry(self.grid.rect())
        self.loading.setVisible(True)
        self.grid.setEnabled(False)
        QApplication.processEvents()
    
    def _hide_loading(self):
        self.loading.setVisible(False)    # <- this is the correct overlay widget
        self.grid.setEnabled(True)

        # Focus the grid only if there are items
        root = self.model.index(self.model.rootPath())
        if self.model.rowCount(root) > 0:
            first = self.model.index(0, 0, root)
            if first.isValid():
                self.grid.setCurrentIndex(first)
                self.grid.setFocus(Qt.OtherFocusReason)

    def _update_nav_state(self):
        # No base path yet -> at "start" state: Back must be disabled
        if not getattr(self, "base_path", None):
            self.back_btn.setEnabled(False)
            return

        root = self.grid.rootIndex()
        at_root = (self.model.filePath(root) == self.base_path)
        self.back_btn.setEnabled(not at_root and len(self._nav_stack) > 0)  

    def _flush_pending_refresh(self):
        if not self._pending_refresh or not self.grid.model():
            return

        # copy to avoid set-size change during iteration
        current_root_norm = self._norm(self.model.filePath(self.grid.rootIndex()))
        for p in list(self._pending_refresh):
            # NEW: Skip if not relevant to current root
            if not p.startswith(current_root_norm):
                self._pending_refresh.discard(p)
                continue

            idx = self.model.index(self._norm(p))
            if idx.isValid():
                self._pending_refresh.discard(p)
                self.model.dataChanged.emit(idx, idx, [Qt.DecorationRole, Qt.DisplayRole, Qt.SizeHintRole])
                rect = self.grid.visualRect(idx)
                self.grid.viewport().update(rect)
    
    def _prewarm_thumbs_for_current_folder(self):
        root = self.grid.rootIndex()
        if not root.isValid():
            return

        rows = self.model.rowCount(root)
        if rows <= 0:
            return

        # Only prewarm the first 800 items, the rest will be loaded lazily on scroll
        MAX_PREWARM = 800
        rows_to_touch = min(rows, MAX_PREWARM)
        CHUNK = 200  # still in small slices

        def kick(start=0):
            if start >= rows_to_touch:
                return
            end = min(start + CHUNK, rows_to_touch)
            for r in range(start, end):
                idx = self.model.index(r, 0, root)
                fi = self.model.fileInfo(idx)
                if fi.isDir():
                    continue
                self._icon_provider.icon(fi)
            QTimer.singleShot(0, lambda: kick(end))

        kick()

    def _is_valid_media_file(self, path: str) -> bool:
        """
        Return True only if:
        - not a symlink
        - not a Windows shortcut (.lnk, .url)
        - extension is in our allowed image/video lists
        - and the content looks like a real image/video (magic-bytes style check)
        """
        if not path or not os.path.isfile(path):
            return False

        # Reject symlinks
        try:
            if os.path.islink(path):
                return False
        except Exception:
            # If os.path.islink fails for some reason, be safe and reject
            return False

        # Reject Windows shortcuts / URL files
        ext = os.path.splitext(path)[1].lower()
        if ext in (".lnk", ".url"):
            return False

        # Allowed extensions (same family as elsewhere in your app)
        img_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".webp"}
        vid_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

        if ext not in img_exts and ext not in vid_exts:
            return False

        # --- Magic-bytes style validation ---

        # Images: use Qt first, then Pillow as fallback
        if ext in img_exts:
            try:
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                if reader.canRead():
                    img = reader.read()
                    if not img.isNull():
                        return True
                # Qt failed or gave null â†’ try Pillow
                try:
                    im = Image.open(path)
                    im.verify()  # raises if not a real image
                    return True
                except Exception:
                    return False
            except Exception:
                return False

        # Videos: try to open and grab one frame with OpenCV
        if ext in vid_exts:
            try:
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    return False
                return True
            except Exception:
                return False

        # Fallback: be conservative
        return False
    
    def on_globe_toggled(self, checked: bool):
        if checked:
            self.globe_toggle_btn.setText("Staging")
            self.left_stack.setCurrentWidget(self.globe_widget)

            # Initialize cache if needed
            if not self.gps_cache and self.selected_folder:
                self.gps_cache = GPSCache(self.selected_folder)

            # **KEY IMPROVEMENT**: Only render globe HTML on FIRST view
            if not hasattr(self.globe_widget, '_has_been_rendered') or not self.globe_widget._has_been_rendered:
                print("[Globe Toggle] First time viewing globe - rendering HTML")

                # Load whatever we have in cache immediately
                cached_points = self.gps_cache.get_all_points(limit=50000) if self.gps_cache else []
                print(f"[Globe Toggle] Loading {len(cached_points)} cached points")

                if cached_points:
                    self.globe_widget.set_points(cached_points, use_clustering=len(cached_points) > 100)
                    self.statusBar().showMessage(f"Map: {len(cached_points)} locations (updating...)", 3000)
                else:
                    # Show empty globe but indicate scanning is in progress
                    self.globe_widget.set_points([], use_clustering=False)
                    self.statusBar().showMessage("Scanning for GPS data...", 3000)

                # Mark as rendered so we don't do it again
                self.globe_widget._has_been_rendered = True
            else:
                # FIX: Always refresh data when switching tabs, in case background indexer added points
                print("[Globe Toggle] Globe already rendered - refreshing view with latest DB data")
                if self.gps_cache:
                    # 1. Get the latest points from the DB (including the 14 new ones)
                    latest_points = self.gps_cache.get_all_points(limit=50000)
                    
                    # 2. Send them to the globe
                    # (Your set_points function is smart—it won't reload the page, just update the dots)
                    self.globe_widget.set_points(latest_points, use_clustering=len(latest_points) > 100)
                    
                    self.statusBar().showMessage(f"Map: {len(latest_points)} locations", 2000)

            # Background thread continues updating regardless of view
        else:
            self.globe_toggle_btn.setText("Globe")
            self.left_stack.setCurrentWidget(self.staging_view)

    def update_globe_for_library(self):
        """
        Backwards-compatible wrapper: always go through GPSCache.
        """
        if not hasattr(self, "globe_widget") or self.globe_widget is None:
            return

        folder = getattr(self, "selected_folder", None)
        if not folder or not os.path.isdir(folder):
            self.globe_widget.set_points([])
            return

        if not self.gps_cache:
            self.gps_cache = GPSCache(folder)

        self.update_globe_for_library_cached()

    def update_globe_for_library_cached(self):
        """Load GPS with progressive strategy"""
        if not self.gps_cache or not hasattr(self, "globe_widget"):
            return

        print("[DEBUG] update_globe_for_library_cached called")

        # Strategy 1: Show cached data immediately (instant)
        cached_points = self.gps_cache.get_all_points(limit=10000)

        if cached_points:
            print(f"[DEBUG] Showing {len(cached_points)} cached points immediately")
            self.globe_widget.set_points(cached_points, use_clustering=len(cached_points) > 100)
            self.statusBar().showMessage(f"Map: {len(cached_points)} locations (cached)", 3000)
        else:
            # Strategy 2: No cache - do a quick scan of visible folders first
            print("[DEBUG] No cache - doing quick scan of top folders")
            quick_points = []

            # Scan first few country folders
            try:
                for item in os.listdir(self.selected_folder)[:5]:  # first 5 folders
                    folder_path = os.path.join(self.selected_folder, item)
                    if os.path.isdir(folder_path) and not item.startswith("_"):
                        folder_points = self.gps_cache.quick_scan_folder(folder_path, max_files=20)
                        quick_points.extend(folder_points)

                        if len(quick_points) >= 50:  # Got enough for initial display
                            break
            except Exception as e:
                print(f"[DEBUG] Quick scan error: {e}")

            if quick_points:
                print(f"[DEBUG] Quick scan found {len(quick_points)} points")
                self.globe_widget.set_points(quick_points, use_clustering=False)
                self.statusBar().showMessage(f"Map: {len(quick_points)} locations (scanning...)", 3000)

        # Strategy 3: Start full background scan (will update progressively)
        self._start_background_gps_index()

    def on_test_gps(self):
        """Debug function to test GPS cache"""
        if not self.selected_folder:
            print("[TEST] No folder selected")
            QMessageBox.information(self, "Test GPS", "Please select a folder first")
            return
        print("\n" + "="*60)
        print("[TEST] Starting GPS cache test")
        print("="*60)
        # Check if cache exists
        if not self.gps_cache:
            print("[TEST] Creating GPS cache...")
            self.gps_cache = GPSCache(self.selected_folder)
        # Check current database contents
        print("\n[TEST] Current database contents:")
        conn = sqlite3.connect(self.gps_cache.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM gps_points")
        count = cursor.fetchone()[0]
        print(f"[TEST] Database has {count} entries")
        if count > 0:
            cursor = conn.execute("SELECT filepath, lat, lon, country FROM gps_points LIMIT 5")
            print("[TEST] Sample entries:")
            for row in cursor:
                print(f"  - {os.path.basename(row[0])}: ({row[1]}, {row[2]}) {row[3]}")
        conn.close()
        # Try to get points
        print("\n[TEST] Testing get_all_points():")
        all_points = self.gps_cache.get_all_points(limit=10)
        print(f"[TEST] Got {len(all_points)} points")
        for p in all_points[:3]:
            print(f"  - {p}")
        print("\n[TEST] Testing get_clustered_points():")
        clustered = self.gps_cache.get_clustered_points(zoom_level=5)
        print(f"[TEST] Got {len(clustered)} clusters")
        for c in clustered[:3]:
            print(f"  - {c}")
        # Force a manual scan
        print("\n[TEST] Starting manual folder scan...")
        #processed = self.gps_cache.update_from_folder(self.selected_folder)
        print("Use the 'Globe' tab to trigger background scanning")
        print(f"[TEST] Scan complete: {processed} files processed")
        # Check again
        print("\n[TEST] After scan - database contents:")
        conn = sqlite3.connect(self.gps_cache.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM gps_points")
        count = cursor.fetchone()[0]
        print(f"[TEST] Database now has {count} entries")
        conn.close()
        # Try to display on globe
        if hasattr(self, "globe_widget"):
            print("\n[TEST] Updating globe...")
            points = self.gps_cache.get_all_points(limit=1000)
            print(f"[TEST] Sending {len(points)} points to globe")
            self.globe_widget.set_points(points, use_clustering=False)
        print("\n" + "="*60)
        print("[TEST] Test complete - check console output above")
        print("="*60 + "\n")
        QMessageBox.information(
            self, 
            "GPS Test Complete", 
            f"Found {count} GPS points in database.\n"
            f"Check console for detailed output."
        )

    def _on_gps_batch_found(self, new_points: list):
        """Progressive update - adds points without camera reset"""
        if not new_points:
            return

        print(f"[GPS Update] Batch: +{len(new_points)} points")

        is_globe_visible = (
            hasattr(self, "left_stack")
            and hasattr(self, "globe_widget") 
            and self.left_stack.currentWidget() == self.globe_widget
            and self.globe_widget.view is not None
        )

        if not is_globe_visible:
            print("[GPS Update] Globe not visible - caching for later")
            self.statusBar().showMessage(f"GPS scanning in background...", 1000)
            return

        # **IMPROVED**: Use incremental update if already loaded
        if hasattr(self.globe_widget, '_is_loaded') and self.globe_widget._is_loaded:
            print(f"[GPS Update] Adding {len(new_points)} points incrementally (no camera reset)")
            self.globe_widget.add_batch_js(new_points)
            all_points = self.gps_cache.get_all_points(limit=50000)
            self.statusBar().showMessage(f"Map: {len(all_points)} locations", 1500)
        else:
            # First load: full render with all cached points
            all_points = self.gps_cache.get_all_points(limit=50000)
            print(f"[GPS Update] Initial render with {len(all_points)} points")
            self.globe_widget.set_points(all_points, use_clustering=len(all_points) > 100)

    def _start_background_gps_index(self):
        """Start background thread with progressive updates"""
        if self.gps_indexer_thread and self.gps_indexer_thread.isRunning():
            print("[DEBUG] GPS indexer already running, skipping")
            return

        if not self.selected_folder:
            print("[DEBUG] No selected folder, cannot start GPS indexer")
            return

        print(f"[DEBUG] Starting progressive GPS indexer for: {self.selected_folder}")

        self.gps_indexer_thread = GPSIndexerThread(self.selected_folder, self)

        # **KEY FIX**: Connect signals that will work regardless of current view
        self.gps_indexer_thread.intermediate_points.connect(self._on_gps_batch_found)
        self.gps_indexer_thread.progress.connect(self._on_gps_index_progress)
        self.gps_indexer_thread.finished_with_count.connect(self._on_gps_index_done)
        self.gps_indexer_thread.error.connect(lambda e: print(f"GPS indexing error: {e}"))

        self.gps_indexer_thread.start()

    def _on_gps_index_progress(self, count):
        if count % 100 == 0:  # Report every 100 files
            self.statusBar().showMessage(f"Scanning GPS data: {count} files processed...", 1000)
            print(f"[DEBUG] GPS index progress: {count}")

    def _on_gps_index_done(self, count):
        """Final update when indexing completes"""
        print(f"[DEBUG] GPS indexing complete: {count} files total")

        # **KEY FIX**: Only do final refresh if globe is visible
        if hasattr(self, "globe_toggle_btn") and self.globe_toggle_btn.isChecked():
            points = self.gps_cache.get_all_points(limit=50000) if self.gps_cache else []
            print(f"[DEBUG] Final globe update: {len(points)} points")

            # Force full re-render on completion (cleanup any issues)
            self.globe_widget.set_points(points, use_clustering=len(points) > 100)

        self.statusBar().showMessage(f"âœ… GPS scan complete: {count} locations indexed", 5000)   

    def _collect_gps_points(self, folder: str):
        """
        Walk the entire library tree under `folder`, skipping _incoming etc,
        and use organizer.run_exiftool_tree + organizer.extract_latlon
        to collect one point per media file with GPS.
        """
        return collect_gps_points(folder)

def main():
    app = QApplication(sys.argv)
    # Optional: consistent look on some Windows setups
    QApplication.setStyle(QStyleFactory.create("Fusion"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()