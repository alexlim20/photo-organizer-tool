# ui_app_qt.py
import os, sys
import main as organizer
import hashlib, json
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "error")
import cv2
from PySide6.QtCore import (
    QDir, Qt, QSize, QObject, QThread, 
    Signal, QTimer, QEvent, QStandardPaths,
    QRunnable, QThreadPool, QItemSelectionModel, QModelIndex,
    QMutex, QMutexLocker
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QListView,
    QAbstractItemView, QStatusBar, QStyleFactory, QMessageBox,
    QFileSystemModel, QListView, QSplitterHandle, QFrame, QFileIconProvider
)
from PySide6.QtGui import (
    QKeySequence, QImageReader, QPixmap, 
    QIcon, QImage, QShortcut
)
from collections import OrderedDict

from PIL import Image
from pillow_heif import register_heif_opener # enable HEIC/HEIF reading in Pillow

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
        # (debug) print("🗂️ Disk thumbnail cache is at:", self.disk_cache_dir)
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
            # PNG is fine for tiny thumbs; it’s fast & supports alpha
            px.save(disk_p, "PNG")
        except Exception:
            pass
        if len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)
        return True

    def _build_image_sync(self, path: str, size: int, expected_gen: int):
        # gen changed → stop
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
                # ✅ Qt read succeeded — cache it
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
            # stale completion from a previous folder — ignore
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
        # ➜ ONLY show images & videos (folders still visible)
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
        self.loading = QLabel("Loading folder…")
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

        self.back_btn = QPushButton("◀ Back")
        self.back_btn.setFocusPolicy(Qt.NoFocus)
        self.path_now = QLabel("—")
        self.path_now.setStyleSheet("color: #ccc;")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.back_btn, 2)  # 2:8 ratio
        header.addWidget(self.path_now, 8)
        right_v.addLayout(header)
        right_v.addWidget(self.grid, 1)

        # --- LEFT panel: controls beside the file view ---
        # (REPLACES the old `left_placeholder = QWidget()` section)
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

        # Header for the left panel
        self.staging_hdr = QLabel("Staging — _incoming")
        self.staging_hdr.setStyleSheet("font-weight: 600; color: #bbb;")

        controls_row = QHBoxLayout()
        left_v.addWidget(self.staging_hdr)
        left_v.addWidget(self.staging_view, 1)
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setEnabled(False)  # enabled after a folder is chosen
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)

        controls_row.addWidget(self.choose_btn)
        controls_row.addWidget(self.upload_btn)
        controls_row.addWidget(self.path_label, 1)  # expands after Choose Folder
        controls_row.addStretch()                   # pushes Run to the far right (next to splitter)
        controls_row.addWidget(self.run_btn)
        self.upload_btn.clicked.connect(self.on_upload_clicked)

        left_v.addStretch(1)  # keep the row at the top-left; rest stays empty
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
        # Tell the model the decoration may have changed → provider.icon() runs again
        self.model.dataChanged.emit(first, last, [Qt.DecorationRole, Qt.DisplayRole, Qt.SizeHintRole])
        # Nudge layout so the paint happens right away
        self.grid.doItemsLayout()
        # Kick the thumbnail provider’s pending queue
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
        super().resizeEvent(event)
        if self.loading and self.grid:
            self.loading.setGeometry(self.grid.rect())
        # keep the 55/45 ratio and update columns as window changes
        self._schedule_layout()
        root_idx = self.grid.rootIndex()
        root_path = self.model.filePath(root_idx) if root_idx.isValid() else (self.selected_folder or "—")
        self._set_current_path(root_path)
    
    def eventFilter(self, obj, event):
        if obj is self.grid.viewport() and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._schedule_layout)
        return super().eventFilter(obj, event)

    def on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to preview")
        if not folder:
            return

        self.selected_folder = folder
        self.path_label.setText(folder)
        self._nav_stack.clear()
        self._set_current_path(folder)
        self.path_label.setStyleSheet("")
        self.statusBar().showMessage("Folder selected.", 2000)

        incoming_root = os.path.join(self.selected_folder, "_incoming")
        os.makedirs(incoming_root, exist_ok=True)
        incoming_idx = self.staging_model.setRootPath(incoming_root)
        self.staging_view.setRootIndex(incoming_idx)
        self.staging_hdr.setText("Staging — _incoming (no uploads yet)")

        # Show overlay + disable view immediately
        self.loading.setGeometry(self.grid.rect())
        self.loading.setVisible(True)
        self.grid.setEnabled(False)
        QApplication.processEvents()  # let UI update right now

        self._icon_provider.cancel_all_and_clear()   
        self._pending_refresh.clear()  # NEW: Clear pending refresh on new folder

        # Defer the heavy work to the next event loop tick
        QTimer.singleShot(0, lambda: self._load_folder(folder))
        QTimer.singleShot(50, self._schedule_layout)

        # Enable Run now that we have a target
        self.run_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        # --- Staging / resume handling ---
        incoming_root = os.path.join(self.selected_folder, "_incoming")
        os.makedirs(incoming_root, exist_ok=True)

        resume = self._scan_resume_state()
        if resume and resume["total"] > 0 and resume["processed"] > 0:
            # There is a session with partial progress -> show it directly
            stg_idx = self.staging_model.setRootPath(resume["session_dir"])
            self.staging_view.setRootIndex(stg_idx)
            self.staging_hdr.setText(
                f"Staging — {resume['session_name']} "
                f"(resume: {resume['processed']}/{resume['total']} done, "
                f"{resume['remaining']} remaining)"
            )
            self.statusBar().showMessage(
                f"Found unfinished session: {resume['processed']} of {resume['total']} already processed.",
                5000,
            )
        else:
            # No unfinished session; show root incoming as empty staging
            stg_idx = self.staging_model.setRootPath(incoming_root)
            self.staging_view.setRootIndex(stg_idx)
            self.staging_hdr.setText("Staging — _incoming (no uploads yet)")

        # Prepare staging view to watch _incoming (empty until first upload)
        incoming_root = os.path.join(self.selected_folder, "_incoming")
        os.makedirs(incoming_root, exist_ok=True)
        incoming_idx = self.staging_model.setRootPath(incoming_root)
        self.staging_view.setRootIndex(incoming_idx)
        self.staging_hdr.setText("Staging — _incoming (no session)")

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
            self.statusBar().showMessage(f"Running organizer… ({mode_text})")

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
        self.staging_hdr.setText(f"Staging — {os.path.basename(session_dir)}")

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

        self.statusBar().showMessage("Done ✅", 3000)
        self.choose_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

        # Reset staging view back to _incoming (session was likely auto-emptied)
        if self.selected_folder:
            incoming_root = os.path.join(self.selected_folder, "_incoming")
            idx = self.staging_model.setRootPath(incoming_root)
            self.staging_view.setRootIndex(idx)
            self.staging_hdr.setText("Staging — _incoming (empty)")

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
        # mark intent; we’ll satisfy when rows exist
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
            # try again shortly—rows may not be in yet
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
        CHUNK = 200  # keep UI responsive on very large folders

        def kick(start=0):
            if start >= rows:
                return
            end = min(start + CHUNK, rows)
            for r in range(start, end):
                idx = self.model.index(r, 0, root)
                fi = self.model.fileInfo(idx)
                if fi.isDir():
                    continue
                # This is the same path your view uses: asking for an icon schedules/queues a thumb.
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
                # Qt failed or gave null → try Pillow
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

def main():
    app = QApplication(sys.argv)
    # Optional: consistent look on some Windows setups
    QApplication.setStyle(QStyleFactory.create("Fusion"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()