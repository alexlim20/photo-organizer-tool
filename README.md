# 📸 Travel Photo Organizer

A simple Python tool that automatically organizes your **photos and videos** by **date → country → city** using EXIF metadata.
Perfect for travelers, photographers, and anyone with messy photo folders!

---

## 🌟 Features

* 📂 Automatically groups media into folders by **year-month** and **country**
* 🗺️ Detects GPS location from EXIF data (works offline — no API key)
* 🧠 Reads metadata from both **images** and **videos**
* 🎞️ Handles iPhone `.AAE` edits automatically
* 💾 Keeps a CSV log of every file moved
* 🪄 Works with a simple **double-click** on Windows

---

## ⚙️ Requirements

Before running:

1. **Python 3.10+** installed
   → [Download here](https://www.python.org/downloads/)
   ✅ During installation, **check “Add Python to PATH”**
2. **ExifTool** downloaded from [exiftool.org](https://exiftool.org/)

   * Save `exiftool.exe` or `exiftool(-k).exe` into the same folder as `main.py`
---

## 🧰 How to Run (Windows)

### 1️⃣ Download or Clone the Repository

```bash
git clone https://github.com/alexlim20/photo-organizer.git
```

or download it as ZIP and extract.

### 2️⃣ Add Your Photos

After extracting, you’ll see this folder structure:

```
travel-photo-organizer/
│
├── main.py
├── run_organizer.bat
├── requirements.txt
└── photos_to_organize/
```

➡️ Put all your photos and videos inside the folder **`photos_to_organize`**.

### 3️⃣ Run the Organizer

Just double-click **`run_organizer.bat`**

It will:

* Install any missing dependencies automatically
* Organize all your photos/videos
* Save a log file in your folder

When it says:

```
Press Enter to exit.../Press Key to continue...
```

press **Enter**, and the window will close automatically.

---

## 🗂️ Output Example

After running, you’ll find something like this:

```
photos_to_organize/
│
├── 2024-09/
│   ├── country_Japan/
│   │   ├── IMG_001.jpg
│   │   ├── video/
│   │   │   └── VID_002.mp4
│   └── country_Germany/
│       └── IMG_050.jpg
│
├── no_date/
│   └── IMG_999.jpg
│
└── organize_fast_log.csv
```

---

## 🪛 If You Want to Run Manually

You can also run it from terminal:

```bash
pip install -r requirements.txt
python main.py
```

---

## ⚠️ Notes

* The script **moves** files (not copies) — make a backup if needed.
* If a photo has no GPS or date data, it will go to `no_date/`.
* Works completely offline — reverse geocoding uses local data.
* On macOS/Linux, replace `exiftool.exe` with `exiftool`.

---

## 💡 Contributing

Pull requests, ideas, or feedback are welcome!
If you find a bug, open an issue or message me.

---

Author: **Alex**
