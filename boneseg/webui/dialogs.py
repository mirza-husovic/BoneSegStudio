"""Native file / folder dialogs for the local desktop app.

A browser page can never learn the real path of a picked file, but this
server runs on the user's own machine — so the server opens the regular
Windows dialog itself (tkinter) and hands the chosen paths back.

Each dialog runs on its own short-lived Tk root inside the calling worker
thread (Tk objects must stay on the thread that made them), forced on top
of the browser window. One dialog at a time.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()

FILETYPES = {
    "graves": [("Grave photos and exports",
                "*.jpg *.jpeg *.png *.tif *.tiff *.dxf *.geojson"),
               ("Photos", "*.jpg *.jpeg *.png *.tif *.tiff"),
               ("Exports (DXF / GeoJSON)", "*.dxf *.geojson"),
               ("All files", "*.*")],
    "site": [("Site plan", "*.dxf *.gpkg *.pdf"), ("All files", "*.*")],
}


def ask(kind: str, title: str = "", initial: str = "", filetypes: str = "graves") -> list[str]:
    """kind: 'files' (multi-select), 'folder', 'save'. Returns [] on cancel."""
    if not _lock.acquire(blocking=False):
        raise RuntimeError("Another file dialog is already open.")
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        opts = {"parent": root, "title": title or None}
        if initial:
            opts["initialdir"] = initial
        try:
            if kind == "files":
                res = filedialog.askopenfilenames(filetypes=FILETYPES.get(filetypes), **opts)
                return [str(p) for p in res] if res else []
            if kind == "folder":
                res = filedialog.askdirectory(mustexist=True, **opts)
                return [str(res)] if res else []
            if kind == "save":
                res = filedialog.asksaveasfilename(filetypes=FILETYPES.get(filetypes),
                                                   initialfile="site_plan", **opts)
                return [str(res)] if res else []
            raise ValueError(f"unknown dialog kind: {kind}")
        finally:
            root.destroy()
    finally:
        _lock.release()
