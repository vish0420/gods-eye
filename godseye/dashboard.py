from __future__ import annotations

from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox

def launch_dashboard() -> Path | None:
    footage = _discover_footage()
    if not footage:
        raise FileNotFoundError("No footage found in datasets/video.")

    return _open_dashboard(footage)


def _discover_footage() -> list[Path]:
    dataset_dir = Path("datasets/video")
    if not dataset_dir.exists():
        return []
    return sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        ],
        key=lambda path: path.name.lower(),
    )


def _open_dashboard(footage: list[Path]) -> Path | None:
    root = tk.Tk()
    root.title("God's Eye Dashboard")
    root.geometry("820x560")
    root.minsize(720, 480)
    root.configure(bg="#101418")

    selected_index: list[int | None] = [None]

    header = tk.Label(
        root,
        text="Select footage from datasets/video",
        fg="#f2f2f2",
        bg="#101418",
        font=("Segoe UI", 18, "bold"),
    )
    header.pack(pady=(18, 8))

    subtitle = tk.Label(
        root,
        text=(
            "Guidance:\n"
            "1. Keep footage inside datasets/video.\n"
            "2. Click or double-click one file to start.\n"
            "3. The dashboard will close before analysis begins.\n"
            "4. Watch the live video window during analysis.\n"
            "5. Wait for the player to close before the next file starts."
        ),
        fg="#b9c2cc",
        bg="#101418",
        wraplength=760,
        justify="left",
        font=("Segoe UI", 10),
    )
    subtitle.pack(pady=(0, 12))

    frame = tk.Frame(root, bg="#101418")
    frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    list_frame = tk.Frame(frame, bg="#101418")
    list_frame.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar.set,
        selectmode=tk.SINGLE,
        activestyle="none",
        bg="#18212b",
        fg="#f4f7fb",
        selectbackground="#2d7dd2",
        selectforeground="#ffffff",
        highlightthickness=1,
        highlightbackground="#2f3a46",
        relief="flat",
        font=("Segoe UI", 11),
    )
    for path in footage:
        listbox.insert(tk.END, path.name)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    controls = tk.Frame(root, bg="#101418")
    controls.pack(fill="x", padx=16, pady=(0, 16))

    status = tk.StringVar(value=f"{len(footage)} footage file(s) ready.")
    status_label = tk.Label(
        controls,
        textvariable=status,
        fg="#b9c2cc",
        bg="#101418",
        anchor="w",
        font=("Segoe UI", 10),
    )
    status_label.pack(side="left", fill="x", expand=True)

    def start_from_selected(_: object | None = None) -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("God's Eye", "Select one footage file first.")
            return
        selected_index[0] = int(selection[0])
        status.set(f"Launching analysis from {footage[selected_index[0]].name} ...")
        root.quit()

    analyze_button = tk.Button(
        controls,
        text="Analyze From Selected",
        command=start_from_selected,
        bg="#2d7dd2",
        fg="#ffffff",
        activebackground="#1f63ae",
        activeforeground="#ffffff",
        relief="flat",
        padx=16,
        pady=8,
        font=("Segoe UI", 10, "bold"),
    )
    analyze_button.pack(side="right")

    def on_close() -> None:
        selected_index[0] = None
        root.quit()

    listbox.bind("<Double-Button-1>", start_from_selected)
    root.protocol("WM_DELETE_WINDOW", on_close)

    if footage:
        listbox.selection_set(0)
        listbox.activate(0)
        listbox.see(0)

    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass

    if selected_index[0] is None:
        return None
    return footage[selected_index[0]]


def _safe_stem(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
    cleaned = cleaned.strip("_")
    return cleaned or "video"
