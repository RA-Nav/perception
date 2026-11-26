#!/usr/bin/env python3
"""
label_and_finetune.py

Small GUI to label real photos of the warehouse exits with bounding boxes and fine-tune the exit sign detector on
them with one button.

  <images>                         folder of real photos (jpg/png), searched non-recursively
  <out>/labels/NAME.txt            YOLO labels (class x_center y_center width height, normalized), written on every
                                   change; an empty file = reviewed image without signs (a hard negative)
  <out>/dataset/                   built by the Fine-tune button: images/{train,val}, labels/{train,val}, data.yaml

Only reviewed images (a label file exists) are used. The split is by contiguous blocks of the sorted file names
(the last --val-frac go to val), not random: neighbouring frames of one video are near duplicates, a random split
would leak them into val and the metrics would look perfect for nothing.

Keys: drag = new box, click a box = select, Del = remove selected, 1/2/3 = class (of the selected box, or of the next
one), A / Left = previous, D / Right = next, N = next unreviewed, Ctrl+Z = undo last box, Esc = unselect.
A box drawn with the left button is added with the current class; reviewing an image with no boxes (D without drawing)
marks it as a negative.

Fine-tune: runs training/train.py in a subprocess (log in the window) from --weights, ONNX at 736x1280 to --export.
--synthetic <data.yaml> adds the synthetic train set (make_synthetic.py) so the model does not forget the other signs;
val stays real only.

Usage (.venv python of the workspace):
  python training/label_and_finetune.py ~/ranav_ws/datasets/exit_sign/real_photos
  python training/label_and_finetune.py <photos> --build-only      # only (re)build <out>/dataset, print stats
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from exit_inspection.sign_art import CLASSES  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = ["#00e676", "#ffea00", "#40c4ff", "#ff6e40"]
HANDLE = 4


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT))


def label_path(out, img):
    return os.path.join(out, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")


def read_boxes(path, w, h):
    """YOLO txt -> [[cls, x1, y1, x2, y2]] in pixels."""
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                boxes.append([c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return boxes


def write_boxes(path, boxes, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def save_scaled(src, dst, max_side):
    """Copy as jpg with the long side <= max_side (labels are normalized, so they stay valid). Phone photos are
    24 MP: decoding them every epoch would starve the GPU, and the detector sees at most 1280 px anyway."""
    from PIL import Image

    im = Image.open(src).convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.LANCZOS)
    im.save(dst, quality=95)


def build_dataset(images, out, val_frac=0.2, synthetic=None, max_side=1920):
    """Copy the reviewed images to <out>/dataset and write data.yaml. Returns (n_train, n_val, n_boxes, yaml_path)."""
    import yaml

    reviewed = [p for p in images if os.path.exists(label_path(out, p))]
    if len(reviewed) < 2:
        raise RuntimeError(f"need at least 2 reviewed images, have {len(reviewed)}")
    n_val = min(max(1, round(len(reviewed) * val_frac)), len(reviewed) - 1)
    split = {"train": reviewed[:-n_val], "val": reviewed[-n_val:]}

    ds = os.path.join(out, "dataset")
    shutil.rmtree(ds, ignore_errors=True)
    n_boxes = 0
    for name, files in split.items():
        os.makedirs(os.path.join(ds, "images", name))
        os.makedirs(os.path.join(ds, "labels", name))
        for p in files:
            base = os.path.basename(p)
            save_scaled(p, os.path.join(ds, "images", name, os.path.splitext(base)[0] + ".jpg"), max_side)
            lab = label_path(out, p)
            shutil.copy(lab, os.path.join(ds, "labels", name, os.path.splitext(base)[0] + ".txt"))
            n_boxes += sum(1 for _ in open(lab) if _.strip())
    train = os.path.join(ds, "images", "train")
    if synthetic:
        syn = yaml.safe_load(open(synthetic))
        root = syn.get("path", os.path.dirname(synthetic))
        s = syn["train"] if os.path.isabs(syn["train"]) else os.path.join(root, syn["train"])
        train = [train, s]
    yml = os.path.join(ds, "data.yaml")
    with open(yml, "w") as f:
        yaml.safe_dump({"path": ds, "train": train, "val": os.path.join(ds, "images", "val"),
                        "names": {i: n for i, n in enumerate(CLASSES)}}, f, sort_keys=False)
    return len(split["train"]), len(split["val"]), n_boxes, yml


class App:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import messagebox, ttk

        from PIL import Image, ImageTk

        self.tk, self.ttk, self.messagebox, self.Image, self.ImageTk = tk, ttk, messagebox, Image, ImageTk
        self.root, self.args = root, args
        self.out = os.path.expanduser(args.out)
        self.images = list_images(os.path.expanduser(args.images))
        if not self.images:
            raise SystemExit(f"no images in {args.images}")
        self.idx = 0
        self.boxes, self.sel, self.cls = [], None, 0
        self.drag = None  # (x0, y0) image px of the box being drawn
        self.training = False
        self.photo = None
        self.size = (1, 1)

        root.title("Exit sign labeling")
        root.geometry("1400x900")
        top = ttk.Frame(root)
        top.pack(fill="x", padx=6, pady=4)
        self.cls_var = tk.IntVar(value=0)
        for i, n in enumerate(CLASSES):
            ttk.Radiobutton(top, text=f"{i + 1} {n}", value=i, variable=self.cls_var,
                            command=self.on_class).pack(side="left", padx=4)
        ttk.Button(top, text="< prev", command=lambda: self.go(-1)).pack(side="left", padx=(20, 2))
        ttk.Button(top, text="next >", command=lambda: self.go(1)).pack(side="left", padx=2)
        ttk.Button(top, text="next unreviewed", command=self.next_unreviewed).pack(side="left", padx=2)
        self.info = ttk.Label(top, text="")
        self.info.pack(side="left", padx=12)

        self.epochs = tk.StringVar(value=str(args.epochs))
        self.btn = ttk.Button(top, text="Fine-tune", command=self.start_training)
        self.btn.pack(side="right", padx=4)
        ttk.Entry(top, textvariable=self.epochs, width=5).pack(side="right")
        ttk.Label(top, text="epochs").pack(side="right", padx=(0, 2))

        self.canvas = tk.Canvas(root, bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=6)
        self.log = tk.Text(root, height=8, bg="#111", fg="#ddd", state="disabled")
        self.log.pack(fill="x", padx=6, pady=4)

        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Configure>", lambda e: self.redraw())
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load(0)

    # ---- image / labels -------------------------------------------------------------------------------------
    def load(self, idx):
        self.idx = idx
        path = self.images[idx]
        self.img = self.Image.open(path).convert("RGB")
        self.size = self.img.size
        k = min(1.0, 2000 / max(self.size))
        self.preview = self.img.resize((round(self.size[0] * k), round(self.size[1] * k)), self.Image.BILINEAR)
        self.boxes = read_boxes(label_path(self.out, path), *self.size)
        self.sel, self.drag = None, None
        self.redraw()

    def save(self):
        write_boxes(label_path(self.out, self.images[self.idx]), self.boxes, *self.size)

    def go(self, step):
        self.save()  # leaving an image reviews it: no boxes = hard negative
        self.load(max(0, min(len(self.images) - 1, self.idx + step)))

    def next_unreviewed(self):
        self.save()
        n = len(self.images)
        for k in range(1, n + 1):
            i = (self.idx + k) % n
            if not os.path.exists(label_path(self.out, self.images[i])):
                return self.load(i)
        self.messagebox.showinfo("Done", "all images reviewed")
        self.load(self.idx)

    def n_reviewed(self):
        return sum(os.path.exists(label_path(self.out, p)) for p in self.images)

    # ---- drawing --------------------------------------------------------------------------------------------
    def geom(self):
        cw, ch = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        w, h = self.size
        s = min(cw / w, ch / h)
        return s, (cw - w * s) / 2, (ch - h * s) / 2

    def to_img(self, ex, ey):
        s, ox, oy = self.geom()
        return (min(max((ex - ox) / s, 0), self.size[0]), min(max((ey - oy) / s, 0), self.size[1]))

    def redraw(self):
        if not hasattr(self, "img"):
            return
        c = self.canvas
        c.delete("all")
        s, ox, oy = self.geom()
        w, h = self.size
        view = self.preview.resize((max(1, int(w * s)), max(1, int(h * s))))
        self.photo = self.ImageTk.PhotoImage(view)
        c.create_image(ox, oy, anchor="nw", image=self.photo)
        for i, (k, x1, y1, x2, y2) in enumerate(self.boxes):
            col = COLORS[k % len(COLORS)]
            c.create_rectangle(ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s, outline=col,
                               width=3 if i == self.sel else 2)
            c.create_text(ox + x1 * s + 3, oy + y1 * s - 8, anchor="w", fill=col, text=CLASSES[k])
        reviewed = os.path.exists(label_path(self.out, self.images[self.idx]))
        self.info.config(text=f"{self.idx + 1}/{len(self.images)}  {os.path.basename(self.images[self.idx])}  "
                              f"boxes {len(self.boxes)}  {'reviewed' if reviewed else 'NEW'}  "
                              f"[{self.n_reviewed()} reviewed]")

    def hit(self, x, y):
        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def on_press(self, e):
        self.root.focus_set()
        self.press = self.to_img(e.x, e.y)
        self.drag = None
        self.moved = False

    def on_motion(self, e):
        x, y = self.to_img(e.x, e.y)
        x0, y0 = self.press
        if not self.moved and (abs(x - x0) + abs(y - y0)) * self.geom()[0] < 4:
            return
        self.moved = True
        s, ox, oy = self.geom()
        self.canvas.delete("rubber")
        self.canvas.create_rectangle(ox + x0 * s, oy + y0 * s, ox + x * s, oy + y * s, outline="white", dash=(4, 3),
                                     tags="rubber")

    def on_release(self, e):
        x, y = self.to_img(e.x, e.y)
        x0, y0 = self.press
        if self.moved:
            if abs(x - x0) > 2 and abs(y - y0) > 2:
                self.boxes.append([self.cls, min(x, x0), min(y, y0), max(x, x0), max(y, y0)])
                self.sel = len(self.boxes) - 1
                self.save()
        else:
            self.sel = self.hit(x, y)
            if self.sel is not None:
                self.cls = self.boxes[self.sel][0]
                self.cls_var.set(self.cls)
        self.redraw()

    def on_class(self):
        self.cls = self.cls_var.get()
        if self.sel is not None:
            self.boxes[self.sel][0] = self.cls
            self.save()
            self.redraw()

    def on_key(self, e):
        if isinstance(self.root.focus_get(), self.tk.Entry) or isinstance(self.root.focus_get(), self.ttk.Entry):
            return
        k = e.keysym
        if k in ("1", "2", "3", "4") and int(k) <= len(CLASSES):
            self.cls_var.set(int(k) - 1)
            self.on_class()
        elif k in ("a", "A", "Left"):
            self.go(-1)
        elif k in ("d", "D", "Right"):
            self.go(1)
        elif k in ("n", "N"):
            self.next_unreviewed()
        elif k in ("Delete", "BackSpace") and self.sel is not None:
            del self.boxes[self.sel]
            self.sel = None
            self.save()
            self.redraw()
        elif k == "Escape":
            self.sel = None
            self.redraw()
        elif k in ("z", "Z") and e.state & 0x4 and self.boxes:  # Ctrl+Z
            self.boxes.pop()
            self.sel = None
            self.save()
            self.redraw()

    # ---- training -------------------------------------------------------------------------------------------
    def say(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def start_training(self):
        if self.training:
            return
        self.save()
        try:
            epochs = int(self.epochs.get())
            n_train, n_val, n_boxes, yml = build_dataset(self.images, self.out, self.args.val_frac,
                                                         self.args.synthetic)
        except (ValueError, RuntimeError) as err:
            self.messagebox.showerror("Cannot train", str(err))
            return
        self.say(f"dataset: {n_train} train, {n_val} val, {n_boxes} boxes -> {yml}\n")
        cmd = [sys.executable, os.path.join(PKG, "training", "train.py"), "--data", yml, "--epochs", str(epochs),
               "--weights", os.path.expanduser(self.args.weights), "--export", os.path.expanduser(self.args.export),
               "--project", os.path.join(self.out, "runs"), "--name", "real"]
        self.say("$ " + " ".join(cmd) + "\n")
        self.training = True
        self.btn.config(state="disabled", text="Training...")
        threading.Thread(target=self.run, args=(cmd,), daemon=True).start()

    def run(self, cmd):
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                # tqdm redraws with \r: keep only the last chunk of each line
                self.root.after(0, self.say, line.split("\r")[-1].rstrip("\n") + "\n")
            code = p.wait()
        except OSError as err:
            self.root.after(0, self.say, f"cannot start training: {err}\n")
            code = -1
        self.root.after(0, self.done, code)

    def done(self, code):
        self.training = False
        self.btn.config(state="normal", text="Fine-tune")
        self.say(f"\ntraining {'finished' if code == 0 else f'FAILED (exit {code})'}\n")
        if code == 0:
            self.messagebox.showinfo("Fine-tune", f"ONNX: {os.path.expanduser(self.args.export)}")

    def on_close(self):
        if self.training and not self.messagebox.askyesno("Quit", "Training is running. Quit anyway?"):
            return
        self.save()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder with the real photos")
    ap.add_argument("--out", default=os.path.expanduser("~/ranav_ws/datasets/exit_sign/real"))
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(PKG), "camera_perception",
                                                      "weights", "yolo26n.pt"),
                    help="starting weights; pass a previous best.pt to continue from an earlier fine-tune")
    ap.add_argument("--synthetic", help="data.yaml of make_synthetic.py, its train set is mixed in")
    ap.add_argument("--export", default=os.path.expanduser("~/ranav_ws/models/exit_sign/exit_sign_real_736x1280.onnx"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--build-only", action="store_true", help="no GUI: build the dataset and print its stats")
    args = ap.parse_args()

    if args.build_only:
        imgs = list_images(os.path.expanduser(args.images))
        n_tr, n_va, n_box, yml = build_dataset(imgs, os.path.expanduser(args.out), args.val_frac, args.synthetic)
        print(f"{n_tr} train, {n_va} val, {n_box} boxes -> {yml}")
        return

    import tkinter as tk
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
