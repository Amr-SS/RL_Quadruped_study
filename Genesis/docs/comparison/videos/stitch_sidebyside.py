"""
Stitch the three per-algorithm evaluation clips into a single labelled
side-by-side comparison video:  [ PPO | SAC | TD3 ].

Pure Python — uses imageio (bundled ffmpeg) + Pillow, no system ffmpeg needed.
All three inputs are recorded under identical conditions by record_eval.py
(same env, same forward command, same camera, same duration, same fps), so the
panels are directly comparable.

Usage:
  python docs/comparison/videos/stitch_sidebyside.py
"""

import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

PANEL_W, PANEL_H = 640, 360          # each video panel
HEADER_H = 64                         # label band above each panel
FPS = 50

# Labels are deliberately neutral and factual. In this deterministic, no-push
# evaluation all three final policies hold balance (0 falls measured), so we do
# NOT label them "stable/unstable/collapse" here — that distinction lives in the
# training curves (fig_peak_vs_final.png), not this clip. Subtitle = policy family.
PANELS = [
    ("raw_ppo.mp4", "PPO", "on-policy"),
    ("raw_sac.mp4", "SAC", "off-policy"),
    ("raw_td3.mp4", "TD3", "off-policy"),
]
OUT = os.path.join(HERE, "ppo_sac_td3_comparison.mp4")


def _font(size, bold=True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE = _font(34, bold=True)
F_SUB = _font(18, bold=False)


def make_header(label, sublabel):
    """Render one panel's header band (black with centered white text)."""
    img = Image.new("RGB", (PANEL_W, HEADER_H), (17, 22, 29))  # GitHub-dark slate
    d = ImageDraw.Draw(img)
    # title
    tb = d.textbbox((0, 0), label, font=F_TITLE)
    d.text(((PANEL_W - (tb[2] - tb[0])) / 2, 4), label, font=F_TITLE, fill=(255, 255, 255))
    # subtitle
    sb = d.textbbox((0, 0), sublabel, font=F_SUB)
    d.text(((PANEL_W - (sb[2] - sb[0])) / 2, 40), sublabel, font=F_SUB, fill=(150, 200, 255))
    return np.asarray(img)


def resize_frame(frame):
    return np.asarray(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.BILINEAR))


def main():
    readers = [imageio.get_reader(os.path.join(HERE, fn)) for fn, _, _ in PANELS]
    headers = [make_header(lbl, sub) for _, lbl, sub in PANELS]

    # Pre-render the static header row once
    header_row = np.concatenate(headers, axis=1)               # (HEADER_H, 3*PANEL_W, 3)
    divider = np.full((PANEL_H, 4, 3), 40, dtype=np.uint8)     # thin gray gap between panels

    writer = imageio.get_writer(OUT, fps=FPS, quality=8, macro_block_size=8)

    iters = [iter(r) for r in readers]
    n = 0
    while True:
        panels = []
        try:
            for it in iters:
                panels.append(resize_frame(next(it)))
        except StopIteration:
            break
        # horizontal concat of video panels with dividers
        video_row = panels[0]
        for p in panels[1:]:
            video_row = np.concatenate([video_row, divider, p], axis=1)
        # pad header row to match width (account for dividers)
        if header_row.shape[1] != video_row.shape[1]:
            pad = video_row.shape[1] - header_row.shape[1]
            header_row_p = np.concatenate(
                [header_row, np.full((HEADER_H, pad, 3), 17, dtype=np.uint8)], axis=1)
        else:
            header_row_p = header_row
        full = np.concatenate([header_row_p, video_row], axis=0)
        writer.append_data(full)
        n += 1

    writer.close()
    for r in readers:
        r.close()
    print(f"wrote {n} frames -> {OUT}  ({full.shape[1]}x{full.shape[0]})")


if __name__ == "__main__":
    main()
