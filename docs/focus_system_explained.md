# VyuhaAI Focus System — How It Works

This document explains the focus scoring system used in VyuhaAI Image Viewer.
Written so anyone — engineer, manager, or customer — can understand the design choices.

---

## The Problem We Solved

Most focus tools do this:

> "Find the sharpest cell in this image. Call that 100%. Score every other cell relative to it."

This is **wrong for industrial inspection**. Here is why:

If you give the tool a completely blurry image, the sharpest cell in that image becomes 100%.
The tool shows a green grid — all passing — even though the image is garbage.
An expert looks at this and immediately loses trust in the tool. **One mistake, tool is gone.**

We needed scoring that tells the truth: *is this image actually sharp enough for defect detection?*

---

## The Science Behind It

We based the focus metrics on peer-reviewed research:

**Pertuz, S. et al. (2013). "Analysis of focus measure operators for shape-from-focus."
Pattern Recognition, 46(5), 1415–1432.**

This paper benchmarked 36 different focus measurement algorithms on real images.
Three came out on top:

| Metric | Formula | Why We Use It |
|--------|---------|---------------|
| **Laplacian Variance** | Variance of ∇²I | Best at low noise. Most used in industrial cameras. #2 overall in Pertuz. |
| **Tenengrad** | Σ(Gx² + Gy²) | Best noise robustness. #1 overall in Pertuz. |
| **Brenner Gradient** | Σ(I[x+2]−I[x])² | Fastest. Good cross-check on high-SNR cameras. |

We use **Laplacian + Tenengrad fused** (55% / 45%) for the main score.
Brenner is shown as a third verification number.

Why two metrics? Because Laplacian and Tenengrad have different sensitivity profiles.
If they **agree** on the verdict → HIGH confidence.
If they **disagree** by one category → MEDIUM confidence.
If they disagree by two or more → LOW confidence. Expert must verify manually.

---

## Three Scoring Modes

### 1. RELATIVE mode (⚠ warning shown)

No reference image. Score = this cell ÷ best cell in this same image × 100%.

- Good for: seeing where the lens is soft vs sharp within one image (field curvature, tilt)
- Bad for: pass/fail decisions. A blurry image scores 100% in its best cell.
- **Do not use this for production inspection.**

### 2. AUTO-REF mode

The app automatically tracks the sharpest image you have opened this session.
That image becomes the reference. Score = this cell ÷ same cell in session-best × 100%.

- Better than RELATIVE — at least you're comparing images to each other
- Still session-dependent — each time you restart, reference resets
- Only a non-blurry image can become AUTO-REF (minimum quality threshold enforced)

### 3. LOCKED REF mode (✓ shown — production grade)

You deliberately capture or select your best known-good image and click **Lock Ref**.
The app saves this reference to disk. It survives restarts.

Score = this cell ÷ same cell in locked reference × 100%.

- 100% means "as sharp as your calibrated golden sample"
- 60% means "only 60% as sharp as your reference — check lens or lighting"
- This is how professional machine vision tools work (Cognex, Keyence, Halcon)
- **This is the mode to use for production inspection**

---

## The Grid

The image is divided into an 8×8 grid (64 cells).
Each cell is scored independently.

Color coding:
- **GREEN** — ≥ 72% — sharp, good for defect detection
- **AMBER** — 38–72% — soft, marginal, review lens settings
- **RED** — < 38% — blurry, reject this image, do not use for inspection

Why per-cell? Because a lens may be sharp in the center but soft at the edges.
This is called **field curvature** — a real optical problem common in industrial lenses.
The grid catches it immediately. The Focus Assist panel also shows tilt direction
(e.g. "sharper on LEFT", "sharper at BOTTOM").

---

## Confidence Levels

| Level | When | What It Means |
|-------|------|---------------|
| **HIGH** | LOCKED REF + Laplacian and Tenengrad agree on verdict | Fully trustworthy. Use for automated pass/fail. |
| **MEDIUM** | AUTO-REF, or metrics are one category apart | Reliable for manual review. Not for unattended automation. |
| **LOW** | RELATIVE mode only | Spatial variation only. Cannot confirm absolute sharpness. |

---

## Raw Numbers — Always Visible

The inspector always shows three raw numbers:

```
Lap: 1,840  |  Ten: 45,230  |  Bren: 12.1
```

These are the unscaled metric values directly from the algorithms.
An expert can verify or challenge any verdict by looking at these directly.
We never hide the math.

---

## How to Use It — Step by Step

### First time setup (one-time, 5 minutes)

1. Mount your camera and part at the correct working distance
2. Capture the sharpest image you can — turn the lens focus ring until the image is crisp
3. Open that image in the app
4. Click **Lock Ref** in the Focus Assist panel (top right when in Focus mode)
5. The app saves this reference. Done. You never need to do this again unless the setup changes.

### Every inspection session

1. Open the image (or folder of images)
2. Switch to **Focus mode** (button at top)
3. The grid shows each cell as % of your locked reference
4. GREEN cells → sharp, proceed to defect inspection
5. AMBER/RED cells → fix lens focus, lighting, or vibration before inspecting
6. The Focus Assist panel shows verdict, confidence, and specific action to take

### When the reference image is shown

If you open the exact same image that was locked as reference, the app shows:

> ⚠ THIS IS THE REFERENCE IMAGE — comparing to itself — 100% is meaningless

This is intentional. The 100% grid is correct math but useless information.
The absolute Score and Verdict at the top are what matter for that image.

---

## What the Verdicts Mean

| Verdict | Absolute Score | Meaning |
|---------|---------------|---------|
| **PERFECT** | ≥ 700 | Excellent. Lock this as reference if you don't have one. |
| **GOOD** | 400–699 | Usable for all defect types including fine scratches. |
| **SOFT** | 200–399 | Marginal. OK for large defects. Risk of missing hairline cracks. |
| **BLURRY** | < 200 | Reject. Do not use for inspection. Fix focus before proceeding. |

The absolute score uses Laplacian Variance ÷ 5, calibrated to a 0–1000 scale.
These thresholds are configurable in the app config file if your camera requires adjustment.

---

## Why This Design

The guiding rule during development:

> *"This tool will be used by experts who know everything. Any mistake they can find immediately. One mistake — the tool is gone."*

Every design decision follows from that:
- Never normalize away a bad image to look good
- Always show raw numbers — expert can verify
- Always show confidence — expert knows when to trust vs verify manually
- Always show scoring mode — expert knows what kind of comparison is being made
- Self-reference warning — expert is not misled by trivial 100% scores
- Blurry images blocked from becoming reference — garbage in, garbage out prevented
