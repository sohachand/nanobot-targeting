# The Real AI Model — Full Explanation (Separate Guide)

This is a separate guide just about the **real AI model** — the actual U-Net that reads real brain MRI scans and finds tumors. This is the only part of the whole project that uses real patient data and a real trained model (everything else — Unity, the simulators — either visualizes this model's output or is a separate toy demo, as explained in `README_CODE.md`).

---

## 1. What it is, in one line

A computer program that looks at a real brain MRI picture and marks, pixel by pixel, which parts are tumor and which parts are normal tissue — by having learned the pattern from thousands of real, doctor-marked example scans.

---

## 2. The real data it was trained on

- Dataset name: **LGG Brain MRI Segmentation dataset** — real, anonymized scans of patients with low-grade brain tumors (gliomas).
- Original source: the Cancer Genome Atlas / Cancer Imaging Archive (TCGA/TCIA), a public medical research archive — packaged for easy use on Kaggle.
- **110 patients total.**
- Each patient's scan is a stack of 2D slices (like pages in a flipbook), and every slice has a matching black-and-white "answer key" image — white pixels mark exactly where a radiologist drew the tumor.
- Split: **88 patients (3,151 slices) used to teach the model**, and **22 patients (778 slices) kept completely hidden**, only used afterward to test it. This split is always done the same way (a fixed setting called `random_state=42`), so the same 88/22 patients are used every time the code is run.

---

## 3. The model itself — U-Net

### What it looks like (the real code)

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = double_conv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))
        u4 = self.up4(bn); u4 = self.upconv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4); u3 = self.upconv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3); u2 = self.upconv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2); u1 = self.upconv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.out(u1))
```

### What this means, step by step, in easy words

**Step 1 — Squinting (the "down" / encoder half):**
`down1` → `down2` → `down3` → `down4`, each followed by `pool` (which cuts the image's width and height in half). Imagine squinting at a photo: you lose fine detail, but you start noticing the big picture — "something unusual is over there." That's exactly what this half does: the picture gets smaller and smaller (so it can't keep every tiny pixel), while the model tracks more and more different kinds of patterns per region (64 → 128 → 256 → 512 different "things it's noticing").

**Step 2 — The bottleneck:**
`bottleneck` is the smallest, most zoomed-out point — where the model has the broadest understanding ("there's definitely something going on here") but the least precise location.

**Step 3 — Opening your eyes back up (the "up" / decoder half):**
`up1` → `up2` → `up3` → `up4`, each one doubling the image size back up. This turns the vague "something's there" idea back into an actual, full-size, pixel-by-pixel picture.

**Step 4 — The key trick: skip connections:**
`torch.cat([u4, d4], dim=1)` (and similarly for u3/d3, u2/d2, u1/d1) — while growing the picture back up, the model also directly copies in what it saw at the matching size on the way down. This combines "the big-picture understanding" with "the original sharp detail," so the final tumor outline is both correctly located AND precisely traced. This is literally why it's called a "U-Net" — picture the letter U: down one side, across the bottom, back up the other side, with lines connecting matching points on both sides.

**Step 5 — The final decision:**
`torch.sigmoid(self.out(u1))` — turns every single pixel into one number between 0 and 1 — "how confident am I this exact pixel is tumor." Anything above 0.5 gets called tumor; below that, normal tissue.

---

## 4. How it actually learns (training)

```python
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.*intersection + smooth) / (pred.sum() + target.sum() + smooth)
```
**In easy words:** this measures "how wrong was the guess" by checking how much the predicted tumor shape overlaps the real tumor shape. Good overlap = low error. This is combined with a second, more standard per-pixel error measure (called BCE) into one combined score.

**The actual training loop:** show the model a training picture → it guesses → compare guess to the real answer using the loss above → calculate which direction to nudge every single internal number to make the error smaller → repeat for every training image → repeat this whole thing 20 times over (20 "epochs"). The optimizer used to decide how big each nudge should be is called Adam — a standard, well-tested choice.

**What actually happened, round by round (the real log):**
- Round 1: training error 1.264, test error 1.188 (both high — it knew nothing yet).
- Round 20 (the last one): training error dropped to 0.174, test error only dropped to about 0.58.
- Around round 9, test error basically stopped improving even though training error kept dropping — a sign called **overfitting**, meaning after round 9 the model mostly got better at memorizing the exact training pictures rather than learning something that generalizes.
- The code just saves whichever round finishes last (round 20), not whichever round scored best (round 19 was very slightly better). Small detail, but good to know.

---

## 5. How good is it, really? (scoring code + real results)

```python
def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    pred = (pred > threshold).astype(np.float32)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = (pred > threshold).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)
```
**In easy words:** both of these just check "how much does the guessed tumor shape overlap the real tumor shape," as one score from 0 to 1 (1 = perfect match). Run across all 22 hidden test patients and averaged:

- **Mean Dice = 0.773** (77.3% average overlap)
- **Mean IoU = 0.746** (74.6% on the other overlap measure)
- **Tumor correctly found in 20 of 22 patients** (completely missed on 2)

---

## 6. The measurement mistake that was found and fixed

The first version of the accuracy-checking code computed precision/recall **per slice**, then averaged those scores. Problem: most slices in a brain scan have **zero tumor** on them, and on those empty slices, the math for precision/recall doesn't have a valid answer (a 0-divided-by-0 situation) — the old code handled this badly, dragging the overall average down unfairly.

**The fix (real code):**
```python
TP = FP = FN = TN = 0
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        preds = (model(images).cpu().numpy() > 0.5).astype(np.uint8)
        masks_np = (masks.numpy() > 0.5).astype(np.uint8)
        for p, m in zip(preds, masks_np):
            p = p[0].flatten()
            m = m[0].flatten()
            TP += int(((p == 1) & (m == 1)).sum())
            FP += int(((p == 1) & (m == 0)).sum())
            FN += int(((p == 0) & (m == 1)).sum())
            TN += int(((p == 0) & (m == 0)).sum())

precision = TP / (TP + FP) if (TP + FP) else 0
recall    = TP / (TP + FN) if (TP + FN) else 0
```
**In easy words:** instead of scoring page-by-page, add up every single correct/wrong pixel count across **all 778 validation slices** first, and only calculate the final score once, from the grand totals.

**Real corrected result:** 351,552 correctly flagged tumor pixels, 49,579 wrongly flagged as tumor, 255,679 missed real tumor pixels, 50,330,198 correctly flagged healthy pixels. This gives:
- **Precision = 0.876** — when it says "tumor," it's right 87.6% of the time (few false alarms).
- **Recall = 0.579** — but it only catches 57.9% of all real tumor pixels that exist (cautious — misses more than it falsely flags).

---

## 7. The reproducibility problem (why re-running gives different numbers)

The code fixes which patients go into the 88/22 split every time (`random_state=42`), but it **never fixes the model's own random starting point** — there's no `torch.manual_seed()` anywhere. So every time you train it, the model starts from different random guesses and improves along a different random path, producing a genuinely different trained model each time — even with identical code.

**Real proof this happened:** the original report's sample data and the new real data both included patient `TCGA_FG_6688` — but the tumor-slice count and tumor positions for that same patient didn't match between the two, which is only possible if two different trained models produced them.

**To fix this for future runs**, add near the top of the notebook, before the model is created:
```python
import torch, random, numpy as np
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 8. Real pictures from the notebook

Actual plots pulled straight out of the training code — a real brain scan slice next to the doctor's real marking and/or the model's own guess:

![scan example 1](demo_images/scan_image_2.png)
![scan example 2](demo_images/scan_image_3.png)
![scan example 3](demo_images/scan_image_4.png)
![scan example 4](demo_images/scan_image_5.png)
![scan example 5](demo_images/scan_image_6.png)
![scan example 6](demo_images/scan_image_7.png)
![scan example 7](demo_images/scan_image_8.png)
![scan example 8](demo_images/scan_image_9.png)

---

## 9. How it's used after training (real-life use, not training)

Once training is finished, the model is **frozen** — no more learning happens. For a brand-new patient:
1. They get a normal MRI scan (no AI involved in taking the scan itself).
2. Each slice is fed through the already-trained model exactly as-is.
3. The model outputs a pixel-by-pixel tumor map for that slice in a fraction of a second — just one quick pass through the fixed math, not a new training process.
4. In a real clinical setting, this kind of tool would typically be used as a fast second-opinion/triage aid for a radiologist to review, not as the sole decision-maker — especially since this exact model misses the tumor entirely on 2 of 22 real patients and only catches 58% of real tumor pixels overall.

---

## 10. Final real numbers — summary

| Metric | Value |
|---|---|
| Patients used to teach it | 88 |
| Patients used to test it | 22 |
| Mean Dice (shape overlap) | 0.773 |
| Mean IoU (other overlap measure) | 0.746 |
| Tumor correctly found | 20 of 22 patients (90.9%) |
| Pixel accuracy | 0.994 |
| Corrected precision | 0.876 |
| Corrected recall | 0.579 |

This is the one and only part of the whole project where every number comes from a real, trained model tested on real patient scans — everything else (Unity, the simulators) either visualizes this model's real output, or is a separate toy demo built on fake data (see `README_CODE.md`).
