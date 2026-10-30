# Project Report — Explained in Easy Words

File: `Nanobot_Targeting_Project_Report_FINAL.docx`
Title: *Deep Learning-Guided Nanorobotic Targeting — U-Net Brain Tumor Segmentation and 3D Navigation Simulation*

Every number below is from the **one real notebook run** (Dice 0.773, IoU 0.746, tumor found in 20 of 22 patients). The report has 8 sections, 7 tables, 7 figures and an appendix.

---

## The report in 30 seconds

1. A nanobot needs coordinates of the tumor, not just "there is a tumor."
2. A U-Net AI reads brain MRI slices and marks the tumor on each one.
3. The center of each marked tumor becomes a point; the points stacked by slice form a 3D path.
4. Two navigation strategies are compared on that path: **static** (one straight line to the average point, short but a bit off) and **dynamic** (visit every slice's point, long but exact). A browser simulator adds a third, obstacle-avoiding one.
5. Result: Dice 0.773; static is about 4x shorter but ends about 6 voxels away; dynamic always lands on the tumor.
6. It is a simulation only: no physical robot, blood flow or tissue mechanics.

---

## Section by section

### Abstract
The whole project in one paragraph: model score (Dice 0.773, IoU 0.746, tumor found in 20 of 22), the path comparison (mean 46.4 vs 186.3 voxels; static ends 5.98 voxels away on average), and the simulator. Read this first; ma'am will.

### 1. Introduction
Why the project exists: nanorobots need coordinates; MRI already shows the tumor but drawing it by hand is slow. Lists the three steps (segment, make a 3D path, navigate) and states clearly that this is a simulation.

### 2. Dataset
- **LGG MRI Segmentation dataset**: 110 patients, 3,929 slices, low-grade glioma, from TCGA/TCIA, assembled by Buda et al. on Kaggle.
- Each slice has 3 channels (3 MRI types: pre-contrast, FLAIR, post-contrast) and a black-and-white tumor mask.
- Most masks are empty because a tumor only covers some slices.
- **Table 1**: split by patient, **88 train (3,151 slices) / 22 validation (778 slices)**. Split by patient (not by slice) so neighbouring, near-identical slices of one person never end up in both sets (that would be cheating by leakage). The split is by patient with seed 42, but patient folders are listed in file-system order, so the exact validation patients can differ between sessions.
- **Figure 1**: one training slice next to its tumor mask.

### 3. Methodology (how it was built)
- **3.1 Preprocessing**: convert colour order, resize to 256x256, scale pixel values to 0-1, resize masks and make them strictly black/white.
- **3.2 U-Net**: shrinks the image in 4 steps (64 to 512 channels, 1,024 at the bottom) then grows it back; skip connections copy sharp detail across; the last layer gives a tumor probability per pixel.
- **3.3 Training (Table 2)**: input 3x256x256, loss = Dice + binary cross-entropy (Dice helps because tumor pixels are few), Adam optimizer 1e-4, batch 16, 20 epochs, Kaggle T4 GPU, threshold 0.5. No augmentation and no pretrained weights.
- **3.4 Centroids**: the center of the tumor on each slice comes from image moments (x = M10/M00, y = M01/M00). Empty slice gives no point. The points in slice order are the 3D path; their mean is the static target.
- **3.5 Navigation strategies**: all start at slice 0, centre (128,128), step 1.5 voxels. Static: straight to the mean point. Dynamic: each point in order, moving on within 1 voxel. Obstacle-aware (simulator only): dynamic plus a push-away from obstacles.
- **3.6 Metrics**: Dice and IoU per slice averaged over 778 slices; detection rate per patient; path length, steps, final distance to nearest true centroid, clearance to obstacles.

### 4. Results
- **4.1 Training (Figure 2)**: training loss 1.264 to 0.174; validation loss 1.188 to 0.584. Validation levels off after epoch 9 while training keeps falling: early overfitting. The saved model is epoch 20 (0.584), not the best epoch 19 (0.580). Epochs 13 and 17 are missing from the saved log.
- **4.2 Segmentation (Table 3, Figure 3)**: Dice 0.7732, IoU 0.7459, tumor found in 20 of 22 (90.9%), accuracy 0.9940, specificity 0.9990. Accuracy and specificity look great only because most pixels are background, so Dice and IoU are the numbers that matter.
- **4.3 Centroids (Figure 4)**: patient **TCGA_HT_7686**, tumor on 9 slices (7 to 15), centres shift smoothly, mean target slice 11.0, x 148.6, y 166.7. Contrast: TCGA_FG_6688 has centroids that jump around, so its static path ends far off.
- **4.4 Navigation (Table 4, Figures 5 and 6)**: over the 20 patients with a detected tumor, static averages 46.4 voxels and dynamic 186.3 (medians 43.1 and 143.5); static ends 5.98 voxels away on average, worst case 27.17 (TCGA_FG_6688); dynamic always ends at 0.00. Figure 5 plots all 20 patients; Figure 6 shows TCGA_HT_7686: static 45.2 voxels (ends 2.24 away), dynamic 66.6 voxels in 53 steps through all 9 slices.

### 5. Correcting the precision and recall metrics
The most "research-like" section. The first version averaged precision/recall per slice and got **0.238 / 0.193 / F1 0.204**, which cannot match a Dice of 0.773. Reason: tumor-free slices give 0 divided by 0, which was counted as 0, and most slices are tumor-free. Fix: add up TP, FP, FN over all pixels first, then compute once (Table 5): **precision 0.876, recall 0.579, F1 0.697, IoU 0.535** from TP 351,552, FP 49,579, FN 255,679, TN 50,330,198. Meaning: when it says tumor it is usually right, but it catches only about 58% of the real tumor pixels, so it is cautious.

### 6. Web simulator (Figure 7)
The browser tool: loads one JSON per patient, draws the brain and a disc per tumor slice, runs the same pursuit maths in JavaScript, lets you change entry point, step size and repulsion, and shows a comparison table. Obstacles are **three simulated spheres**, not real anatomy. On 5 synthetic sample patients, static and dynamic enter the obstacles (clearance -4.8 to -3.2) while obstacle-aware stays clear by 5.1 to 5.2 voxels with 15 to 18 voxels extra path. These samples are demos, not patient results.

### 7. Limitations and future work (Table 6)
2D model; plateau after epoch 9; stray detections pull the dynamic path (TCGA_HT_7605 has 13 slices but a 310.7-voxel path); only 22 validation patients and no test set; earlier per-slice metrics understated the model; simulated obstacles; straight-line motion in voxel units.

### 8. Conclusion
Restates: Dice 0.773 on 22 unseen patients, 20 detected; static about 4x shorter but misses by 5.98 voxels; dynamic always lands on tumor; precision 0.876 and recall 0.579 (conservative model); the simulator makes the comparison explorable.

### References
Ronneberger 2015 (U-Net), Buda 2019 (dataset), Clark 2013 (TCIA), Milletari 2016 (Dice loss), Kingma and Ba 2015 (Adam), Khatib 1986 (potential-field obstacle avoidance), and the Kaggle dataset page.

### Appendix A (Table 7)
All **20** patients with a detected tumor, sorted by dynamic path length: slices detected, static path, dynamic path, static final distance. Two patients had no detected tumor so they are not listed. Dynamic is always 0.00 so that column is omitted.

---

## What I corrected while preparing this README

When I checked the report I had sent earlier, a few leftovers from the old run were still in it. They are now fixed in `Nanobot_Targeting_Project_Report_FINAL.docx`:
- Results intro said Dice 0.828 and "3x"; now 0.773 and "4x".
- Table 5 (pooled metrics) still had placeholders; now filled (0.8764 / 0.5789 / 0.6973 / 0.5352).
- Figure 2 chart, Figure 5 chart and Figure 6 plot were from the old run; redrawn from the real epoch log and the real 20 patients (Figure 6 now shows TCGA_HT_7686).
- Figure 2 caption now says epochs 13 and 17 were missing; Table 4 caption says 20 patients; Section 5 first-version values updated to 0.238 / 0.193 / 0.204.

Figures 1, 3, 4 and 7 come from the notebook/simulator and are unchanged.

---

## Questions ma'am may ask about the report

- **Why split by patient?** Neighbouring slices are almost identical; mixing them would leak test data into training.
- **Why is accuracy 99% but recall only 58%?** Almost every pixel is background, so accuracy is easy; recall measures only the tumor pixels.
- **Why Dice + BCE?** Tumor pixels are rare; BCE alone is dominated by background, Dice rewards overlap directly.
- **Why does dynamic take longer?** It visits every slice's centre, tracing the tumor, instead of one straight line.
- **Is this clinical?** No. Simulation only: no blood flow, tissue mechanics or physical robot (stated in Sections 1 and 7).
- **Why would numbers change if retrained?** The random seed for the model was not fixed (and the patient split depends on file-system folder order), so each training run gives a slightly different model.
