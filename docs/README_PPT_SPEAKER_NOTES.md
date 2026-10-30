# PPT Speaker Notes — Deep Learning-Guided Nanorobotic Targeting (15 slides)

How to use this file: for every slide you get (1) what is on the slide, (2) what to SAY in easy words, and (3) questions ma'am may ask, with answers. Total talk time is about 12-15 minutes (roughly 1 minute per slide).

## READ THIS FIRST — the deck has old numbers on some slides

The slide deck (PDF) was made from an earlier training run. The real notebook run your group member sent gives different numbers. The report is already fixed to the real run, but the deck is not (we only have the locked PDF, not the editable file). So **on these slides, say the corrected number out loud, and ideally fix the slide before presenting**:

| Slide | What the slide shows (old) | What is actually true (real run) |
|---|---|---|
| 6 | Training loss 1.285 → 0.158, validation loss 1.150 → 0.303, plateau after epoch 13 | Training loss 1.264 → 0.174, validation loss 1.188 → 0.584, plateau after epoch 9 |
| 7 | Dice 0.828, IoU 0.789, 22/22 patients | Dice 0.773, IoU 0.746, 20/22 patients (90.9%) |
| 8 | Precision [__], recall [__], F1 [__]; "0.31 and 0.23" | Precision 0.876, recall 0.579, F1 0.697 (pooled). The broken per-slice values in the real run were about 0.24 and 0.19, against a Dice of 0.77 |
| 9 | Example patient TCGA_DU_5855, 11 slices | Use TCGA_HT_7686: 9 tumor slices, mean target slice 11.0, x 148.6, y 166.7 |
| 11 | Static 45.7 vs dynamic 148.8, final distance 6.19, worst 39.4, "about 3× shorter" | Static 46.4 vs dynamic 186.3, final distance 5.98, worst 27.17, "about 4× shorter" (median 43.1 vs 143.5) |
| 12 | Patient TCGA_DU_5855, static 40.2, dynamic 79.5 | TCGA_HT_7686: static 45.2 (ends 2.24 away), dynamic 66.6, visits all 9 tumor slices |
| 14 | Validation loss plateaus after epoch 13 | Plateaus after epoch 9 |
| 15 | Dice 0.828, final distance 6.19 | Dice 0.773, final distance 5.98 |

Why the numbers differ: the training code never fixes the model's random starting point (no `torch.manual_seed()`), so every training run gives a slightly different model. The patient split (88 train / 22 test) is fixed, but the model itself is not. If ma'am notices a mismatch, this is your honest, correct answer.

Slides 1-5, 10, 13 need no number changes.

---

## Slide 1 — Title

**On the slide:** "Deep Learning-Guided Nanorobotic Targeting — U-Net tumor segmentation on brain MRI, driving 3D nanobot navigation."

**Say:** "Good morning ma'am. My project is about helping a tiny medical robot, a nanobot, find a brain tumor. The AI part reads a brain MRI scan and marks where the tumor is. Then we turn that into a path the nanobot can follow. I'll show how it works, what results we got, and what its limits are."

**Possible question:** *Did you build a real nanobot?* No. We built the AI that finds the target and the simulation that shows how a nanobot would move to it. Real nanobots are still a research topic.

---

## Slide 2 — The problem

**On the slide:** Three boxes: targeting needs coordinates; MRI already holds the map; this project.

**Say:** "A nanobot can only deliver medicine to a tumor if it knows where the tumor is. It needs an actual position, not just a diagnosis like 'this patient has a tumor'. An MRI already contains that information, slice by slice, but marking the tumor by hand is slow. So we automate it: an AI marks the tumor, we convert that into 3D coordinates, and then we compare three ways a nanobot could travel to it."

**Possible question:** *Why MRI?* It shows soft tissue like the brain clearly without radiation, and it comes as a stack of slices, which is perfect for building a 3D path.

---

## Slide 3 — Method (5 stages)

**On the slide:** MRI slices → U-Net → Centroids → 3D target → Navigation.

**Say:** "Our pipeline has five steps. One: take the MRI slices, resized to 256 by 256. Two: a U-Net gives, for every pixel, the probability that it is tumor, and we call it tumor above 0.5. Three: for each slice we find the center of the tumor blob. Four: we stack those centers by slice number to get a 3D path. Five: we compare three navigation strategies on that path. Steps 1 to 3 run in PyTorch on a Kaggle GPU, and steps 4 and 5 run both in the notebook and in a browser simulator."

**Possible question:** *What is a centroid?* The average position of all tumor pixels in a slice, like the balance point of the blob.

---

## Slide 4 — Data

**On the slide:** 110 patients, 3,929 slices; LGG dataset; 88 train / 22 test.

**Say:** "We used the public LGG Brain MRI dataset: real scans of 110 patients with low-grade glioma, a type of brain tumor. Each slice has a doctor-drawn mask showing where the tumor is. We split by patient, not by slice: 88 patients for training and 22 completely held-out patients for testing. That matters because if slices of the same patient were in both sets, the model could partly memorize that patient and the test would be unfairly easy."

**Possible question:** *What is data leakage?* When information from the test set sneaks into training, making results look better than they really are. Splitting by patient prevents it.

---

## Slide 5 — Model

**On the slide:** U-Net from scratch; settings table (input 3×256×256, Dice+BCE loss, Adam 1e-4, batch 16, 20 epochs, threshold 0.5).

**Say:** "The model is a U-Net. Think of squinting at a photo to get the big picture, then opening your eyes to trace exact edges. The encoder shrinks the image while learning bigger and bigger patterns, going from 64 up to 1,024 feature channels. The decoder grows it back to full size. Skip connections copy sharp details from the way down to the way up, so tumor edges stay precise. The output is one number per pixel between 0 and 1. For the loss we use Dice plus binary cross-entropy: tumors cover very few pixels, so Dice rewards overlap directly, and cross-entropy keeps each pixel's training stable. We trained from scratch for 20 epochs with the Adam optimizer."

**Possible questions:**
- *Why not a pretrained model?* We trained from scratch for simplicity. A pretrained encoder is listed as a next step.
- *What is an epoch?* One full pass over all training images.
- *What does the 0.5 threshold do?* Pixels scoring above 0.5 are called tumor. Raising it means fewer false alarms but more missed tumor, and lowering it does the opposite.

---

## Slide 6 — Training (fix numbers!)

**On the slide:** Loss curves; "validation loss levels off after epoch 13".

**Say (real run):** "Training loss fell from 1.26 to 0.17 over 20 epochs. Validation loss, measured on patients the model is not learning from, fell from 1.19 to about 0.58, but flattened out around epoch 9 while training loss kept falling. That is an early sign of overfitting: after epoch 9 the model mostly gets better at the training images rather than learning something general. Also, our code saves the final epoch, not the best one, and epoch 19 was very slightly better than epoch 20."

**Possible questions:**
- *What is overfitting?* Memorizing training data instead of learning general patterns. Training error keeps dropping while validation error stalls.
- *How would you reduce it?* Data augmentation (flips, rotations), early stopping, a pretrained encoder, or more data.

---

## Slide 7 — Segmentation results (fix numbers!)

**On the slide:** Dice 0.828, IoU 0.789, 22/22.

**Say (real run):** "On the 22 patients the model never saw, the mean Dice score was 0.773 and the mean IoU was 0.746. Both measure how much the predicted tumor overlaps the doctor's drawing, where 1.0 is perfect. The model found tumor in 20 of the 22 patients, so it completely missed two."

**Possible questions:**
- *Dice vs IoU?* Both measure overlap. IoU = overlap divided by combined area; Dice = twice the overlap divided by the sum of both areas. Dice is always a bit higher.
- *Why did it miss 2 patients?* Likely small or faint tumors. A 2D model that looks at one slice at a time has no volume context.

---

## Slide 8 — A correction to the metrics (fill the blanks!)

**On the slide:** Per-slice precision and recall were misleading; fix by pooling counts. The corrected values are blank: precision [__], recall [__], F1 [__].

**Say:** "Our first precision and recall numbers looked wrong: about 0.24 and 0.19, while Dice was 0.77. They can't both be true. The reason: most slices contain no tumor at all. On those slices, a correct empty prediction gives 0 divided by 0, and our first code counted that as zero, which dragged the average down unfairly. The fix is to add up true positives, false positives and false negatives across every pixel of every validation slice first, and compute the score once. The corrected values are precision 0.876, recall 0.579 and F1 0.697. In plain words: when the model says 'tumor' it is right about 88% of the time, but it only catches about 58% of the real tumor pixels. It is cautious: it misses more than it falsely flags."

**Possible questions:**
- *Is missing tumor worse than a false alarm?* Usually yes in medicine, so this weakness is in the riskier direction. That is why a tool like this should assist a radiologist, not replace one.
- *Precision vs recall?* Precision: of the pixels I flagged, how many were truly tumor. Recall: of the true tumor pixels, how many did I catch.

---

## Slide 9 — From mask to target (swap the example!)

**On the slide:** Centroid formula x = M10/M00, y = M01/M00; example patient with the tumor shown.

**Say:** "To turn a tumor mask into a target point we use image moments: M00 is the tumor area, and M10 over M00 and M01 over M00 give the average x and y position, which is the center of the blob. A slice with no tumor gives no point. Stacking these points in slice order gives a 3D path as slice, x, y. For example, patient TCGA_HT_7686 has tumor on 9 slices, and the center moves smoothly from slice to slice, which means the model found one consistent tumor."

**Possible question:** *Is this step AI?* No, it is simple geometry: averaging pixel positions.

---

## Slide 10 — Navigation (three strategies)

**On the slide:** Static, Dynamic, Obstacle-aware, with formulas.

**Say:** "We compare three ways for the nanobot to move, all starting from the same entry point and moving 1.5 voxels per step. Static: go in a straight line to the average of all centers. It is the shortest route, but a tumor is a 3D shape and not a single point, so you can end up beside it. The formula just says: each step, move a bit toward the target. Dynamic: visit each slice's center in order, moving on once within 1 voxel. This traces the tumor's real shape. Obstacle-aware: dynamic plus a push-away force from critical structures. The second formula says: add a repulsion from each obstacle, stronger when closer, and faded near each waypoint so the nanobot cannot get stuck."

**Possible questions:**
- *Is dynamic the shortest possible path?* No, it is a simple sequential strategy, not an optimal one.
- *Are obstacles real?* No. They are simulated spheres, stated on slide 14.

---

## Slide 11 — Navigation results (fix numbers!)

**On the slide:** Table static vs dynamic.

**Say (real run, 20 patients with detected tumor):** "Static paths average 46 voxels and dynamic average 186 voxels, so static is about four times shorter. But static ends on average 6 voxels away from the nearest tumor point, and in the worst case, patient TCGA_FG_6688 with a tiny scattered tumor, 27 voxels away. Dynamic always ends exactly on tumor tissue. So it is a trade-off: fast but imprecise, against slow but accurate."

**Possible question:** *Why only 20 patients and not 22?* Two patients had no detected tumor, so there is no path to compute.

---

## Slide 12 — One patient, two paths (swap the example!)

**Say (real run):** "Here is one patient, TCGA_HT_7686. The static path is 45 voxels and ends about 2 voxels from the tumor. The dynamic path is 67 voxels, visits all 9 tumor slices and ends on a center point. You can see the static line cuts straight across while the dynamic line follows the tumor."

---

## Slide 13 — Interactive simulator

**On the slide:** Browser simulator; loads the notebook's per-patient JSON.

**Say:** "We also built a browser simulator. It loads the JSON file exported from the notebook, runs the same pursuit math in JavaScript, and lets you change the entry point, step size and obstacle repulsion, with a live comparison table. On the sample patients, only the obstacle-aware bot stays clear of the critical structures, by about 5 voxels, while the other two cut through them. The patient shown on this slide is a synthetic sample, not a real patient. We say so on the slide."

**Possible question:** *Does the simulator detect cancer by itself?* No. Which cells are tumor is already decided by the AI model before the simulation starts. The simulator only shows movement.

---

## Slide 14 — Limitations and next steps

**Say:** "To be honest about the limits: our model is 2D and looks at one slice at a time, so a 3D U-Net would use volume context. Validation loss plateaus after epoch 9, so augmentation, early stopping or a pretrained encoder could help. We only have 22 validation patients and no separate test set, so cross-validation is a next step. The critical structures are simulated spheres, not real anatomy. And the motion is a straight line in voxel units, with no blood flow or tissue drag."

**Bonus point to mention:** "Re-running our training gives slightly different numbers because the model's random seed was not fixed. We found this ourselves when our report numbers did not match the notebook, and the fix is to set `torch.manual_seed`."

---

## Slide 15 — Conclusion

**Say (real run):** "To conclude: segmentation turns an MRI into a target a nanobot can steer to. Our model reached a mean Dice of 0.773 on 22 unseen patients. With dynamic tracking the nanobot always ends on tumor tissue, against about 6 voxels away for static targeting. We compared three navigation strategies in an open browser simulator. Thank you. I am happy to take questions."

---

## Extra: 5 questions ma'am is most likely to ask, with short answers

1. **Does a real nanobot work like this?** No. Real nanoparticles cannot see. They use chemistry: receptor binding (lock and key), leaky tumor blood vessels (EPR effect), pH sensitivity, or magnets guided by an earlier scan. Our project covers the scan-reading and path-planning side only.
2. **How does the AI tell tumor from healthy tissue?** It learned the pattern from thousands of doctor-labeled slices: filters first pick up edges and brightness, deeper layers pick up shapes and unusual regions, and training nudges all its internal numbers to reduce the error against the doctor's mask.
3. **Why do numbers change if you retrain?** No fixed random seed. The patient split depends on folder listing order, and the model's starting weights are random each run.
4. **Is 77% Dice good?** Reasonable for a basic 2D U-Net trained from scratch. Published models on this dataset are often higher, using 3D context, augmentation or pretrained encoders.
5. **What would you do next?** 3D U-Net, augmentation, cross-validation, real anatomical obstacles, and realistic physics.
