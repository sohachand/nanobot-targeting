# Literature Review — Explained in Easy Words

A literature review answers one question: **"What have other researchers already done on this topic, and what is still missing that our project adds?"** This file explains your paper's literature review (Section 2 of `Fixed_Nanorobotic_Paper.docx`), paper by paper, and then adds the part that is missing for the MRI / U-Net half of the project.

Reference numbers below follow the **fixed paper** (the one with clean sequential numbering, [1] to [15]).

---

## 1. The one-line story of the whole review

1. Scientists have built real nanorobots (DNA origami, chemical motors, magnetic robots) that go to tumors in animals.
2. Newer work uses AI to decide where a nanorobot should go.
3. But every AI study uses **one** fixed model, never compares small vs. big models, never reports how many tumor cells were **missed**, and uses simple tumor shapes.
4. Almost nothing has reached patients yet.
5. **So the gap:** compare models of different size, report missed tumor cells as a main result, use a realistic irregular tumor shape.

---

## 2. Section 2.1 — What kinds of nanorobots exist? (Papers [1], [2], [3], [5], [6], [7])

There are three main kinds:

**a) DNA origami robots** (most studied). DNA is folded like paper origami into a tiny tube or box. It is safe for the body (biocompatible), breaks down naturally (biodegradable), and you can program its shape.
- Best-known example, **Li et al. 2018, Nature Biotechnology [3]**: a DNA tube carrying a blood-clotting drug (thrombin), locked shut. A lock made of a short DNA strand (an aptamer) only opens when it touches **nucleolin**, a marker found on tumor blood vessels. The tube opens, thrombin makes the tumor's blood vessels clot, and the tumor starves (shown in mice). Paper [4] (J Cell Mol Med) studies the same nucleolin idea.
- Later versions carry two drugs, trigger cell death directly, or open in response to acid (pH) or light ([2], [5]).

**b) Self-powered chemical motors.** They use fuel from the body itself. Example [6]: tiny motors with an enzyme (urease) that burns urea, which is plentiful in the bladder, so they swim by themselves and push into bladder tumors better than ordinary particles.

**c) Magnetic robots.** Steered from outside by magnetic fields. Paper [7] combines this with a wearable gesture controller and AI path planning during heat therapy (photothermal therapy).

Review paper [1] (Kong et al. 2023, J Hematology & Oncology) and [14] (Fu et al. 2025, Frontiers in Chemistry) give the big overview of all these types.

---

## 3. Section 2.2 — How do they know it is a tumor? (papers [2], [3], [4], [5])

Three tricks, repeated across all papers:
1. **Passive (EPR effect):** tumor blood vessels are leaky, so small particles leak in and get stuck. No "smartness" at all.
2. **Active targeting:** the robot surface carries a key (aptamer, folate, antibody piece) that fits a lock (marker) on cancer cells.
3. **Stimulus-triggered release:** the drug stays sealed until it senses something tumor-like: acid, special enzymes, light or a magnetic signal.

This is why DNA origami is popular: the same folding that gives the robot its shape also stores its "open when you meet the tumor" rule.

---

## 4. Section 2.3 — AI-controlled navigation (papers [8], [9], [10], [11])

This is the part closest to your project.

| Paper | What they did, in easy words |
|---|---|
| **Kavousinejad 2024 [8]** (arXiv) | Simulated nanorobots that **learn** to move using reinforcement learning (Q-learning) through a 3D space with cancer cells and obstacles, following chemical (biomarker) signals. |
| **Ezzat et al. 2019 [9]** (AMLTA) | Used **particle swarm optimization** plus "quorum sensing" (bacteria-style group communication) so a crowd of nanorobots converges on cancer cells. |
| **Distributed learning, Heliyon 2024 [10]** | Each robot learns its own policy by talking to neighbours. The group works better than one robot alone (like a team vs. a solo player). |
| **Intelligent deep learning model [11]** (PMC) | Used deep learning on the "whole delivery network" (a molecular-communication model) to control how much drug reaches the target cell, and got higher drug concentration than a normal compartment model. |

**What the review notices:** in all of these, the decision-making model is **one fixed design**. Nobody asked, "what if the model is simpler or deeper? What changes?"

---

## 5. Section 2.4 — Why hasn't this reached patients? (papers [12], [13], [14])

- **"Nanorobotics in cancer care: innovation without translation?" [12]**: great lab results, but no proper clinical evidence yet.
- **Abu Nahla et al. 2025 [13]**: using the TITAN reporting guidelines for AI, they reach the same conclusion: DNA-origami and magnetic systems look good in labs but face systematic barriers before clinical use.
- **Fu et al. 2025 [14]**: the difficulty is not bad studies; a totally new type of medicine is hard to test against rules written for ordinary drugs and devices.

In your own words: *"There is a big gap between lab/simulation success and real hospital use, so simulations like ours are for exploring ideas, not proving treatments."*

---

## 6. Section 2.5 — The research gap (the most important part to say to ma'am)

The review finds **three gaps**, which your paper then turns into its three tested variables:

1. **One model only.** Existing studies do not compare a small model against a bigger one. → *You compare Linear, Shallow and Deep networks.*
2. **Missed tumor is never reported.** They report accuracy or "successful delivery," but not how many cancer cells the model **never flags at all** (false negatives). → *You report missed cells as a main result.*
3. **Simple tumor shapes.** Past simulations use round/convex tumors that a straight line can separate easily. → *You use an irregular, multi-lobed tumor.*

---

## 7. IMPORTANT: what your current review does NOT cover

Your paper's review and methodology describe the **synthetic-cell classifier study** (the Unity game and first browser demo: 150 fake cells, Linear/Shallow/Deep networks). But your **report, deck and notebook** are about a **different, real-data pipeline**: a U-Net reading real brain MRI, then centroids, then navigation.

The paper's review has **no papers on the MRI side**. If ma'am asks "where is the literature for your U-Net?", this is the gap. Suggested extra references (all well-known; **please check volume/page details yourself before you cite them**):

| Topic | Paper | One-line use in your project |
|---|---|---|
| U-Net itself | Ronneberger, Fischer & Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351, pp. 234-241. | The model you trained. Encoder-decoder with skip connections. |
| Your dataset | Buda, Saha & Mazurowski (2019). *Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine, 109, 218-225. | Source of the LGG MRI + mask data (TCGA/TCIA via Kaggle). |
| Dice loss | Milletari, Navab & Ahmadi (2016). *V-Net: Fully convolutional neural networks for volumetric medical image segmentation.* 3DV 2016, pp. 565-571. | Origin of the Dice loss used for small targets. |
| Optimizer | Kingma & Ba (2015). *Adam: A method for stochastic optimization.* ICLR 2015 (arXiv:1412.6980). | The optimizer you used. |
| Weight start | He, Zhang, Ren & Sun (2015). *Delving deep into rectifiers.* ICCV 2015. | He initialization used in the hand-built network. |
| Brain tumor benchmark | Menze et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE Transactions on Medical Imaging, 34(10), 1993-2024. | Shows what good brain-tumor segmentation scores look like. |
| Best practice | Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18, 203-211. | Shows what strong U-Net training looks like; supports your "next steps". |
| 3D version | Cicek et al. (2016). *3D U-Net: Learning dense volumetric segmentation from sparse annotation.* MICCAI 2016. | Your own "next step": a 3D U-Net instead of slice-by-slice. |

A short paragraph you could add to the review:

> *In medical imaging, U-Net (Ronneberger et al., 2015) has become the standard encoder-decoder architecture for pixel-level segmentation, and its 3D extension (Cicek et al., 2016) and self-configuring variants (Isensee et al., 2021) set the benchmark on tasks such as BRATS (Menze et al., 2015). Overlap-based losses such as Dice (Milletari et al., 2016) are favoured when the target occupies few pixels. However, segmentation work stops at the mask. The reviewed nanorobot studies, in turn, assume the target location is given. Our work connects the two: tumor masks from a U-Net are converted into slice-wise centroids that define a 3D target path, and navigation strategies are compared on it.*

This also states a **second gap** you can claim: segmentation papers stop at the mask, and nanorobot papers assume the target is known; few connect the two.

---

## 8. How the review connects to each part of your project

| Part of your project | Which literature supports it |
|---|---|
| Why a nanobot needs a target | Sections 2.1-2.2 (what nanorobots are, how they recognise tumors) |
| Linear / Shallow / Deep comparison and missed-cell metric | Section 2.3 and gap in 2.5 |
| U-Net tumor finding on MRI | Section 7 above (new references) |
| Static / dynamic / obstacle-aware navigation | Papers [8], [9], [10] (navigation by learning, swarms, potential-field style attraction) |
| "This is only a simulation" honesty | Section 2.4 ([12], [13], [14]) |

---

## 9. Quick answers if ma'am asks

- **What is a literature review?** A summary of existing research that shows what is known and what is missing.
- **What gap did you find?** Nobody compared classifier size, reported missed tumor cells, or used irregular tumor shapes; and few studies connect real MRI segmentation to nanorobot navigation.
- **Is your work clinical?** No. It is a simulation to explore ideas, like most of the reviewed studies; the real-world gap is described in [12]-[14].
- **Which paper is closest to yours?** Kavousinejad 2024 [8] (AI nanorobot simulation in 3D with cancer cells), but it uses one learning method and no MRI.
