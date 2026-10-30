# Nanobot Targeting — Unity Simulation

A 3D port of the browser prototype: three networks (linear, shallow, deep)
train on the same synthetic tissue sample, and nanobots act only on whichever
one you assign them — including its blind spots. Unity can't run inside this
chat, so this is the project's source, ready to drop into an editor.

**Requires:** Unity 2021 LTS or newer (Unity 6 works fine too). No packages
beyond the built-in `UnityEngine.UI` module.

## 1. Scripts

Copy the five files from `Scripts/` into your project's `Assets/Scripts/`
folder:

- `NeuralNetwork.cs` — the shared feedforward net (configurable hidden layers)
- `CancerCell.cs` — per-cell state and color
- `TissueManager.cs` — generates the sample, owns and trains all three networks
- `NanobotController.cs` — per-bot seek-and-neutralize behavior
- `SimulationManager.cs` — wires buttons/sliders to everything above

## 2. Prefabs

**Cell prefab**
1. Create a 3D Sphere, scale it to about `(0.35, 0.35, 0.35)`.
2. Add the `CancerCell` component.
3. Give it any Standard/URP material — its color is overwritten at runtime.
4. Drag it into `Assets/Prefabs/` to make it a prefab, then delete the scene instance.

**Nanobot prefab**
1. Create a 3D Cone (or Capsule), scale to about `(0.25, 0.25, 0.4)`.
2. Add the `NanobotController` component.
3. Give it an emissive amber material so it reads clearly against the cells.
4. Drag into `Assets/Prefabs/`, delete the scene instance.

## 3. Scene objects

1. Empty GameObject **`TissueOrigin`** at world origin — this is the parent
   transform cells spawn under.
2. Empty GameObject **`TissueManager`**, add the `TissueManager` script.
   - `Cell Prefab` → your cell prefab
   - `Tissue Origin` → the `TissueOrigin` transform
   - `Plane Size` → `(10, 10)` is a good start
   - `Cell Count` → `150`
3. Empty GameObject **`NanobotSpawnArea`**, positioned a few units above
   `TissueOrigin` (nanobots spawn near it and drop in).
4. Empty GameObject **`SimulationManager`**, add the `SimulationManager` script.
   - `Tissue` → the `TissueManager` object
   - `Nanobot Prefab` → your nanobot prefab
   - `Nanobot Spawn Area` → the `NanobotSpawnArea` transform

## 4. UI (uGUI, all optional but recommended)

Add a Canvas with:
- Buttons: **New Sample**, **Train**, **Deploy**, **Reset**
- A **Dropdown** with exactly three options, in this order: `Linear`,
  `Shallow`, `Deep` (the script reads the selected index as the enum)
- A **Slider**: min `2`, max `16`, whole numbers on — nanobot count
- Three **Text** elements: progress/training readout, mission stats, mission log

Drag each UI element into the matching field on `SimulationManager` in the
Inspector. Anything left unassigned is simply skipped — you can test with a
partial UI.

## 5. Play

Press Play, click **Train** and watch the progress text count epochs (training
runs across several frames so the app stays responsive), then click **Deploy**.
Switch the dropdown and click **Deploy** again without retraining to compare
how the linear, shallow, and deep models each handle the same tumor shape —
in particular, watch the "tumor cells the model never flagged" count in the
mission stats. That number is the real story: it doesn't show up in accuracy,
only in what nanobots literally never go near.

## Notes

- The tumor is an irregular blob (3–4 overlapping regions) in a normalized
  0–1 feature space, mapped onto the X/Z plane — same shape logic as the web
  version, so a straight decision boundary (the linear model) structurally
  can't trace it.
- This is a teaching sketch, not a real targeting system: two synthetic
  features stand in for what would be much richer, noisier biology in
  practice.
