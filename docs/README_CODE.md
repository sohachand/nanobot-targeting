# Code Files — Full Explanation (Separate Guide)

This is a separate guide just for the actual code files in the project — what each one does, in easy words, with the real code and what it means. Split into 3 groups: the Unity game scripts, the Python training notebook, and the two browser simulator files.

**Important correction/clarification first:** the Unity game is actually its own **separate, self-contained demo** — it does **not** use the real MRI brain scans or the real U-Net tumor-finding model at all. It uses made-up, synthetic tissue data and the simple hand-built neural network (Part 5 from the main README) to tell "malignant" vs "healthy" cells apart. Think of the Unity game as a simplified toy version showing the same overall idea (a robot chasing cells a network flags as cancerous), just using fake data instead of real patient scans. The real scan-reading work only happens in the Python notebook.

---

# GROUP 1 — THE UNITY GAME SCRIPTS

There are 5 files. Here's what each one does and how they connect:

## `NeuralNetwork.cs` — the hand-built "brain"

This is a neural network written completely from scratch — every single calculation done manually, no AI library used at all.

```csharp
public NeuralNetwork(int[] hiddenLayers, int inputSize = 2, int outputSize = 1, int seed = -1)
{
    ...
    float scale = Mathf.Sqrt(2f / cols); // He init, suits ReLU hidden layers
    ...
    weights[l][i][j] = (float)(rng.NextDouble() * 2 - 1) * scale;
}
```
**In easy words:** when a network is created, all its internal numbers (called "weights") start out as small random values — basically random guesses. The `scale` part is just a standard trick (called "He initialization") that keeps those random starting numbers in a sensible range so training works smoothly, rather than exploding or vanishing.

```csharp
static float ReLU(float x) => x > 0f ? x : 0f;
static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));
```
**In easy words:** two simple rules used while "thinking." ReLU just means: if a number is negative, treat it as zero; if positive, keep it as is. This simple rule is what lets the network build up complex decisions out of simple pieces. Sigmoid squashes any number into a range between 0 and 1 — used only at the very end, to turn the network's raw output into something readable as "how confident am I this cell is cancerous" (0 = definitely not, 1 = definitely yes).

```csharp
public float Forward(float[] input, out float[][] activations)
```
**In easy words:** this is the network actually "thinking" — it takes in the 2 numbers describing a cell (explained below) and passes them through layer by layer, each layer doing a little math and applying ReLU, until the final layer gives one number via Sigmoid. This single pass (no learning yet) is called a "forward pass."

```csharp
public void TrainStep(float[] input, float label, float lr)
```
**In easy words:** this is the "learning" part. After the network guesses, this function compares the guess to the correct answer (`label` — 1 if the cell really is cancerous, 0 if not), figures out how wrong it was, and then nudges every single internal number very slightly in the direction that would have made the guess less wrong. This happens over and over, for every cell, many times (called epochs), and that's literally the entire "training" process.

**Three versions are built, all using this same code, just with different shapes:**
```csharp
linear = new NeuralNetwork(new int[0]);       // logistic regression
shallow = new NeuralNetwork(new int[] { 6 }); // one hidden layer
deep = new NeuralNetwork(new int[] { 8, 8 }); // two hidden layers
```
**In easy words:** "Linear" has no hidden thinking steps at all — it's the simplest possible version (basically just straight math, not really a deep network). "Shallow" adds one layer of thinking in between. "Deep" adds two layers. These three are trained side-by-side so you can compare how adding more "thinking depth" changes how well it tells cells apart.

## `CancerCell.cs` — one cell in the simulation

```csharp
public float irregularity; // network input 1
public float biomarker;    // network input 2
public bool isMalignant;
```
**In easy words:** each cell in this Unity demo is just 2 made-up numbers — "irregularity" (how oddly shaped it is) and "biomarker" (some made-up chemical signal level), both between 0 and 1 — plus a hidden true answer, `isMalignant` (is it really cancerous or not). The network only ever gets to see the 2 numbers, never the real answer — that's the whole point, it has to guess from the clues, same as the real world.

```csharp
public float[] Features() => new float[] { irregularity, biomarker };
```
**In easy words:** this is literally the only information handed to the neural network — just those 2 numbers, nothing else.

## `TissueManager.cs` — builds the fake tissue sample and trains the networks

```csharp
int blobCount = Random.Range(3, 5);
for (int i = 0; i < blobCount; i++)
{
    float angle = Random.value * Mathf.PI * 2f;
    float dist = Random.Range(0.03f, 0.16f);
    blobCenters.Add((0.5f + Mathf.Cos(angle) * dist, 0.5f + Mathf.Sin(angle) * dist, Random.Range(0.09f, 0.15f)));
}
```
**In easy words:** this randomly creates 3-4 "cancerous zones" (blob centers) in a made-up 2D space. Any cell that falls inside one of these invisible circles is secretly marked as malignant; everything else is healthy. This is all fake/synthetic — not real biology, not real patient data — just a controlled way to generate a test case with a known right answer.

```csharp
if (Random.value < 0.05f) malignant = !malignant; // measurement noise
```
**In easy words:** 5% of the time, the cell's true label gets randomly flipped on purpose — to simulate the fact that real-world measurements are never perfectly clean, there's always a bit of noise/error.

```csharp
linear.TrainStep(feats, label, learningRate + 0.15f);
shallow.TrainStep(feats, label, learningRate);
deep.TrainStep(feats, label, learningRate);
```
**In easy words:** all 3 networks are trained on the exact same cells at the same time, so you can fairly compare them afterward — none of them gets an advantage from seeing different data.

```csharp
public (float acc, float prec, float rec) ComputeStats(NeuralNetwork net)
```
**In easy words:** after training, this checks each network's accuracy (how often it's right overall), precision (when it says "cancer," how often is it actually right), and recall (out of all the real cancer cells, how many did it actually catch) — the same kind of scoring explained for the real U-Net model in the main README, just applied here to this toy version.

## `NanobotController.cs` — the robot that chases flagged cells

```csharp
CancerCell FindNearestTarget()
{
    var net = tissue.GetNetwork(architecture);
    ...
    foreach (var c in tissue.cells)
    {
        if (c.neutralized) continue;
        if (net.Predict(c.Features()) <= 0.5f) continue;
        ...
    }
}
```
**In easy words:** the nanobot asks the trained network, one cell at a time, "do you think this one is cancerous?" (`net.Predict(...)`). It only considers cells the network scores above 0.5 (meaning "yes, I think this is cancer") as valid targets, then picks whichever one of those is physically closest. The comment at the top of this file says it best: *"the bot is exactly as good, and exactly as blind, as the model driving it"* — meaning if the network is wrong about a cell, the robot will be wrong too. It has no independent way of checking — it fully trusts the network's guess.

## `SimulationManager.cs` — wires everything together

This file just connects buttons (New Sample, Train, Deploy, Reset) and a dropdown (choosing Linear/Shallow/Deep) to the other scripts, and displays the live stats on screen. Nothing complicated here — it's just the "glue" that makes the buttons in the game actually do something.

## The bug that was found and fixed (recap)

The nanobot moved by directly setting its position every frame (`transform.position += dir * speed * Time.deltaTime;` in `NanobotController.cs`), but the nanobot's GameObject in the scene still had Unity's physics engine active on it — gravity turned on and solid (non-passthrough) colliders. This meant physics was fighting the script: gravity pulled the robot down off its path, and its solid collision shape physically bounced it away from a cancer cell's own collision shape before it could ever get close enough to register as "arrived" (`dist < arriveDistance`).

**The fix** — a new method added to `NanobotController.cs`:
```csharp
void Awake()
{
    var rb = GetComponent<Rigidbody>();
    if (rb != null)
    {
        rb.useGravity = false;   // stop gravity pulling it off course
        rb.isKinematic = true;   // let the script fully control movement, not physics
    }
    foreach (var col in GetComponentsInChildren<Collider>())
        col.isTrigger = true;    // let it detect cells without bouncing off them
}
```
This runs automatically once, the moment the nanobot is created, and switches off the physics behavior that was interfering. This fix was already made and sent to you separately — the copy you should be using has this `Awake()` method added.

---

# GROUP 2 — THE PYTHON TRAINING NOTEBOOK (THE REAL AI MODEL)

This is the actual AI work — training the U-Net on real brain MRI scans. Full code-by-code breakdown of this is already in the main `README.md` (Part 2), including:
- The U-Net model's shrink-then-expand design (`class UNet`)
- How it learns (`dice_loss` and the training loop)
- How it's scored (`dice_coefficient`, `iou_score`)
- The precision/recall bug and its fix (the pooled `TP/FP/FN/TN` counting code)

If you haven't read that part yet, that's where the real model's code lives — this file is just for the Unity and simulator code.

---

# GROUP 3 — THE TWO BROWSER SIMULATOR FILES

Both are plain HTML files with JavaScript and 3D graphics (Three.js) built in — no server, no install, just open in a browser.

## `nanobot_simulator.html`

Loads one patient's real centroid data (the dots from Part 3 of the main README) and runs all 3 navigation strategies (static/dynamic/obstacle-aware) in the browser, showing a live comparison table of the real numbers.

## `NanoBot_Tumor_Treatment_Sim.html`

The fuller swarm version. Two key functions worth understanding:

```javascript
function pickTarget(b) {
  ...
  for (const c of cells) {
    if (c.dead || c.dying || c.claims >= 1) continue;
    const d = c.p.distanceTo(from) - c.r;
    if (d < bestD) { bestD = d; best = c; }
    if (d < P.detect && d < bestLocalD) { bestLocalD = d; bestLocal = c; }
  }
  return bestLocal || (P.mode === "nearest" ? best : best);
}
```
**In easy words:** each robot only ever searches a pre-made list called `cells` (the cancer cells) — never the separate `healthy` list — when deciding what to chase. It picks the closest one within its "detection radius" (`P.detect`), skipping anything already dead or already being chased by another robot. This confirms what's explained in the main README: the simulator never actually figures out "is this cancer," it's just picking from a list that was already labeled before the simulation even started.

```javascript
function steer(b, goal, dt, arriveDist) {
  ...
  if (P.avoid) for (const h of healthy) {
    tmp2.copy(b.pos).sub(h.p); const d = tmp2.length() - h.r;
    if (d < 6) desired.add(tmp2.normalize().multiplyScalar((6 - Math.max(d, 0.2)) * 2.2));
  }
  ...
  for (const h of healthy) if (!h.affected && b.pos.distanceTo(h.p) < h.r * 0.9) affect(h);
}
```
**In easy words:** this is the movement logic. If "avoid healthy cells" is turned on, the robot gently pushes itself away from any nearby healthy cell while traveling. If it still physically touches one anyway, that cell gets marked `affected` — this is exactly what shows up as "Healthy cells affected" in the stats panel.

---

# QUICK SUMMARY — WHAT EACH CODE FILE ACTUALLY DOES

| File | What it does | Uses real data? |
|---|---|---|
| `NeuralNetwork.cs` | Hand-built learning brain (Linear/Shallow/Deep) | No — toy/demo |
| `CancerCell.cs` | Represents one fake cell with 2 made-up features | No — toy/demo |
| `TissueManager.cs` | Generates fake tissue, trains the 3 networks | No — toy/demo |
| `NanobotController.cs` | Robot that chases cells the network flags as cancer | No — toy/demo (now fixed) |
| `SimulationManager.cs` | Connects UI buttons to everything else | No — toy/demo |
| Training notebook | The real U-Net model, trained on real MRI scans | **Yes — real patient data** |
| `nanobot_simulator.html` | Compares 3 navigation strategies using real centroid data | **Yes, when you load a real patient file** |
| `NanoBot_Tumor_Treatment_Sim.html` | Fuller visual swarm demo | Only if you load a real patient file; otherwise fake/procedural |

**The one thing worth remembering:** the Unity game is a simplified, self-contained demo of the same idea using fake data — the real scientific result (77.3% match, 20-of-22 detection) only comes from the Python notebook and the real MRI dataset.
