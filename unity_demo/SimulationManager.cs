using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Wires the scene together: tissue generation, training, nanobot deployment,
/// and a plain uGUI readout of what happened. All UI fields are optional —
/// leave any of them unassigned in the Inspector and that piece just won't
/// update, so you can test with only some UI built.
/// </summary>
public class SimulationManager : MonoBehaviour
{
    [Header("References")]
    public TissueManager tissue;
    public GameObject nanobotPrefab;
    public Transform nanobotSpawnArea;

    [Header("UI (optional — wire up in Inspector)")]
    public Button newSampleButton;
    public Button trainButton;
    public Button deployButton;
    public Button resetButton;
    public Dropdown architectureDropdown; // options in order: Linear, Shallow, Deep
    public Slider nanobotCountSlider;     // suggested range 2–16
    public Text progressText;
    public Text statsText;
    public Text logText;

    private readonly List<NanobotController> activeBots = new List<NanobotController>();
    private int confirmedCount, falseAlarmCount;

    void Start()
    {
        tissue.GenerateSample();

        if (newSampleButton) newSampleButton.onClick.AddListener(OnNewSample);
        if (trainButton) trainButton.onClick.AddListener(OnTrain);
        if (deployButton) deployButton.onClick.AddListener(OnDeploy);
        if (resetButton) resetButton.onClick.AddListener(OnReset);

        tissue.OnTrainingProgress += (epoch, target) =>
        {
            if (progressText) progressText.text = $"Training… epoch {epoch}/{target}";
        };
        tissue.OnTrainingComplete += () =>
        {
            var lin = tissue.ComputeStats(tissue.linear);
            var sha = tissue.ComputeStats(tissue.shallow);
            var deep = tissue.ComputeStats(tissue.deep);
            if (progressText) progressText.text =
                $"Linear   acc {Pct(lin.acc)}  prec {Pct(lin.prec)}  rec {Pct(lin.rec)}\n" +
                $"Shallow  acc {Pct(sha.acc)}  prec {Pct(sha.prec)}  rec {Pct(sha.rec)}\n" +
                $"Deep     acc {Pct(deep.acc)}  prec {Pct(deep.prec)}  rec {Pct(deep.rec)}";
        };
    }

    string Pct(float v) => Mathf.RoundToInt(v * 100f) + "%";

    void OnNewSample()
    {
        ClearBots();
        tissue.GenerateSample();
        if (progressText) progressText.text = "";
        if (statsText) statsText.text = "";
        Log("New tissue sample loaded.");
    }

    void OnTrain()
    {
        ClearBots();
        tissue.StartTraining();
    }

    void OnDeploy()
    {
        if (!tissue.trained)
        {
            if (progressText) progressText.text = "Train the models first.";
            return;
        }
        ClearBots();
        int count = nanobotCountSlider ? Mathf.RoundToInt(nanobotCountSlider.value) : 6;
        Architecture arch = architectureDropdown ? (Architecture)architectureDropdown.value : Architecture.Deep;
        confirmedCount = 0;
        falseAlarmCount = 0;

        for (int i = 0; i < count; i++)
        {
            Vector3 spawnPos = nanobotSpawnArea.position + new Vector3(Random.Range(-2f, 2f), 2f + i * 0.3f, Random.Range(-2f, 2f));
            var go = Instantiate(nanobotPrefab, spawnPos, Quaternion.identity);
            var bot = go.GetComponent<NanobotController>();
            bot.tissue = tissue;
            bot.sim = this;
            bot.architecture = arch;
            activeBots.Add(bot);
        }
        Log($"Deployed {count} nanobots using the {arch} model.");
        UpdateStats();
    }

    void OnReset()
    {
        ClearBots();
        foreach (var c in tissue.cells)
        {
            c.neutralized = false;
            c.wasFalsePositive = false;
            c.UpdateVisual(false);
        }
        confirmedCount = 0;
        falseAlarmCount = 0;
        if (statsText) statsText.text = "";
        Log("Mission reset.");
    }

    void ClearBots()
    {
        foreach (var b in activeBots) if (b != null) Destroy(b.gameObject);
        activeBots.Clear();
    }

    public void ReportNeutralized(CancerCell cell)
    {
        if (cell.wasFalsePositive)
        {
            falseAlarmCount++;
            Log("Flagged cell cleared — turned out healthy (false positive).");
        }
        else
        {
            confirmedCount++;
            Log("Target neutralized — confirmed malignant.");
        }
        UpdateStats();
    }

    void UpdateStats()
    {
        if (!statsText) return;
        Architecture arch = architectureDropdown ? (Architecture)architectureDropdown.value : Architecture.Deep;
        var net = tissue.GetNetwork(arch);
        int totalMalignant = 0, flagged = 0;
        foreach (var c in tissue.cells)
        {
            if (c.isMalignant) totalMalignant++;
            if (c.isMalignant && net.Predict(c.Features()) > 0.5f) flagged++;
        }
        int missed = totalMalignant - flagged;
        statsText.text =
            $"{confirmedCount} / {flagged} flagged tumor cells cleared (of {totalMalignant} total)\n" +
            $"{falseAlarmCount} false alarms on healthy cells\n" +
            $"{missed} tumor cells the model never flagged";
    }

    void Log(string msg)
    {
        if (!logText) return;
        var lines = new List<string>((msg + "\n" + logText.text).Split('\n'));
        while (lines.Count > 8) lines.RemoveAt(lines.Count - 1);
        logText.text = string.Join("\n", lines);
    }
}
