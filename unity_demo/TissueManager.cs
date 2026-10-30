using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Architecture { Linear = 0, Shallow = 1, Deep = 2 }

/// <summary>
/// Generates a synthetic tissue sample (an irregular tumor blob in a 2D
/// feature/position space mapped onto a flat plane), and owns the three
/// networks that are trained on it side by side.
/// </summary>
public class TissueManager : MonoBehaviour
{
    [Header("Setup")]
    public GameObject cellPrefab;
    public int cellCount = 150;
    public Vector2 planeSize = new Vector2(10f, 10f); // world units along X/Z
    public Transform tissueOrigin;

    [Header("Training")]
    public int epochs = 220;
    public float learningRate = 0.35f;
    public int epochsPerFrame = 4;

    public List<CancerCell> cells = new List<CancerCell>();
    private List<(float cx, float cz, float r)> blobCenters = new List<(float, float, float)>();

    public NeuralNetwork linear, shallow, deep;
    public bool trained = false;

    public System.Action<int, int> OnTrainingProgress; // (epoch, target)
    public System.Action OnTrainingComplete;

    void Awake()
    {
        InitNetworks();
    }

    void InitNetworks()
    {
        linear = new NeuralNetwork(new int[0]);       // logistic regression
        shallow = new NeuralNetwork(new int[] { 6 }); // one hidden layer
        deep = new NeuralNetwork(new int[] { 8, 8 }); // two hidden layers
        trained = false;
    }

    public NeuralNetwork GetNetwork(Architecture a) =>
        a == Architecture.Linear ? linear : a == Architecture.Shallow ? shallow : deep;

    public void GenerateSample()
    {
        StopAllCoroutines();
        foreach (var c in cells) if (c != null) Destroy(c.gameObject);
        cells.Clear();
        InitNetworks();

        blobCenters.Clear();
        int blobCount = Random.Range(3, 5);
        for (int i = 0; i < blobCount; i++)
        {
            float angle = Random.value * Mathf.PI * 2f;
            float dist = Random.Range(0.03f, 0.16f);
            blobCenters.Add((0.5f + Mathf.Cos(angle) * dist, 0.5f + Mathf.Sin(angle) * dist, Random.Range(0.09f, 0.15f)));
        }

        for (int i = 0; i < cellCount; i++)
        {
            float nx = Random.value, nz = Random.value;
            bool malignant = IsMalignant(nx, nz);
            if (Random.value < 0.05f) malignant = !malignant; // measurement noise

            Vector3 pos = tissueOrigin.position + new Vector3(
                (nx - 0.5f) * planeSize.x,
                Random.Range(-0.15f, 0.15f),
                (nz - 0.5f) * planeSize.y);

            var go = Instantiate(cellPrefab, pos, Quaternion.identity, tissueOrigin);
            var cell = go.GetComponent<CancerCell>();
            cell.Initialize(nx, nz, malignant);
            cells.Add(cell);
        }
    }

    bool IsMalignant(float x, float z)
    {
        foreach (var b in blobCenters)
        {
            float dx = x - b.cx, dz = z - b.cz;
            if (Mathf.Sqrt(dx * dx + dz * dz) < b.r) return true;
        }
        return false;
    }

    public void StartTraining()
    {
        InitNetworks();
        StartCoroutine(TrainCoroutine());
    }

    IEnumerator TrainCoroutine()
    {
        int epoch = 0;
        var order = new List<int>(cells.Count);
        for (int i = 0; i < cells.Count; i++) order.Add(i);

        while (epoch < epochs)
        {
            for (int f = 0; f < epochsPerFrame && epoch < epochs; f++, epoch++)
            {
                Shuffle(order);
                foreach (var idx in order)
                {
                    var c = cells[idx];
                    float label = c.isMalignant ? 1f : 0f;
                    var feats = c.Features();
                    linear.TrainStep(feats, label, learningRate + 0.15f);
                    shallow.TrainStep(feats, label, learningRate);
                    deep.TrainStep(feats, label, learningRate);
                }
            }
            OnTrainingProgress?.Invoke(epoch, epochs);
            yield return null; // spread training across frames so the app stays responsive
        }
        trained = true;
        OnTrainingComplete?.Invoke();
    }

    void Shuffle(List<int> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = Random.Range(0, i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    public (float acc, float prec, float rec) ComputeStats(NeuralNetwork net)
    {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        foreach (var c in cells)
        {
            bool pred = net.Predict(c.Features()) > 0.5f;
            if (pred && c.isMalignant) tp++;
            else if (pred && !c.isMalignant) fp++;
            else if (!pred && c.isMalignant) fn++;
            else tn++;
        }
        float acc = (float)(tp + tn) / cells.Count;
        float prec = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0f;
        float rec = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0f;
        return (acc, prec, rec);
    }
}
