using UnityEngine;

/// <summary>
/// One cell in the tissue sample. Carries the same two synthetic features the
/// web prototype uses (irregularity, biomarker level), both normalized 0..1,
/// plus the ground-truth label. Networks never see isMalignant directly —
/// only Features().
/// </summary>
public class CancerCell : MonoBehaviour
{
    [Header("Ground truth (set by TissueManager)")]
    public float irregularity; // network input 1
    public float biomarker;    // network input 2
    public bool isMalignant;

    [Header("Mission state")]
    public bool neutralized;
    public bool wasFalsePositive;

    private Renderer rend;

    static readonly Color HealthyColor = new Color(0.36f, 0.55f, 0.93f);
    static readonly Color MalignantColor = new Color(0.94f, 0.40f, 0.29f);
    static readonly Color ConfirmedColor = new Color(0.18f, 0.83f, 0.75f);
    static readonly Color FalsePositiveColor = new Color(0.75f, 0.52f, 0.98f);

    void Awake()
    {
        rend = GetComponentInChildren<Renderer>();
    }

    public void Initialize(float irregularity, float biomarker, bool isMalignant)
    {
        this.irregularity = irregularity;
        this.biomarker = biomarker;
        this.isMalignant = isMalignant;
        neutralized = false;
        wasFalsePositive = false;
        UpdateVisual(false);
    }

    public float[] Features() => new float[] { irregularity, biomarker };

    public void Neutralize()
    {
        neutralized = true;
        wasFalsePositive = !isMalignant;
        UpdateVisual(false);
    }

    /// <summary>flaggedByModel just controls the idle (pre-neutralized) size cue — a small "the model is watching this one" tell.</summary>
    public void UpdateVisual(bool flaggedByModel)
    {
        if (rend == null) return;
        Color c = neutralized
            ? (wasFalsePositive ? FalsePositiveColor : ConfirmedColor)
            : (isMalignant ? MalignantColor : HealthyColor);
        rend.material.color = c;
        transform.localScale = Vector3.one * (!neutralized && flaggedByModel ? 0.45f : 0.35f);
    }
}
