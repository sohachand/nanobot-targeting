using UnityEngine;

/// <summary>
/// A single nanobot. It never looks at ground truth — only at whether the
/// assigned network's Predict() call for a cell is above 0.5. That's the
/// entire point of the demo: the bot is exactly as good, and exactly as
/// blind, as the model driving it.
/// </summary>
public class NanobotController : MonoBehaviour
{
    public TissueManager tissue;
    public SimulationManager sim;
    public Architecture architecture = Architecture.Deep;
    public float speed = 4f;
    public float arriveDistance = 0.4f;

    private CancerCell target;

    void Update()
    {
        if (tissue == null || !tissue.trained) return;

        if (target == null || target.neutralized)
            target = FindNearestTarget();

        if (target == null)
        {
            // nothing left the model flags — hover instead of vanishing
            transform.position += Vector3.up * Mathf.Sin(Time.time * 2f + transform.position.x) * 0.002f;
            return;
        }

        Vector3 toTarget = target.transform.position - transform.position;
        float dist = toTarget.magnitude;

        if (dist < arriveDistance)
        {
            target.Neutralize();
            sim.ReportNeutralized(target);
            target = null;
        }
        else
        {
            Vector3 dir = toTarget / dist;
            transform.position += dir * speed * Time.deltaTime;
            transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(dir), 10f * Time.deltaTime);
        }
    }

    CancerCell FindNearestTarget()
    {
        var net = tissue.GetNetwork(architecture);
        CancerCell best = null;
        float bestDist = float.MaxValue;

        foreach (var c in tissue.cells)
        {
            if (c.neutralized) continue;
            if (net.Predict(c.Features()) <= 0.5f) continue;

            float d = Vector3.Distance(transform.position, c.transform.position);
            if (d < bestDist) { bestDist = d; best = c; }
        }
        return best;
    }
}
