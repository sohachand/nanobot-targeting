using System;
using UnityEngine;

/// <summary>
/// A minimal, dependency-free feedforward network: configurable hidden layers,
/// ReLU activations, sigmoid output, trained with plain SGD backprop.
/// Same architecture family as the web prototype (Linear = no hidden layers,
/// Shallow = one hidden layer, Deep = two hidden layers) so results are
/// directly comparable.
/// </summary>
[Serializable]
public class NeuralNetwork
{
    public int[] layerSizes;         // e.g. [2, 8, 8, 1]
    public float[][][] weights;      // weights[layer][neuron][inputIndex]
    public float[][] biases;         // biases[layer][neuron]

    private System.Random rng;

    public NeuralNetwork(int[] hiddenLayers, int inputSize = 2, int outputSize = 1, int seed = -1)
    {
        rng = seed >= 0 ? new System.Random(seed) : new System.Random();

        var sizes = new int[hiddenLayers.Length + 2];
        sizes[0] = inputSize;
        for (int i = 0; i < hiddenLayers.Length; i++) sizes[i + 1] = hiddenLayers[i];
        sizes[sizes.Length - 1] = outputSize;
        layerSizes = sizes;

        int layerCount = sizes.Length - 1;
        weights = new float[layerCount][][];
        biases = new float[layerCount][];

        for (int l = 0; l < layerCount; l++)
        {
            int rows = sizes[l + 1], cols = sizes[l];
            float scale = Mathf.Sqrt(2f / cols); // He init, suits ReLU hidden layers
            weights[l] = new float[rows][];
            biases[l] = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                weights[l][i] = new float[cols];
                for (int j = 0; j < cols; j++)
                    weights[l][i][j] = (float)(rng.NextDouble() * 2 - 1) * scale;
                biases[l][i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            }
        }
    }

    static float ReLU(float x) => x > 0f ? x : 0f;
    static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    /// <summary>Forward pass. Returns the scalar output and (via out param) every layer's activations for backprop.</summary>
    public float Forward(float[] input, out float[][] activations)
    {
        activations = new float[weights.Length + 1][];
        activations[0] = input;
        float[] a = input;

        for (int l = 0; l < weights.Length; l++)
        {
            var Wm = weights[l];
            var b = biases[l];
            int rows = Wm.Length;
            var z = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                float s = b[i];
                for (int j = 0; j < Wm[i].Length; j++) s += Wm[i][j] * a[j];
                z[i] = s;
            }
            bool isOutput = l == weights.Length - 1;
            var act = new float[rows];
            for (int i = 0; i < rows; i++) act[i] = isOutput ? Sigmoid(z[i]) : ReLU(z[i]);
            activations[l + 1] = act;
            a = act;
        }
        return a[0];
    }

    /// <summary>Convenience forward pass when you only need the classification score (0..1).</summary>
    public float Predict(float[] input) => Forward(input, out _);

    /// <summary>One step of SGD on a single (input, label) pair. label is 0 or 1.</summary>
    public void TrainStep(float[] input, float label, float lr)
    {
        Forward(input, out var activations);
        int L = weights.Length;
        int lastLayer = activations.Length - 1;
        float[] delta = { activations[lastLayer][0] - label }; // sigmoid + BCE simplification

        for (int l = L - 1; l >= 0; l--)
        {
            float[] aPrev = activations[l];
            var Wm = weights[l];
            var b = biases[l];
            float[] deltaPrev = null;

            if (l > 0)
            {
                deltaPrev = new float[aPrev.Length];
                for (int j = 0; j < aPrev.Length; j++)
                {
                    float s = 0f;
                    for (int i = 0; i < Wm.Length; i++) s += delta[i] * Wm[i][j];
                    deltaPrev[j] = s * (aPrev[j] > 0f ? 1f : 0f); // ReLU derivative
                }
            }

            for (int i = 0; i < Wm.Length; i++)
            {
                for (int j = 0; j < Wm[i].Length; j++)
                    Wm[i][j] -= lr * delta[i] * aPrev[j];
                b[i] -= lr * delta[i];
            }
            delta = deltaPrev;
        }
    }
}
