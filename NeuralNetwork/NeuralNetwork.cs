using NeuralNetwork.ActivationFunctions;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public IActivationFunction ActivationFunction { get; private set; }
        public NeuralLayer[] Layers { get; private set; }

        public uint[] Topology { get; private set; }

        public double LearningRate { get; private set; }

        public double Gamma { get; private set; }

        public NeuralNetwork(IActivationFunction funct, uint[] topology, double learningRate = 0.1, double gamma = 0.3)
        {
            LearningRate = learningRate;
            Gamma = gamma;
            Topology = topology;
            ActivationFunction = funct;

            Layers = new NeuralLayer[topology.Length - 1];
            for (int i = 0; i < Layers.Length; i++)
                Layers[i] = new NeuralLayer((int)Topology[i + 1], (int)Topology[i], funct);
        }

        public NeuralNetwork(IActivationFunction funct, double learningRate = 0.1, double gamma = 0.3)
        {
            ActivationFunction = funct;
            LearningRate = learningRate;
            Gamma = gamma;
        }

        public double[] FeedForward(double[] inputs)
        {
            double[] outputs = inputs;

            foreach (NeuralLayer layer in Layers)
                outputs = layer.Calculate(outputs);

            return outputs;
        }

        public override string ToString()
        {
            StringBuilder output = new StringBuilder();

            for (int i = 0; i < Layers.Length; i++)
                output.Append($"Layer_{i}:{Environment.NewLine}{Layers[i]}{Environment.NewLine}");

            return output.ToString();
        }

        public void Train(double[][] trainingInputs, double[][] trainingOutputs, int epochs)
        {
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < trainingInputs.Length; i++)
                {
                    double[] outputs = FeedForward(trainingInputs[i]);
                    Backpropagate(trainingOutputs[i], trainingInputs[i]);
                }
            }
        }

        public void Train(double[] trainingInputs, double[] trainingOutputs, int epochs)
        {
            for (int e = 0; e < epochs; e++)
            {
                double[] outputs = FeedForward(trainingInputs);
                Backpropagate(trainingOutputs, trainingInputs);
            }
        }

        public void UpdateActivatinFunction(IActivationFunction funct)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    Layers[i].SetActivationFunction(j, funct);
                }
            }
        }

        private void Backpropagate(double[] expected, double[] inputs)
        {
            NeuralLayer outLayer = Layers[Layers.Length - 1];
            outLayer.CalculateDeltas(expected);
            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                var currLayer = Layers[i];
                var childLayer = Layers[i + 1];
                var childSums = childLayer.GetDeltasSums(currLayer.Neurons.Length);
                currLayer.CalculateHiddenDeltas(childSums);
            }

            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                var currLayer = Layers[i];
                if (i == 0)
                {
                    currLayer.UpdateWeights(inputs, LearningRate, Gamma);
                }
                else
                {
                    var nextLayer = Layers[i - 1];
                    currLayer.UpdateWeights(nextLayer.GetValues(), LearningRate, Gamma);
                }
            }
        }

        public void SaveWeightsAndBiases(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(string.Join(" ", Topology));

                for (int i = 0; i < Layers.Length; i++)
                {
                    writer.WriteLine($"Layer_{i}");
                    for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    {
                        writer.Write($"Neuron_{j}_Weights ");
                        writer.WriteLine(string.Join(" ", Layers[i].Neurons[j].Weights));
                        writer.Write($"Neuron_{j}_Biases ");
                        writer.WriteLine(string.Join(" ", Layers[i].Neurons[j].Bias));
                    }
                }
            }
        }

        public void LoadWeightsAndBiases(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                var topologyLine = reader.ReadLine();
                var topology = topologyLine.Split(' ').Select(uint.Parse).ToArray();

                Topology = topology;

                Layers = new NeuralLayer[topology.Length - 1];
                for (int i = 0; i < Layers.Length; i++)
                    Layers[i] = new NeuralLayer((int)Topology[i + 1], (int)Topology[i], ActivationFunction);

                for (int i = 0; i < Layers.Length; i++)
                {
                    reader.ReadLine();
                    for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    {
                        var weightsLine = reader.ReadLine().Split(' ');
                        var biasesLine = reader.ReadLine().Split(' ');

                        Layers[i].Neurons[j].SetWeights(weightsLine.Skip(1).Select(double.Parse).ToArray());
                        Layers[i].Neurons[j].SetBias(double.Parse(biasesLine.Last()));
                    }
                }
            }
        }
    }
}

