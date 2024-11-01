using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.Utils;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            XOR_Task();
            DigitRecognition_Task();
        }

        private static void XOR_Task()
        {
            NeuralNetwork nn = new NeuralNetwork(new SigmoidActivationFunction(), new uint[] { 2, 2, 1 });
            double[][] trainingInputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] trainingOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            nn.Train(trainingInputs, trainingOutputs, 10000);

            Console.WriteLine("XOR Task");
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                var result = nn.FeedForward(trainingInputs[i]);
                var val = Math.Round(result[0]);
                Console.WriteLine($"Input: [{trainingInputs[i][0]}, {trainingInputs[i][1]}] Result: {val}");
            }
        }

        private static void DigitRecognition_Task()
        {
            int epochs = 20;
            int trainingAmount = 60000;
            MNIST_Parser parser = new MNIST_Parser("../../../DataSets/train-images.idx3-ubyte", 
                                                   "../../../DataSets/train-labels.idx1-ubyte", trainingAmount);
            List<DigitImage> trainingImages = parser.Images;

            NeuralNetwork nn = new NeuralNetwork(new SigmoidActivationFunction(), new uint[] { 784, 128, 64, 10 });

            double[][] trainingInputs = new double[trainingAmount][];
            double[][] trainingOutputs = new double[trainingAmount][];
            for (int i = 0; i < trainingAmount; i++)
            {
                trainingInputs[i] = trainingImages[i].pixels;
                trainingOutputs[i] = trainingImages[i].target;
            }

            //nn.Train(trainingInputs, trainingOutputs, epochs);
            //nn.SaveWeightsAndBiases("DigitRecognition_Task.txt");
            nn.LoadWeightsAndBiases("DigitRecognition_Task.txt");

            Console.WriteLine("\nDigitRecognition_Task");
            for (int i = 0; i < 4; i++)
            {
                var result = nn.FeedForward(trainingImages[i].pixels);
                int predictedClass = result.ToList().IndexOf(result.Max());
                Console.WriteLine($"Input: {trainingImages[i].label} Result: {predictedClass}");
            }
        }
    }
}