using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Utils
{
    public class MNIST_Parser
    {
        public List<DigitImage> Images { get; private set; }

        public MNIST_Parser(string imageFilePath, string labelFilePath, int amount)
        {
            Images = new List<DigitImage>();
            LoadData(imageFilePath, labelFilePath, amount);
        }

        private void LoadData(string imageFilePath, string labelFilePath, int amount)
        {
            using (var imageStream = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read))
            using (var labelStream = new FileStream(labelFilePath, FileMode.Open, FileAccess.Read))
            using (var imageReader = new BinaryReader(imageStream))
            using (var labelReader = new BinaryReader(labelStream))
            {
                var magicNumberImages = imageReader.ReadInt32();
                var numberOfImages = ReverseBytes(imageReader.ReadInt32());
                var numberOfRows = ReverseBytes(imageReader.ReadInt32());
                var numberOfColumns = ReverseBytes(imageReader.ReadInt32());

                var magicNumberLabels = labelReader.ReadInt32();
                var numberOfLabels = ReverseBytes(labelReader.ReadInt32());

                if (numberOfImages != numberOfLabels)
                    throw new Exception("Количество изображений не соответствует количеству меток.");

                numberOfImages = amount > numberOfImages ? numberOfImages : amount;
                magicNumberLabels = amount > magicNumberLabels ? magicNumberLabels : amount;

                int len = numberOfRows * numberOfColumns;
                double[] pixels = new double[len];

                for (int i = 0; i < numberOfImages; i++)
                {
                    for (int j = 0; j < len; j++)
                    {
                        pixels[j] = imageReader.ReadByte() / 255.0;
                    }
                    int label = labelReader.ReadByte();
                    Images.Add(new DigitImage(pixels, label, Classification.GetTarget(label)));
                }
            }
        }

        private static int ReverseBytes(int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes);
        }
    }
}
