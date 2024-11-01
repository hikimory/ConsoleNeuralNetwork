using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Utils
{
    public class DigitImage
    {
        public double[] pixels;
        public int label;
        public double[] target;

        public DigitImage(double[] pixels, int label, double[] target)
        {
            this.label = label;
            this.pixels = new double[pixels.Length];
            this.target = new double[target.Length];

            for (int i = 0; i < pixels.Length; ++i)
                    this.pixels[i] = pixels[i];

            for (int i = 0; i < target.Length; i++)
                this.target[i] = target[i];

        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(label.ToString());
            sb.AppendLine();
            sb.Append('[');
            for (int j = 0; j < 10; ++j)
            {
                sb.Append($"{target[j]}"); // white
            }
            sb.Append(']');
            sb.AppendLine();
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    int idx = i * 28 + j;
                    if (pixels[idx] == 0)
                        sb.Append(' '); // white
                    else if (pixels[idx] == 255)
                        sb.Append('O'); // black
                    else
                        sb.Append('.'); // gray
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
