using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Utils
{
    public static class Classification
    {
        private static readonly double[][] targets = new double[10][];

        static Classification()
        {
            for (int i = 0; i < 10; i++)
            {
                targets[i] = new double[10];
                targets[i][i] = 1;
            }
        }

        public static double[] GetTarget(int num)
        {
            if (num >= 0 && num < targets.Length)
            {
                return targets[num];
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(num), "Индекс должен быть в диапазоне от 0 до 9");
            }
        }
    }
}
