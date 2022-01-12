import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class Tests {
    private static double[][] weightsInputToHidden;
    private static double[][] weightsHiddenToOutput;
    private static double[][] biasHidden;
    private static double[][] biasOutput;
    private static NeuralNetwork neuralNetwork;
    private static double[] input;
    private static double[] expectedOutput;

    @BeforeEach
    public void initialize(){
        input = new double[]{-0.8,1};
        expectedOutput = new double[]{-1,0};
        weightsInputToHidden = new double[][]{
                {-1,2},
                {0,-0.5},
                {1.5,1}
        };
        weightsHiddenToOutput = new double[][]{
                {0.4,1,-1},
                {0.3,-0.5,-0.1}
        };
        biasHidden = new double[3][1];
        biasOutput = new double[2][1];
        neuralNetwork = new NeuralNetwork(2,3,2,weightsInputToHidden,weightsHiddenToOutput,biasHidden,biasOutput,1);
    }

    @Test
    public void addMatrices(){

        double[][] firstMatrix  = new double[][]{
                {1.0,0.0,-1.0},
                {2.0,1.0,0.0}
        };

        double[][] secondMatrix  = new double[][]{
                {0.01,-2.0,1.0},
                {0.0,0.0,0.0}
        };

        double[][] thirdMatrix = new double[][]{
                {0.0,1.8,2.0,4.0},
                {1.3,2.0,-0.0,8.0}
        };

        double[][] sum = Matrix.addMatrices(firstMatrix,secondMatrix);
        Assertions.assertEquals(1.01,sum[0][0],0.001);
        Assertions.assertEquals(-2.0,sum[0][1],0.001);
        Assertions.assertEquals(0.0,sum[0][2],0.001);
        Assertions.assertEquals(2.0,sum[1][0],0.001);
        Assertions.assertEquals(1.0,sum[1][1],0.001);
        Assertions.assertEquals(0.0,sum[1][2],0.001);
        Assertions.assertThrows(IllegalArgumentException.class, ()->{Matrix.addMatrices(firstMatrix,thirdMatrix);});

    };

    @Test
    public void subtract(){
        double[] A = new double[]{2.0,-1.5,1.0};
        double[] B = new double[]{3.0,-1.0,0.0};
        double[] C = Matrix.subtract(A,B);
        Assertions.assertEquals(-1.0,C[0],0.01);
        Assertions.assertEquals(-0.5,C[1],0.01);
        Assertions.assertEquals(1.0,C[2],0.01);

    }

    @Test
    public void multiplyMatrices(){
        double[][] firstMatrix  = new double[][]{
                {1.0,2.0,3.0},
                {3.0,1.0,1.0}
        };

        double[][] secondMatrix  = new double[][]{
                {2.0,1.0},
                {1.0,2.0},
                {2.0,1.0}

        };

        double[][] thirdMatrix = new double[][]{
                {0.0,1.8,2.0,4.0},
                {1.3,2.0,-0.0,8.0}
        };

        double[][] product = Matrix.multiplyMatrices(firstMatrix,secondMatrix);
        Assertions.assertEquals(10.0,product[0][0],0.001);
        Assertions.assertEquals(8.0,product[0][1],0.001);
        Assertions.assertEquals(9.0,product[1][0],0.001);
        Assertions.assertEquals(6.0,product[1][1],0.001);
        Assertions.assertThrows(IllegalArgumentException.class, ()->{Matrix.multiplyMatrices(firstMatrix,thirdMatrix);});

    };

    @Test
    public void activationFunction(){

        double[][] input = new double[][]{
                {-0.8},
                {2.0}
        };
        neuralNetwork.setActivationFunction("ReLu");

        double[][] result = neuralNetwork.activationFunction(input);
        Assertions.assertEquals(0.0,result[0][0],0.001);
        Assertions.assertEquals(2.0,result[1][0],0.001);

        neuralNetwork.setActivationFunction("tanh");
        input = new double[][]{
                {-0.8},
                {2.0}
        };
        double [][] result2 = neuralNetwork.activationFunction(input);

        Assertions.assertEquals(Math.tanh(-0.8),result2[0][0],0.001);
        Assertions.assertEquals(Math.tanh(2.0),result2[1][0],0.001);

        neuralNetwork.setActivationFunction("sigmoid");
        input = new double[][]{
                {-0.8},
                {2.0}
        };

        result = neuralNetwork.activationFunction(input);
        Assertions.assertEquals(1/ (1 + Math.exp(-(-0.8))),result[0][0],0.001);
        Assertions.assertEquals(1/ (1 + Math.exp(-2.0)),result[1][0],0.001);

    };

    @Test
    public void computeVectorLength(){
        double[] vector = new double[]{-1.0,3.0};
        double result = neuralNetwork.computeVectorLength(vector);
        double expectedResult = Math.sqrt(10);
        Assertions.assertEquals(expectedResult, result,0.001);

        vector = new double[]{1.0,2.0,3.0};
        result = neuralNetwork.computeVectorLength(vector);
        expectedResult = Math.sqrt(14);
        Assertions.assertEquals(expectedResult, result,0.001);

    };

    @Test
    public void transposeMatrix(){
        double[][] transposedMatrix = Matrix.transposeMatrix(new double[][]{input});
        Assertions.assertEquals(transposedMatrix[0][0],input[0],0.001);
        Assertions.assertEquals(transposedMatrix[1][0],input[1],0.001);
        Assertions.assertEquals(2,transposedMatrix.length);
        Assertions.assertEquals(1,transposedMatrix[0].length);
    }

    @Test
    public void makePrediction(){
       double[] prediction = neuralNetwork.makePrediction(input);
       double[][] finalInput = Matrix.multiplyMatrices(weightsHiddenToOutput,new double[][]{{2.8},{0},{0}});
       Assertions.assertEquals(finalInput[0][0], prediction[0],0.001);
        Assertions.assertEquals(finalInput[1][0], prediction[1],0.001);
    };

    @Test
    public void costFunction(){
      double[] expectedOutput = new double[]{-0.5,2.0,1.0};
      double[] prediction = new double[]{2.0,1.9,0.0};

      double cost = 6.25+0.01+1.0;
      double len = Math.sqrt(cost);
      double costPow = 1.0/2 * Math.pow(len,2);


    // Assertions.assertEquals(costPow, neuralNetwork.costFunction(expectedOutput,prediction));
    };
}
