import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class Tests {
    private static double[][] weightsHiddenToOutput;
    private static double[][] weightsInputToHidden;
    private static NeuralNetwork neuralNetwork;
    private static double[] input;
    private static double[] expectedOutput;
    private static double learningRate;

    @BeforeEach
    public void initialize(){
        input = new double[]{-0.8,1};
        expectedOutput = new double[]{-1,0};
        weightsInputToHidden = new double[][]{
                {-1, 2},
                {0, -0.5},
                {1.5, 1}
        };
        weightsHiddenToOutput = new double[][]{
                {0.4,1,-1},
                {0.3,-0.5,-0.1}
        };
        double[][] biasHidden = new double[3][1];
        double[][] biasOutput = new double[2][1];
        learningRate = 0.3;
        neuralNetwork = new NeuralNetwork(2,3,2, weightsInputToHidden,weightsHiddenToOutput, biasHidden, biasOutput,learningRate,input);
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
        Assertions.assertThrows(IllegalArgumentException.class, ()-> Matrix.addMatrices(firstMatrix,thirdMatrix));

    }

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
        Assertions.assertThrows(IllegalArgumentException.class, ()-> Matrix.multiplyMatrices(firstMatrix,thirdMatrix));

    }

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

    }


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
       double[] prediction = neuralNetwork.makePrediction();
       double[][] finalInput = Matrix.multiplyMatrices(weightsHiddenToOutput,new double[][]{{2.8},{0},{0}});
       Assertions.assertEquals(finalInput[0][0], prediction[0],0.001);
        Assertions.assertEquals(finalInput[1][0], prediction[1],0.001);
    }

    @Test
    public void derivativeActivationFunction(){
        neuralNetwork.setActivationFunction("ReLu");
       Assertions.assertEquals(1,neuralNetwork.derivativeActivationFunction(2.0));
        Assertions.assertEquals(0,neuralNetwork.derivativeActivationFunction(-2.0));

        neuralNetwork.setActivationFunction("tanh");
        Assertions.assertEquals(1-(Math.pow(Math.tanh(1.0),2)),neuralNetwork.derivativeActivationFunction(1.0));

        neuralNetwork.setActivationFunction("sigmoid");
        Assertions.assertEquals(neuralNetwork.sigmoid(2)*(1-neuralNetwork.sigmoid(2)),neuralNetwork.derivativeActivationFunction(2.0));



    }

    @Test
    public void computeOutputError() {
        neuralNetwork.makePrediction();
        System.out.println(Arrays.toString(neuralNetwork.prediction));
        double[] error = new double[]{1.1199999999999999 - (-1.0), 0.84 - 0};
        for (int i = 0; i < error.length; i++) {
            error[i] = error[i] * neuralNetwork.derivativeActivationFunction(neuralNetwork.inputFinalLayer[i][0]);
        }
        double[][] gradientMatrix = Matrix.transposeMatrix(new double[][]{error});
        neuralNetwork.computeOutputError(expectedOutput);
        System.out.println("expected: " + Arrays.deepToString(gradientMatrix)+"\n"+ "Actual: " + Arrays.deepToString(neuralNetwork.outputGradients));
        Assertions.assertTrue(Arrays.deepEquals(neuralNetwork.outputGradients, gradientMatrix));
    }

    @Test
    public void backPropagateError(){
      neuralNetwork.makePrediction();
      System.out.println("Prediction: "+Arrays.toString(neuralNetwork.prediction));
      neuralNetwork.backpropagateError(expectedOutput);
      System.out.println("Output Layer Gradient: " + Arrays.deepToString(neuralNetwork.outputGradients));
      double[][] transposedWeights = Matrix.transposeMatrix(weightsHiddenToOutput);
        System.out.println("Transposed Weights: "+Arrays.deepToString(transposedWeights));
       double[][] multipliedByGradients = Matrix.multiplyMatrices(transposedWeights,neuralNetwork.outputGradients);
        System.out.println("Multiplied By Gradient: "+Arrays.deepToString(multipliedByGradients));
        for(int i=0; i < multipliedByGradients.length; i++){
            multipliedByGradients[i][0] = multipliedByGradients[i][0] * neuralNetwork.derivativeActivationFunction(neuralNetwork.inputHiddenLayer[i][0]);
        }

        System.out.println("Expected Gradient: " +  Arrays.deepToString(multipliedByGradients));
        System.out.println("Actual Gradient: "+Arrays.deepToString(neuralNetwork.hiddenGradients));
        Assertions.assertTrue(Arrays.deepEquals(multipliedByGradients,neuralNetwork.hiddenGradients));

    }

    @Test
    public void updateInputToHiddenWeights(){
        neuralNetwork.makePrediction();
        neuralNetwork.backpropagateError(expectedOutput);
       double inputIn1Hidden1 = input[0] * weightsInputToHidden[0][0];
       double newWeightIn1ToHidden1 =  weightsInputToHidden[0][0] - learningRate * (inputIn1Hidden1 - neuralNetwork.hiddenGradients[0][0]);
       System.out.println("Old Weights From Input to Hidden Layer: \n" + Arrays.deepToString(weightsInputToHidden));
       neuralNetwork.updateWeights();
       System.out.println("Expected Weight From Input-Neuron 1 to Hidden-Neuron 1: " + newWeightIn1ToHidden1);
        System.out.println("Actual Weight From Input-Neuron 1 to Hidden-Neuron 1: " + weightsInputToHidden[0][0]);
        System.out.println("New Weights Input to Hidden Layer  \n" + Arrays.deepToString(weightsInputToHidden));
        Assertions.assertEquals(newWeightIn1ToHidden1,weightsInputToHidden[0][0]);





    }

    @Test
    public void updateHiddenToOutputWeights(){
        neuralNetwork.makePrediction();
        neuralNetwork.backpropagateError(expectedOutput);
        double inputHidden1Out1 = neuralNetwork.outputHiddenLayer[0][0] * weightsHiddenToOutput[0][0];
        double newWeightHidden1ToOut1 =  weightsHiddenToOutput[0][0] - learningRate * (inputHidden1Out1 - neuralNetwork.outputGradients[0][0]);

        System.out.println("Old Weights From  Hidden Layer to Output Layer: \n" + Arrays.deepToString(weightsHiddenToOutput));
        neuralNetwork.updateWeights();
        System.out.println("Expected Weight From Hidden-Neuron 1 to Output-Neuron 1: " + newWeightHidden1ToOut1);
        System.out.println("Actual Weight From Input-Neuron 1 to Hidden-Neuron 1: " + weightsHiddenToOutput[0][0]);
        System.out.println("New Weights Hidden Layer to Output Layer  \n" + Arrays.deepToString(weightsHiddenToOutput));
        Assertions.assertEquals(newWeightHidden1ToOut1 ,weightsHiddenToOutput[0][0]);
    }

}
