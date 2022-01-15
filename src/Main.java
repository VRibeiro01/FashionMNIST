import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        List<MNISTDecoder.Fashion> e = MNISTDecoder.loadDataSet("C:\\Users\\vivia\\IdeaProjects\\is_lernen\\resources\\t10k-images-idx3-ubyte","C:\\Users\\vivia\\IdeaProjects\\is_lernen\\resources\\t10k-labels-idx1-ubyte");
        double[] input = e.get(0).image;
        double[] expected = e.get(0).label;
        double[] prediction;
        double error;
        NeuralNetwork nn = new NeuralNetwork(input.length, 5,expected.length,0.5,input);
        //System.out.println("Input: " + Arrays.toString(input));
        System.out.println("Output to Learn: " + Arrays.toString(expected));
        prediction = nn.makePrediction();
        System.out.println("Prediction: " + Arrays.toString(prediction));
        nn.backpropagateError(expected);
        //System.out.println("Hidden Gradient: " + Arrays.deepToString(nn.hiddenGradients));
        //System.out.println("Output Gradient: " + Arrays.deepToString(nn.outputGradients));
        nn.updateWeightsAndBiases();
        //System.out.println("Hidden Weights: " + Arrays.deepToString(nn.weightsInputToHidden));
      //  System.out.println("Output Weights: " + Arrays.deepToString(nn.weightsHiddenToOutput));
        System.out.println("___________________________________________________________________________");
        for(int i=0; i < 50; i++) {
            prediction = nn.makePrediction();
            System.out.println("Prediction: " + Arrays.toString(prediction));
            error = nn.costFunction(expected);
            System.out.println("Output to Learn: " + Arrays.toString(expected));
            System.out.println("Error: " + error);
            error = nn.costFunction(expected);
            nn.backpropagateError(expected);
           // System.out.println("Hidden Gradient: " + Arrays.deepToString(nn.hiddenGradients));
           // System.out.println("Output Gradient: " + Arrays.deepToString(nn.outputGradients));
            nn.updateWeightsAndBiases();
           // System.out.println("Hidden Weights: " + Arrays.deepToString(nn.weightsInputToHidden));
           // System.out.println("Output Weights: " + Arrays.deepToString(nn.weightsHiddenToOutput));
            System.out.println("___________________________________________________________________________");
        }


    }


}
