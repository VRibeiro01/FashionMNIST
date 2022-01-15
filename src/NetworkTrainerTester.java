import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NetworkTrainerTester {

    public final NeuralNetwork neuralNetwork;
    public final List<MNISTDecoder.Fashion> dataSet;
    public int epochs;
    public double sizeTrainingsData;

    public NetworkTrainerTester(String activationFunction, int epochs, double learningRate, int width, double sizeTrainingsData) throws IOException {
        dataSet = MNISTDecoder.loadDataSet("C:\\Users\\vivia\\IdeaProjects\\is_lernen\\resources\\t10k-images-idx3-ubyte","C:\\Users\\vivia\\IdeaProjects\\is_lernen\\resources\\t10k-labels-idx1-ubyte");
        neuralNetwork = new NeuralNetwork(784,width,10,activationFunction,learningRate,sizeTrainingsData);
        this.epochs = epochs;
        System.out.println("New Neural Network with " + width + " hidden Neurons and a learning rate of " + learningRate);
        System.out.println("Using " + activationFunction + " as activationFunction");
        this.sizeTrainingsData = sizeTrainingsData;
    }

    public static void main(String[] args) throws IOException {
        new NetworkTrainerTester("sigmoid",30,0.3,3,10000).batchTrain();

        }



    public void batchTrain() {

        for(int i=0; i<epochs;i++){
           double cost =0.0;

            for(int j=0; j<sizeTrainingsData;j++){
                double[] inputImage = dataSet.get(j).image;
                double[] targetOutput = dataSet.get(j).label;

                double[] prediction =neuralNetwork.makePrediction(inputImage);
                cost += costFunction(prediction, targetOutput);
                System.out.println("Prediction: " + Arrays.toString(prediction));
                System.out.println("Target Output: " + Arrays.toString(targetOutput) + "\n");

                neuralNetwork.backpropagateError(targetOutput);

                neuralNetwork.sumUpCorrectionValues(inputImage);
            }
            cost = 1.0/(2*sizeTrainingsData) * cost;
            neuralNetwork.batchUpdate(sizeTrainingsData);


            System.out.println("Error: " + cost);
            System.out.println("________________________________________________________________\n");

        }

    }

    public void stochasticTrain() {
        for (int i = 0; i < epochs; i++) {
            double cost = 0.0;
            Collections.shuffle(dataSet);
            for (int j = 0; j < sizeTrainingsData; j++) {
                double[] inputImage = dataSet.get(j).image;
                double[] targetOutput = dataSet.get(j).label;

                double[] prediction = neuralNetwork.makePrediction(inputImage);
                cost += costFunction(prediction, targetOutput);
                System.out.println("Prediction: " + Arrays.toString(prediction));
                 System.out.println("Target Output: " + Arrays.toString(targetOutput) + "\n");

                neuralNetwork.backpropagateError(targetOutput);

                neuralNetwork.stochasticUpdate(inputImage);
            }
            cost = 1.0 / (2 * sizeTrainingsData) * cost;


            System.out.println("Error: " + cost);
            System.out.println("________________________________________________________________\n");

        }
    }


        /**
         * Diese Methode berechnet die Kostenfunktion. Diese berechnet den Fehler in der Vorhersage(Methode makePrediction) des Netzes
         * Die Kosten werden anhand der quadratischen Kostenfunktion berechnet
         *
         * @return
         */
    public double costFunction(double[] prediction, double[] expectedResults) {

         return    neuralNetwork.computeVectorLength(  Matrix.subtract(expectedResults, prediction) );


    }



}
