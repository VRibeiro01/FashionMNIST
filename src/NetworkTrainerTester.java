import java.io.IOException;
import java.util.*;

public class NetworkTrainerTester {

    public NeuralNetwork neuralNetwork;
    public final List<MNISTDecoder.Fashion> dataSet;
    public final List<MNISTDecoder.Fashion> testDataSet;
    public int epochs;
    public int sizeTestData;
    public double sizeTrainingsData;
    private double tp;

    public NetworkTrainerTester(String activationFunction, int epochs, double learningRate, int width, double sizeTrainingsData) throws IOException {
        dataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/train-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/train-labels-idx1-ubyte");
        testDataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/t10k-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/t10k-labels-idx1-ubyte");
        sizeTestData = testDataSet.size();
        neuralNetwork = new NeuralNetwork(784,width,10,activationFunction,learningRate,sizeTrainingsData);
        this.epochs = epochs;
        System.out.println("New Neural Network with " + width + " hidden Neurons and a learning rate of " + learningRate);
        System.out.println("Using " + activationFunction + " as activationFunction");
        this.sizeTrainingsData = sizeTrainingsData;
    }

    public static void main(String[] args) throws IOException {

        NetworkTrainerTester ntt = new NetworkTrainerTester("ReLu",2,0.5,1568,60000);

        ntt.batchTrain();
        ntt.testNetwork();

        }



    public void batchTrain() {

        for(int i=0; i<epochs;i++){
           double cost =0.0;

            for(int j=0; j<sizeTrainingsData;j++){
                double[] inputImage = dataSet.get(j).image;
                double[] targetOutput = dataSet.get(j).label;

                double[] prediction =neuralNetwork.makePrediction(inputImage);
                cost += costFunction(prediction, targetOutput);
//                System.out.println("Prediction: " + Arrays.toString(prediction));
//                System.out.println("Target Output: " + Arrays.toString(targetOutput) + "\n");

                neuralNetwork.backpropagateError(targetOutput);

                neuralNetwork.sumUpCorrectionValues(inputImage);
            }
            cost = 1.0/(2*sizeTrainingsData) * cost;
            neuralNetwork.batchUpdate(sizeTrainingsData);


            System.out.println("Error: " + cost);
            System.out.println("________________________________________________________________\n");

        }

    }

    public void testNetwork() {

        for (MNISTDecoder.Fashion fashion : testDataSet) {
            double[] inputImage = fashion.image;
            double[] targetOutput = fashion.label;
            double[] prediction = neuralNetwork.makePrediction(inputImage);
            compareTargetOutputWithPrediction(prediction, targetOutput);
        }

        System.out.println(calculateAccuracy());
        System.out.println("________________________________________________________________\n");

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

    public String getClass(MNISTDecoder.Fashion data, double[] prediction){
        String[] classNames = new String[]{"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

        ArrayList<Double> list=new ArrayList<>();
        for(int i=0; i<prediction.length;i++){
            list.add(prediction[i]);
        }
        int predictedLabel = list.indexOf(Collections.max(list));
        return classNames[predictedLabel];
    }

    public double calculateAccuracy() {
        return tp / sizeTestData;
    }

    public void compareTargetOutputWithPrediction(double[] prediction, double[] targetOutput) {
        int indexOfCorrectLabel = getMax(targetOutput);
        int indexOfMaxPrediction = getMax(prediction);

        if (indexOfCorrectLabel == indexOfMaxPrediction) {
            tp++;
        }
    }

    public int getMax(double[] array) {
        int max = 0;
        for ( int i = 1; i < array. length; i++ )
        {
            if ( array[i] > array[max] ) max = i;
        }
        return max;
    }



}
