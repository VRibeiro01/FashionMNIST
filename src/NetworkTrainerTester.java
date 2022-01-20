import java.io.IOException;
import java.util.*;

public class NetworkTrainerTester {

    public NeuralNetwork neuralNetwork;
    public final List<MNISTDecoder.Fashion> dataSet;
    public final List<MNISTDecoder.Fashion> testDataSet;




    public NetworkTrainerTester(int epochs, double learningRate, int width) throws IOException {
        dataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/train-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/train-labels-idx1-ubyte");
        testDataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/t10k-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/t10k-labels-idx1-ubyte");
        neuralNetwork = new NeuralNetwork(784,width,10,learningRate);
        System.out.println("New Neural Network with " + width + " hidden Neurons and a learning rate of " + learningRate);
        System.out.println("Using Sigmoid as activationFunction");
    }

    public static void main(String[] args) throws IOException {

        NetworkTrainerTester ntt = new NetworkTrainerTester(300,0.5,50);



        ntt.neuralNetwork.trainNetwork(300,ntt.dataSet, 10000);

        //ntt.validateNetwork();
        ntt.testNetwork();

        }





    public void validateNetwork() {
        System.out.println("-------------------------------Validating the network-----------------------");
        double tp = 0.0;
        for (int i=40000; i <dataSet.size(); i++ ) {
            double[] inputImage = dataSet.get(i).image;
            double[] targetOutput = dataSet.get(i).label;
            double[] prediction = neuralNetwork.forwardPropagation(inputImage);
            tp+=compareTargetOutputWithPrediction(prediction, targetOutput);
        }

        System.out.println(calculateAccuracy(tp, dataSet.size()-40000));
        System.out.println("________________________________________________________________\n");

    }

    public void testNetwork() {
        System.out.println("-------------------------------Testing the network-----------------------");
        double truePositives = 0.0;
        for (MNISTDecoder.Fashion fashion : testDataSet) {
            double[] inputImage = fashion.image;
            double[] targetOutput = fashion.label;
            double[] prediction = neuralNetwork.forwardPropagation(inputImage);
            truePositives +=compareTargetOutputWithPrediction(prediction, targetOutput);
        }

        System.out.println(calculateAccuracy( truePositives,testDataSet.size()));
        System.out.println("________________________________________________________________\n");

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

    public double calculateAccuracy(double truePositives,int sizeTestData) {
        return truePositives / sizeTestData;
    }

    public double compareTargetOutputWithPrediction(double[] prediction, double[] targetOutput) {
        int indexOfCorrectLabel = getMax(targetOutput);
        int indexOfMaxPrediction = getMax(prediction);

        System.out.println("Expected: " + indexOfCorrectLabel );
        System.out.println("Actual: " + indexOfMaxPrediction);
        if (indexOfCorrectLabel == indexOfMaxPrediction) {
            return 1.0;
        }
        return 0.0;
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
