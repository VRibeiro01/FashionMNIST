import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class NetworkTrainerTester {


    public NeuralNetwork neuralNetwork;
    public final List<MNISTDecoder.Fashion> dataSet;
    public final List<MNISTDecoder.Fashion> testDataSet;




    public NetworkTrainerTester( double learningRate, int width, String actFunc) throws IOException {
        dataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/train-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/train-labels-idx1-ubyte");
        testDataSet = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/t10k-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/t10k-labels-idx1-ubyte");
        neuralNetwork = new NeuralNetwork(784,width,10,learningRate,actFunc);
        System.out.println("New Neural Network with " + width + " hidden Neurons and a learning rate of " + learningRate);
        System.out.println("Using " + actFunc + " as activationFunction");
    }

    public static void main(String[] args) throws IOException {
        
        NetworkTrainerTester ntt_2 = new NetworkTrainerTester(0.1, 90, "sigmoid");
        ntt_2.neuralNetwork.trainNetworkWithAnnealing(100, ntt_2.dataSet, 59999);
        new File("testNet4.txt");
        try {
            FileWriter fw = new FileWriter("testNet4.txt");
            double accuracy = ntt_2.testNetwork();
            fw.write(String.valueOf(accuracy));
            fw.close();
        }catch (Exception e){
            System.out.println("Error with filewriter");
        }

    }





    public double validateNetwork() {
        System.out.println("-------------------------------Validating the network-----------------------");
        double tp = 0.0;
        for (int i=40000; i <dataSet.size(); i++ ) {
            double[] inputImage = dataSet.get(i).image;
            double[] targetOutput = dataSet.get(i).label;
            double[] prediction = neuralNetwork.forwardPropagation(inputImage);
            tp+=compareTargetOutputWithPrediction(prediction, targetOutput);
        }
        double acc = calculateAccuracy(tp, dataSet.size()-40000);
        System.out.println(acc);
        System.out.println("________________________________________________________________\n");
        return acc;

    }

    public double testNetwork() {
        System.out.println("-------------------------------Testing the network-----------------------");
        double truePositives = 0.0;
        for (MNISTDecoder.Fashion fashion : testDataSet) {
            double[] inputImage = fashion.image;
            double[] targetOutput = fashion.label;
            double[] prediction = neuralNetwork.forwardPropagation(inputImage);
            truePositives +=compareTargetOutputWithPrediction(prediction, targetOutput);
        }

        double accuracy = calculateAccuracy( truePositives,testDataSet.size());
        System.out.println(accuracy);
        System.out.println("________________________________________________________________\n");
        return accuracy;

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
