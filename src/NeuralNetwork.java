import java.io.IOException;
import java.util.*;

public class NeuralNetwork {
    public List<ArrayList<HashMap<String,double[]>>> network;
    public ArrayList<HashMap<String,double[]>> hiddenLayer;
    public ArrayList<HashMap<String,double[]>> outputLayer;
    public double learningRate;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate){
        this.learningRate = learningRate;
        this.network = new ArrayList<>();

        // Bias in der letzten Spalte
        hiddenLayer = new ArrayList<>();
        initialize(hiddenLayer, hiddenNeurons, inputNeurons);
        network.add(hiddenLayer);

        // Bias in der letzten Spalte
        this.outputLayer = new ArrayList<>();;
        initialize(outputLayer,outputNeurons,hiddenNeurons);
        network.add(outputLayer);

    }

    private void initialize(ArrayList<HashMap<String,double[]>> layer, int layerSize, int inputNeuronCount) {
        Random rand = new Random();
        for(int i =0; i < layerSize; i++){
            HashMap<String, double[]> neuron= new HashMap<>();
            double[] weight = new double[inputNeuronCount+1];
            for(int j=0; j < weight.length; j++){
                weight[j] = rand.nextDouble();
            }
            neuron.put("weights",weight);
            layer.add(neuron);

        }

    }

    // Berechnet die Eingabe eines Neurons: Summe der Multiplikation der Gewichte und der Eingabe + Bias
    private double calculateInput(double[] weights, double[] inputs){
        //Bias dazu addieren
        double neuronInput = weights[weights.length-1];

        for(int i=0; i < weights.length-1 ; i++){
            neuronInput += weights[i] * inputs[i];
        }
        return neuronInput;
    }

    // Berechnet Ausgabe des Neurons mit Sigmoid als Aktivierungsfunktion
    private double activate(double neuronInput){
        return 1.0 / (1.0 + Math.exp(-neuronInput));
    }

    // Forward-Propagation, um Ausgabe des Netzes zu produzieren
    public double[] forwardPropagation(double inputFromData[]){
        double[] input = inputFromData;
        int i;
        for(ArrayList<HashMap<String,double[]>> layer: network){
            i=0;
            double[] new_inputs = new double[layer.size()];
            for(HashMap<String,double[]> neuron : layer ){
               double neuronInput = calculateInput(neuron.get("weights"), input);
               double output = activate(neuronInput);
               neuron.put("output", new double[]{output});
               new_inputs[i] = output;
               i++;
            }
            input = new_inputs;
        }
        return input;
    }

    private double activationDerivative(double output){
        return output*(1.0 - output);
    }

    private void backpropagate(double[] expected){
        double error;
        // RÜckwärts über Netzschichten iterieren
        for(int i=network.size()-1; i >=0; i--){
            ArrayList<HashMap<String,double[]>> layer = network.get(i);
            List<Double> errors= new ArrayList<>();

            // Fehler in verteckten Schicht berechnen
            if(i != network.size()-1){
                for(int j=0; j < layer.size(); j++){
                     error = 0.0;
                    for(HashMap<String,double[]> neuron : network.get(i+1)){
                        error += (neuron.get("weights")[j] * neuron.get("delta")[0]);
                    }
                    errors.add(error);
                }
            } else {
                for(int j=0; j < layer.size(); j++){
                    HashMap<String,double[]> neuron = layer.get(j);
                    errors.add(neuron.get("output")[0] - expected[j]);
                }
            }
            for(int j=0; j < layer.size(); j++){
                HashMap<String,double[]> neuron = layer.get(j);
                neuron.put("delta",new double[]{errors.get(j)*activationDerivative(neuron.get("output")[0])});
            }
        }
    }

    public void updateWeights(double[] inputFromData){
        double[] inputs;
        for(int i=0; i < network.size(); i++){
            // Input für Eingabeschicht ist Eingabe vom Datensatz
            inputs = inputFromData;
            if(i!=0){
                inputs = new double[network.get(i-1).size()];
                int idx =0;

                // Inputwerte aus vorhergehende Schicht ermitteln
                for(HashMap<String, double[]> neuron : network.get(i-1)){
                    inputs[idx] = neuron.get("output")[0];
                    idx++;
                }
            }
            // Über Neuronen iterieren und Gewichte und Bias aktualisieren
            for(HashMap<String, double[]> neuron : network.get(i)){
                for(int j=0;  j<inputs.length; j++){
                    neuron.get("weights")[j] -= learningRate*neuron.get("delta")[0]*inputs[j];
                }
                neuron.get("weights")[neuron.get("weights").length-1] -= learningRate*neuron.get("delta")[0];
            }
        }
    }

    public void trainNetwork(int epochs, List<MNISTDecoder.Fashion> trainingDataSet,int trainingDataSize){
        for(int i=0; i <= epochs; i++){
            double sumError = 0;
            for(int input=0; input < trainingDataSize; input++){
                double[] image = trainingDataSet.get(input).image;
                double[] prediction = forwardPropagation(image);
                double[] expected = trainingDataSet.get(input).label;
                sumError += computeError(prediction, expected);
                backpropagate(expected);
                updateWeights(image);

            }
            System.out.println("Epoch: " + i +", lRate: " + learningRate +
                    ", Error: " + sumError);
        }
    }

    private double computeError(double[] prediction, double[] expected){
        double error=0;
        for(int i=0; i < prediction.length; i++){
            error += Math.pow(expected[i] - prediction[i],2);
        }
        return error;
    }

    public static void main(String[] args) throws IOException {
        NeuralNetwork net = new NeuralNetwork(784,10,10,0.01);
        List<MNISTDecoder.Fashion> data = MNISTDecoder.loadDataSet(System.getProperty("user.dir") + "/resources/train-images-idx3-ubyte",System.getProperty("user.dir") + "/resources/train-labels-idx1-ubyte");
        net.trainNetwork(900,data,100);

       /*for(int i=0; i < net.network.size(); i++){
            ArrayList<HashMap<String, double[]>> layer = net.network.get(i);
            System.out.println("layer " + i);
            for(int j=0; j < layer.size(); j++) {
                HashMap<String,double[]> neuron = layer.get(j);
                System.out.println("neuron " + j);
               // System.out.println("weights: " + Arrays.toString(neuron.get("weights") )+ "\n");
            }
        }*/



        /*for(int i=0; i < net.network.size(); i++){
            ArrayList<HashMap<String, double[]>> layer = net.network.get(i);
            System.out.println("layer " + i);
            for(int j=0; j < layer.size(); j++) {
                HashMap<String,double[]> neuron = layer.get(j);
                System.out.println("neuron " + j);
                System.out.println("weights: " + "\n");
                System.out.println("delta: " + Arrays.toString(neuron.get("delta") )+ "\n");
            }
        }*/
    }
}
