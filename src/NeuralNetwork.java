import java.util.*;

public class NeuralNetwork {
    public List<ArrayList<HashMap<String,double[]>>> network;
    public ArrayList<HashMap<String,double[]>> hiddenLayer;
    public ArrayList<HashMap<String,double[]>> outputLayer;
    public double learningRate;
    public String activationFunction;
    public String lossFunction;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate, String activationFunction){
        this.learningRate = learningRate;
        this.network = new ArrayList<>();
        this.activationFunction = activationFunction;

        // Bias in der letzten Spalte
        hiddenLayer = new ArrayList<>();
        initialize(hiddenLayer, hiddenNeurons, inputNeurons);
        network.add(hiddenLayer);

        // Bias in der letzten Spalte
        this.outputLayer = new ArrayList<>();
        initialize(outputLayer,outputNeurons,hiddenNeurons);
        network.add(outputLayer);

    }

    private void initialize(ArrayList<HashMap<String,double[]>> layer, int layerSize, int inputNeuronCount) {
        Random rand = new Random();
        for(int i =0; i < layerSize; i++){
            HashMap<String, double[]> neuron= new HashMap<>();
            double[] weight = new double[inputNeuronCount+1];
            for(int j=0; j < weight.length; j++){
                //Änderungen gem. Stelldinger
                weight[j] = (rand.nextDouble() -0.5) / layerSize;
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

    // Berechnet Ausgabe des Neurons anhand der eingegebenen Aktivierungsfunktion
    private double activate(double neuronInput){
        if(activationFunction.equalsIgnoreCase("Sigmoid"))
        return 1.0 / (1.0 + Math.exp(-neuronInput));
        else if(activationFunction.equalsIgnoreCase("tanh"))
            return Math.tanh(neuronInput);
        else if(activationFunction.equalsIgnoreCase("ReLu"))
            return Math.max(0.0, neuronInput);
            // Sigmoid als Default
            return 1.0 / (1.0 + Math.exp(-neuronInput));
    }

    // Forward-Propagation, um Ausgabe des Netzes zu produzieren
    public double[] forwardPropagation(double[] inputFromData){
        double[] input = inputFromData;
        int i;

        // Iteriert über alle Schichten
        for(ArrayList<HashMap<String,double[]>> layer: network){
            i=0;
            double[] new_inputs = new double[layer.size()];

            // Berechnet die Ausgabe von jedem Neuron. Die Ausgabe wird im Neuron unter Schlüssel "output" eingetragen
            for(HashMap<String,double[]> neuron : layer ){
               double neuronInput = calculateInput(neuron.get("weights"), input);
               double output = activate(neuronInput);
               neuron.put("output", new double[]{output});

               // Ausgabe der Neuronen der aktuellen Schicht werden zur Eingabe der nächsten Schicht
               new_inputs[i] = output;
               i++;
            }
            input = new_inputs;
        }
        // Ausgabe der Neuronen der letzten Schicht ist die Vorhersage des Netzes
        return input;
    }



    private double activationDerivative(double output){
        if(activationFunction.equalsIgnoreCase("Sigmoid")) {
            return output * (1.0 - output);
        }
        else if(activationFunction.equalsIgnoreCase("tanh")) {
            return 1.0 - Math.pow(Math.tanh(output), 2);
        }
        else if(activationFunction.equalsIgnoreCase("ReLu")) {
            if(output <=0) return 0.0;
            if(output > 0) return 1.0;
        }
        // Sigmoid als Default
        return output * (1.0 - output);
    }

    private void backpropagate(double[] expected){
        double error;
        // RÜckwärts über Netzschichten iterieren
        for(int i=network.size()-1; i >=0; i--){
            ArrayList<HashMap<String,double[]>> layer = network.get(i);
            List<Double> errors= new ArrayList<>();

            // Fehler in versteckten Schicht berechnen
            if(i != network.size()-1){
                for(int j=0; j < layer.size(); j++){
                     error = 0.0;
                    for(HashMap<String,double[]> neuron : network.get(i+1)){
                        //Fehler in verstecktem Neuron j = (Summe gewichteter Fehler von Neuron j zu Neuronen der nächsten Schicht) * Ableitung
                        error += (neuron.get("weights")[j] * neuron.get("delta")[0]);
                    }
                    errors.add(error);
                }
                // Ermitteln des Fehlers der Ausgabeschicht: (Ausgabe - Zielwert) * Ausgabe
            } else {
                for(int j=0; j < layer.size(); j++){
                    HashMap<String,double[]> neuron = layer.get(j);
                    errors.add(neuron.get("output")[0] - expected[j]);
                }
            }

            // Fehler für jedes Neuron zu Ende berechnen, also mit Ableitung von der Aktivierungsfunktion multiplizieren und eintragen
                for (int j = 0; j < layer.size(); j++) {
                    HashMap<String, double[]> neuron = layer.get(j);
                    neuron.put("delta", new double[]{errors.get(j) * activationDerivative(neuron.get("output")[0])});
            }
        }
    }



    public void updateWeights(double[] inputFromData) {
        double[] inputs;
        for (int i = 0; i < network.size(); i++) {
            // Input für Eingabeschicht ist Eingabe vom Datensatz
            inputs = inputFromData;
            if (i != 0) {
                inputs = new double[network.get(i - 1).size()];
                int idx = 0;

                // Inputwerte aus vorhergehenden Schicht ermitteln
                for (HashMap<String, double[]> neuron : network.get(i - 1)) {
                    inputs[idx] = neuron.get("output")[0];
                    idx++;
                }
            }
            // Über Neuronen iterieren und Gewichte und Bias anhand der berechneten Gradienten aktualisieren
            for (HashMap<String, double[]> neuron : network.get(i)) {
                for (int j = 0; j < inputs.length; j++) {
                    neuron.get("weights")[j] -= learningRate * neuron.get("delta")[0] * inputs[j];
                }
                // Aktualiere die Bias-Werte
                neuron.get("weights")[neuron.get("weights").length - 1] -= learningRate * neuron.get("delta")[0];

            }
        }
    }

// Update Strategie: Stochastischer Gradientenabstieg
    public void trainNetwork(int epochs, List<MNISTDecoder.Fashion> trainingDataSet,int trainingDataSize){
        for(int i=0; i <= epochs; i++){
            Collections.shuffle(trainingDataSet);
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
                    ", Error: " + sumError/(2*trainingDataSize));
        }
    }

    public void trainNetworkWithAnnealing(int epochs, List<MNISTDecoder.Fashion> trainingDataSet,int trainingDataSize) {
        learningRate= 0.001;
        for (int i = 0; i <= epochs; i++) {
            //Sigmod Decay Annealing
            learningRate = 0.01 + (0.1 - 0.01) * (1.0 / (1.0 + Math.exp((1.0 / 2.0) * (2.0 * (i / 100.0) - 1.0))));
            Collections.shuffle(trainingDataSet);
            double sumError = 0;
            for (int input = 0; input < trainingDataSize; input++) {
                double[] image = trainingDataSet.get(input).image;
                double[] prediction = forwardPropagation(image);
                double[] expected = trainingDataSet.get(input).label;
                sumError += computeError(prediction, expected);
                backpropagate(expected);
                updateWeights(image);
            }
            System.out.println("Epoch: " + i + ", lRate: " + learningRate +
                    ", Error: " + sumError / (2 * trainingDataSize));
        }
    }

// Fehler anhand der quadratischen Kostenfunktion
    private double computeError(double[] prediction, double[] expected){
        double error=0;
        for(int i=0; i < prediction.length; i++){
            error += Math.pow(expected[i] - prediction[i],2);
        }
        return error;
    }


}
