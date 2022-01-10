public class NeuralNetwork {
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private String activationFunction;

    // Gewichtematrix von Eingabeschicht zur verborgenen Schicht
    double[][] weightsInputToHidden;

    // Gewichtematrix von verborgenen Schicht zur Ausgabeschicht
    double[][] weightsHiddenToOutput;

    // Biasmatrix der verborgenen Schicht
    double[][] biasHiddenLayer;

    // Biasmatrix der Ausgabeschicht
    double[][] biasOutputLayer;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons){
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;

        weightsInputToHidden = new double[hiddenNeurons][inputNeurons];
        weightsHiddenToOutput = new double[outputNeurons][hiddenNeurons];

        initializeMatrix(weightsInputToHidden);
        initializeMatrix(weightsHiddenToOutput);

        biasHiddenLayer = new double[hiddenNeurons][1];
        biasOutputLayer = new double[outputNeurons][1];

        activationFunction = "ReLu";


    }

    /**
     * Diese Methode initialisiert eine Matrix mit zufälligen Werten zwischen -1 und 1
     * @param matrix Die Gewichtematrix, die mit zufälligen Werten initialisiert werden soll
     */
    private void initializeMatrix(double[][] matrix){
        double range = 1.0 - (-1) + 1;
        for(int i=0; i < matrix.length; i++){
            for(int j=0; j < matrix[i].length; j++){
                matrix[i][j] = (Math.random()*range) - 1.0;
            }
        }
    }

    /**
     * Diese Methode wendet eine durch Benutzereingabe gewählte Aktivierungsfunktion auf den Eingabewert
     * der Schicht an. Als Default wird die ReLu-Funktion angewendet
     * @param z Eingabewert der Schicht
     * @return Ausgabewert der Schicht
     */
    private double activationFunction(double z){
        if(activationFunction.equalsIgnoreCase("ReLu")) {
            return Math.max(0, z);
        }
        else  if(activationFunction.equalsIgnoreCase("tanh")) {
            return Math.tanh(z);
        }
        else if(activationFunction.equalsIgnoreCase("sigmoid")) {
            return 1/ 1 + Math.exp(-z);
        }
        else{
            return Math.max(0, z);
        }
    }

    private double[] makePrediction(double[] input){
        // Eingabevektor mit weightsInputToHidden-Matrix multiplizieren

        //  biasHiddenLayer zu Ergebnis der Multiplikation addieren

        // --> Ergebnis: Eingabewert
        return new double[1];
    }

    /**
     * Diese Methode berechnet den Fehler anhand der quadratischen Kostenfunktion
     * @return Fehlerfaktor Delta
     */
    private double computeOutputError(){return 1.0;};


    private void backpropagateError(){}

    /**
     * Diese Methode aktualisiert die Gewichte und die Bias anhand des Gradienten
     */
    private void updateWeightsAndBiases(){}

    public static void main(String[] args){
  }
}
