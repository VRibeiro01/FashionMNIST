import java.util.Arrays;

public class NeuralNetwork {
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private String activationFunction;
    private int epochs;

    // Gewichtematrix von Eingabeschicht zur verborgenen Schicht
    double[][] weightsInputToHidden;

    // Gewichtematrix von verborgenen Schicht zur Ausgabeschicht
    double[][] weightsHiddenToOutput;

    // Biasmatrix der verborgenen Schicht
    double[][] biasHiddenLayer;

    // Biasmatrix der Ausgabeschicht
    double[][] biasOutputLayer;

    // Vorhersage des Netzes
    double[] prediction;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, int epochs){
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.epochs = epochs;

        weightsInputToHidden = new double[hiddenNeurons][inputNeurons];
        weightsHiddenToOutput = new double[outputNeurons][hiddenNeurons];

        initializeMatrix(weightsInputToHidden);
        initializeMatrix(weightsHiddenToOutput);


        biasHiddenLayer = new double[hiddenNeurons][1];
        biasOutputLayer = new double[outputNeurons][1];

        activationFunction = "ReLu";



    }

    /**
     * Überladener Konstruktor, um Gewichte explizit zu setzen und nicht zufällig erstellen.
     * Dieser Konstruktor gibt es nur, um Tests für diese Klasse zu schreiben
     * @param inputNeurons
     * @param hiddenNeurons
     * @param outputNeurons
     * @param weightsInputToHidden
     * @param weightsHiddenToOutput
     * @param biasHiddenLayer
     * @param biasOutputLayer
     * @param epochs
     */
    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double[][] weightsInputToHidden, double[][] weightsHiddenToOutput, double[][] biasHiddenLayer, double[][] biasOutputLayer, int epochs) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.activationFunction = "ReLu";
        this.weightsInputToHidden = weightsInputToHidden;
        this.weightsHiddenToOutput = weightsHiddenToOutput;
        this.biasHiddenLayer = biasHiddenLayer;
        this.biasOutputLayer = biasOutputLayer;
        this.epochs=epochs;

    }

    /**
     * Diese Methode initialisiert eine Matrix mit zufälligen Werten zwischen -1 und 1
     * Diese Methode wird benötigt, um am Anfang zufällige Gewichte zu verteilen
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

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Diese Methode wendet eine durch Benutzereingabe gewählte Aktivierungsfunktion auf den Eingabewert
     * der Schicht an. Als Default wird die ReLu-Funktion angewendet
     * @param z Eingabewert der Schicht
     * @return Ausgabewert der Schicht
     */
    public double[][] activationFunction(double[][] z){
        if(activationFunction.equalsIgnoreCase("ReLu")) {
            for(int i=0; i < z.length; i++){
                for(int j=0; j<z[0].length; j++)
                z[i][j] = Math.max(0,z[i][j]);
            }
            return z;
        }
        else  if(activationFunction.equalsIgnoreCase("tanh")) {
            for(int i=0; i < z.length; i++){
                for(int j=0; j<z[0].length; j++){
                    z[i][j] = Math.tanh(z[i][j]);
            }
            }

            return z;
        }
        else if(activationFunction.equalsIgnoreCase("sigmoid")) {
            for(int i=0; i < z.length; i++){
                for(int j=0; j<z[0].length; j++)
                    z[i][j] = 1/ (1 + Math.exp(-z[i][j]));
            }
            return z;
        }
        else{
            for(int i=0; i < z.length; i++){
                for(int j=0; j<z[0].length; j++)
                    z[i][j] = Math.max(0,z[i][j]);
            }
            return z;
        }
    }

    public double[] makePrediction(double[] input){
        // Eingabe als Matrix aufbereiten
        double[][] inputAsMatrix = Matrix.transposeMatrix(new double[][]{input});

        // Eingabevektor mit weightsInputToHidden-Matrix multiplizieren
         double[][] hiddenInput = Matrix.multiplyMatrices(weightsInputToHidden,inputAsMatrix);

        //  biasHiddenLayer zu Ergebnis der Multiplikation addieren
        hiddenInput = Matrix.addMatrices(hiddenInput,biasHiddenLayer);

        // Aktivierungsfunktion anwenden
        double[][] hiddenOutput = activationFunction(hiddenInput);

        // Eingabematrix der letzten Schicht berechnen: hiddenOutput Mal weightsHiddenToOutput
        double[][] finalInput = Matrix.multiplyMatrices(weightsHiddenToOutput,hiddenOutput);

        //  biasOutputLayer zu Ergebnis der Multiplikation addieren
        finalInput = Matrix.addMatrices(finalInput,biasOutputLayer);

        // Aktivierungsfunktion anwenden
        double[][] finalOutput = activationFunction(finalInput);

        prediction = Arrays.stream(finalOutput)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        return prediction;
    }

    /**
     * Diese Methode berechnet den Fehler anhand der quadratischen Kostenfunktion
     * @return Fehlerfaktor Delta
     */
    private double computeOutputError(){return 1.0;};

    /**
     * Diese Methode berechnet die Kostenfunktion. Diese berechnet den Fehler in der Vorhersage(Methode makePrediction) des Netzes
     * Die Kosten werden anhand der quadratischen Kostenfunktion berechnet
     * @return
     */
    //TODO Methode für mehrere Eingabevektoren implementieren
    public double costFunction(double[] expectedResults){
        if(prediction.length != expectedResults.length){
            throw new IllegalArgumentException("Prediction and Expectation don't match in size!");
        }
        double[] costVector = Matrix.subtract(expectedResults,prediction);

        double cost = 1.0/2 * Math.pow(computeVectorLength(costVector),2);
        return cost;
    }

    /**
     * Das ist eine Hilfsmethode, um die Kosten zu berechnen. Diese Methode berechnet die Länge eines Vektors
     * @param vector Vektor, dessen Länge ermittelt werden soll
     * @return Länge des Vektors
     */
        public double computeVectorLength(double[] vector){
         double length = 0.0;
         for(int i=0;i<vector.length;i++){
             length += Math.pow(vector[i],2);
         }
         length = Math.sqrt(length);
         return length;
        }


    private void backpropagateError(){}

    /**
     * Diese Methode aktualisiert die Gewichte und die Bias anhand des Gradienten
     */
    private void updateWeightsAndBiases(){}





    public static void main(String[] args){



        double[] input = new double[]{1.0, -3.456, 0.54};
        double[] expected = new double[]{2.0, 0.1, -1};


        NeuralNetwork nn = new NeuralNetwork(3,4,3,1);

        System.out.println("Creating Neural Network: " + nn.inputNeurons + "-Input Neurons, " +
                nn.hiddenNeurons + "-Hidden Neurons, " +
                nn.outputNeurons+ "-Output Neurons");

        System.out.println("Using " + nn.activationFunction + " function" + " for activation\n");
        System.out.println("Initial Weights: \n Input To Hidden: " + Arrays.deepToString(nn.weightsInputToHidden) + "\n" +
                "Hidden To Output:\n" + Arrays.deepToString(nn.weightsHiddenToOutput)+"\n");

        System.out.println("The input is:" + Arrays.toString(input)+ "\n");
        System.out.println("The Output is: "+ Arrays.toString(nn.makePrediction(input)) + "\n");
        System.out.println("The Output to learn is: "+ Arrays.toString(expected) + "\n");

        System.out.println("The error is: " + nn.costFunction(expected));

  }
}
