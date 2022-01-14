import java.util.Arrays;

public class NeuralNetwork {
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private String activationFunction;
    private int epochs;
    private double learningRate;

    private double[] input;

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

    // Speichert Gradienten der Neuronen in der versteckten Schicht
    double[][] hiddenGradients;

    //speichert Gradienten der Neuronen in der Ausgabeschicht
    double[][] outputGradients;

    //Eingabe Z der versteckten Schicht
    double[][] inputFinalLayer;

    //Eingabe Z der versteckten Schicht
    double[][] inputHiddenLayer;

    double[][] outputHiddenLayer;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, int epochs, double learningRate, double[] input) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.input = input;

        weightsInputToHidden = new double[hiddenNeurons][inputNeurons];
        weightsHiddenToOutput = new double[outputNeurons][hiddenNeurons];

        initializeMatrix(weightsInputToHidden);
        initializeMatrix(weightsHiddenToOutput);


        biasHiddenLayer = new double[hiddenNeurons][1];
        biasOutputLayer = new double[outputNeurons][1];

        activationFunction = "ReLu";

        hiddenGradients = new double[hiddenNeurons][1];
        outputGradients = new double[outputNeurons][1];

        inputFinalLayer = new double[hiddenNeurons][1];
        inputHiddenLayer = new double[hiddenNeurons][1];


    }

    /**
     * Überladener Konstruktor, um Gewichte explizit zu setzen und nicht zufällig erstellen.
     * Dieser Konstruktor gibt es nur, um Tests für diese Klasse zu schreiben
     *
     * @param inputNeurons
     * @param hiddenNeurons
     * @param outputNeurons
     * @param weightsInputToHidden
     * @param weightsHiddenToOutput
     * @param biasHiddenLayer
     * @param biasOutputLayer
     * @param epochs
     */
    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double[][] weightsInputToHidden, double[][] weightsHiddenToOutput, double[][] biasHiddenLayer, double[][] biasOutputLayer, int epochs, double learningRate, double[] input) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.activationFunction = "ReLu";
        this.weightsInputToHidden = weightsInputToHidden;
        this.weightsHiddenToOutput = weightsHiddenToOutput;
        this.biasHiddenLayer = biasHiddenLayer;
        this.biasOutputLayer = biasOutputLayer;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.input = input;
        this.outputGradients = new double[outputNeurons][1];
        this.hiddenGradients = new double[hiddenNeurons][1];

    }

    /**
     * Diese Methode initialisiert eine Matrix mit zufälligen Werten zwischen -1 und 1
     * Diese Methode wird benötigt, um am Anfang zufällige Gewichte zu verteilen
     *
     * @param matrix Die Gewichtematrix, die mit zufälligen Werten initialisiert werden soll
     */
    private void initializeMatrix(double[][] matrix) {
        double range = 1.0 - (-1) + 1;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = (Math.random() * range) - 1.0;
            }
        }
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Diese Methode wendet eine durch Benutzereingabe gewählte Aktivierungsfunktion auf den Eingabewert
     * der Schicht an. Als Default wird die ReLu-Funktion angewendet
     *
     * @param z Eingabewert der Schicht
     * @return Ausgabewert der Schicht
     */
    public double[][] activationFunction(double[][] z) {
        if (activationFunction.equalsIgnoreCase("ReLu")) {
            for (int i = 0; i < z.length; i++) {
                z[i][0] = Math.max(0, z[i][0]);
            }
            return z;

        } else if (activationFunction.equalsIgnoreCase("tanh")) {
            for (int i = 0; i < z.length; i++) {
                z[i][0] = Math.tanh(z[i][0]);

            }
            return z;

        } else if (activationFunction.equalsIgnoreCase("sigmoid")) {
            for (int i = 0; i < z.length; i++) {
                z[i][0] = sigmoid(z[i][0]);
            }
            return z;

        } else {
            for (int i = 0; i < z.length; i++) {
                z[i][0] = Math.max(0, z[i][0]);
            }
            return z;
        }
    }

    /**
     * Hilfsmethode, die Sigmoidfunktion für einen gegebenen Wert berechnet
     *
     * @param z
     * @return
     */
    public double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * Diese Methode implementiert die Produktion der Ausgabe des neuronalen Netzes
     *
     * @return eindimensionales byte-Array. Array hat die gleiche Länge wie es Output-Neuronen gibt
     * Der Index repräsentiert der Ausgabewert des Ausgabeneurons an der entsprechenden Stelle
     * Jedes Ausgabeneuron repräsentiert eine Klasse. Die Eingabe wird der Klasse zugeordnet,
     * deren Neuron den höchsten Wert aufweist
     */
    public double[] makePrediction() {
        // Eingabe als Matrix aufbereiten
        double[][] inputAsMatrix = Matrix.transposeMatrix(new double[][]{input});

        // Eingabevektor mit weightsInputToHidden-Matrix multiplizieren
        double[][] hiddenInput = Matrix.multiplyMatrices(weightsInputToHidden, inputAsMatrix);

        //  biasHiddenLayer zu Ergebnis der Multiplikation addieren
        inputHiddenLayer = Matrix.addMatrices(hiddenInput, biasHiddenLayer);

        // Aktivierungsfunktion anwenden
        outputHiddenLayer = activationFunction(inputHiddenLayer);

        // Eingabematrix der letzten Schicht berechnen: hiddenOutput Mal weightsHiddenToOutput
        inputFinalLayer = Matrix.multiplyMatrices(weightsHiddenToOutput, outputHiddenLayer);

        //  biasOutputLayer zu Ergebnis der Multiplikation addieren
        inputFinalLayer = Matrix.addMatrices(inputFinalLayer, biasOutputLayer);

        // Aktivierungsfunktion anwenden
        double[][] finalOutput = activationFunction(inputFinalLayer);

        prediction = Arrays.stream(finalOutput)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        return prediction;
    }


    /**
     * Diese Methode berechnet die Kostenfunktion. Diese berechnet den Fehler in der Vorhersage(Methode makePrediction) des Netzes
     * Die Kosten werden anhand der quadratischen Kostenfunktion berechnet
     *
     * @return
     */
    //TODO Methode für mehrere Eingabevektoren implementieren
    public double costFunction(double[] expectedResults) {
        if (prediction.length != expectedResults.length) {
            throw new IllegalArgumentException("Prediction and Expectation don't match in size!");
        }
        double[] costVector = Matrix.subtract(expectedResults, prediction);

        double cost = 1.0 / 2.0 * computeVectorLength(costVector);
        return cost;
    }

    /**
     * Das ist eine Hilfsmethode, um die Kosten zu berechnen. Diese Methode berechnet die Länge eines Vektors
     *
     * @param vector Vektor, dessen Länge ermittelt werden soll
     * @return Länge des Vektors
     */
    public double computeVectorLength(double[] vector) {
        double length = 0.0;
        for (int i = 0; i < vector.length; i++) {
            length += Math.pow(vector[i], 2);
        }
        length = Math.sqrt(length);
        return length;
    }


    /**
     * Diese Methode berechnet den Gradienten für jedes Neuron mit dem Backprogropagation-Algorithmus. Die Gradienten für jede Schicht
     * werden in dem entsprechenden Array gespeichert: hiddenGradients/outputGradients
     *
     * @param targetOutputs Die Zielwerte, die vom Netz erlernt werden sollen
     */
    public void backpropagateError(double[] targetOutputs) {
        /**
         * 1. Schritt: Fehler(Gradient) in der Ausgabe-Ebene ermitteln:
         */
        computeOutputError(targetOutputs);


        /**
         * 2. Schritt: Fehler(Gradient) in der verdeckten Schicht berechnen
         * Matrix der Gewichte Hidden To Output transponieren,
         * Matrix mit Gradientenmatrix der Ausgabeschicht multiplizieren
         * Matrix mit Ableitung der Aktivierungsfunktion multiplizieren
         */

        double[][] resultMatrix = Matrix.transposeMatrix(weightsHiddenToOutput);
        resultMatrix = Matrix.multiplyMatrices(resultMatrix, outputGradients);
        for (int i = 0; i < resultMatrix.length; i++) {
            hiddenGradients[i][0] = resultMatrix[i][0] * derivativeActivationFunction(inputHiddenLayer[i][0]);
        }


    }

    /**
     * Diese Methode berechnet den Fehler(Gradient)in der Ausgabeschicht anhand der quadratischen Kostenfunktion
     * Gradient = Ableitung der Kostenfunktion nach dem Ausgabewert des Neurons * Ableitung der Aktivierungsfunktion
     * Gradienten werden im Gradientenarray der Ausgabeschicht gespeichert
     */
    public void computeOutputError(double[] targetOutputs) {

        // Ableitung der quadratischen Kostenfunktion: Vorhersage - Zielwert
        for (int i = 0; i < prediction.length; i++) {
            outputGradients[i][0] = prediction[i] - targetOutputs[i];
        }
        for (int i = 0; i < outputGradients.length; i++) {
            outputGradients[i][0] = outputGradients[i][0] * derivativeActivationFunction(inputFinalLayer[i][0]);
        }


    }

    ;

    /**
     * Das ist eine Hilfsmethode, um die Ableitung der aktuellen Aktivierungsfunktion zu berechnen
     *
     * @param z
     * @return
     */
    public double derivativeActivationFunction(double z) {
        if (activationFunction.equalsIgnoreCase("ReLu")) {
            if (z <= 0) return 0;
            else return 1;
        } else if (activationFunction.equalsIgnoreCase("tanh")) {
            return 1 - (Math.pow(Math.tanh(z), 2));
        } else if (activationFunction.equalsIgnoreCase("sigmoid")) {
            return sigmoid(z) * (1 - sigmoid(z));
        } else {
            if (z <= 0) return 0;
            else return 1;
        }
    }

    /**
     * Diese Methode aktualisiert die Gewichte und die Bias anhand des Gradienten
     */
    private void updateWeightsAndBiases() {


        updateWeights();
        updateBiases();
    }

    /**
     * Diese Methode aktualisiert die Gewichte und Biaswerte anhand der berechneten Gradienten
     * neues Gewicht von Neuron i zu Neuron j = altes Gewicht  - eta * (Eingabewert von Neuron i zu Neuron j - Gradient des Neurons)
     * neuer Biaswert = alter Biaswert - eta * Gradient des Neurons
     */
    public void updateWeights() {
        for (int i = 0; i < weightsInputToHidden.length; i++) {
            for (int j = 0; j < weightsInputToHidden[0].length; j++) {
                weightsInputToHidden[i][j] = weightsInputToHidden[i][j] - (learningRate * hiddenGradients[i][0] * input[j]);
            }

        }

        for (int i = 0; i < weightsHiddenToOutput.length; i++) {
            for (int j = 0; j < weightsHiddenToOutput[0].length; j++) {
                weightsHiddenToOutput[i][j] = weightsHiddenToOutput[i][j] - (learningRate * outputGradients[i][0] * outputHiddenLayer[i][0]);
                ;
            }

        }
    }

    public void updateBiases() {
        for (int i = 0; i < biasHiddenLayer.length; i++) {
            biasHiddenLayer[i][0] = biasHiddenLayer[i][0] - (learningRate * hiddenGradients[i][0]);
        }
        for (int i = 0; i < biasOutputLayer.length; i++) {
            biasOutputLayer[i][0] = biasOutputLayer[i][0] - (learningRate * outputGradients[i][0]);
        }
    }


    public static void main(String[] args) {


        double[] input = new double[]{1.0, -3.456, 0.54};
        double[] expected = new double[]{2.0, 0.1, -1};


        NeuralNetwork nn = new NeuralNetwork(3, 4, 3, 1, 0.5, input);

        System.out.println("Creating Neural Network: " + nn.inputNeurons + "-Input Neurons, " +
                nn.hiddenNeurons + "-Hidden Neurons, " +
                nn.outputNeurons + "-Output Neurons");

        System.out.println("Using " + nn.activationFunction + " function" + " for activation\n");
        System.out.println("Initial Weights: \n Input To Hidden: " + Arrays.deepToString(nn.weightsInputToHidden) + "\n" +
                "Hidden To Output:\n" + Arrays.deepToString(nn.weightsHiddenToOutput) + "\n");
        System.out.println("The input is:" + Arrays.toString(input) + "\n");
        for (int i = 0; i < 20; i++) {
            System.out.println("The Output is: " + Arrays.toString(nn.makePrediction()) + "\n");
            System.out.println("The Output to learn is: " + Arrays.toString(expected) + "\n");
            System.out.println("The error is: " + nn.costFunction(expected));
            System.out.println("------------------------------------------------------------------------------------");
            nn.backpropagateError(expected);
            nn.updateWeightsAndBiases();
        }
    }
}
