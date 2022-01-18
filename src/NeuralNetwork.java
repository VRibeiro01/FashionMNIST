import java.util.Arrays;


public class NeuralNetwork {
    public final int inputNeurons;
    public final int hiddenNeurons;
    public final int outputNeurons;
    private final double learningRate;
    private double sizeTrainingsData;
    public String activationFunction;
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

    // Ausgabe der versteckten Schicht
    double[][] outputHiddenLayer;

    public double[][] hiddenLayerBatchCorrectionValues;

    public double[][] outLayerBatchCorrectionValues;




    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, String activationFunction, double learningRate, double sizeTrainingsData) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.learningRate = learningRate;
        this.sizeTrainingsData = sizeTrainingsData;


        weightsInputToHidden = new double[hiddenNeurons][inputNeurons];
        weightsHiddenToOutput = new double[outputNeurons][hiddenNeurons];

        initializeMatrix(weightsInputToHidden);
        initializeMatrix(weightsHiddenToOutput);


        biasHiddenLayer = new double[hiddenNeurons][1];
        biasOutputLayer = new double[outputNeurons][1];

        this.activationFunction = activationFunction;

        hiddenGradients = new double[hiddenNeurons][1];
        outputGradients = new double[outputNeurons][1];

        inputFinalLayer = new double[hiddenNeurons][1];
        inputHiddenLayer = new double[hiddenNeurons][1];

        this.hiddenLayerBatchCorrectionValues = new double[hiddenNeurons][inputNeurons+1];
        this.outLayerBatchCorrectionValues = new double[outputNeurons][hiddenNeurons+1];



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
     */
    @SuppressWarnings("JavaDoc")
    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double[][] weightsInputToHidden, double[][] weightsHiddenToOutput, double[][] biasHiddenLayer, double[][] biasOutputLayer, double learningRate) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.activationFunction = "ReLu";
        this.weightsInputToHidden = weightsInputToHidden;
        this.weightsHiddenToOutput = weightsHiddenToOutput;
        this.biasHiddenLayer = biasHiddenLayer;
        this.biasOutputLayer = biasOutputLayer;
        this.learningRate = learningRate;
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
        double range = 0.5-(-0.5);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = -0.5+(Math.random() * range);
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
                if( Double.isNaN(z[i][0]))System.out.println("NaN at tanh: Index - " + i );
                if( Double.isInfinite(z[i][0]))System.out.println("Infinite at tanh: Index - " + i );
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
    @SuppressWarnings("JavaDoc")
    public double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * Diese Methode implementiert die Produktion der Ausgabe des neuronalen Netzes
     *
     * @return eindimensionales double-Array. Array hat die gleiche Länge wie es Output-Neuronen gibt
     * Der Index repräsentiert der Ausgabewert des Ausgabeneurons an der entsprechenden Stelle
     * Jedes Ausgabeneuron repräsentiert eine Klasse. Die Eingabe wird der Klasse zugeordnet,
     * deren Neuron den höchsten Wert aufweist
     */
    public double[] makePrediction(double[] input) {
        // Eingabe als Matrix aufbereiten
        double[][] inputAsMatrix = Matrix.transposeMatrix(new double[][]{input});


        // Eingabevektor mit weightsInputToHidden-Matrix multiplizieren
        double[][] hiddenInput = Matrix.multiplyMatrices(weightsInputToHidden, inputAsMatrix);


        //  biasHiddenLayer zu Ergebnis der Multiplikation addieren
        inputHiddenLayer = Matrix.addMatrices(hiddenInput, biasHiddenLayer,"inputHiddenLayer");

        // Aktivierungsfunktion anwenden
        outputHiddenLayer = activationFunction(inputHiddenLayer);


        // Eingabematrix der letzten Schicht berechnen: hiddenOutput Mal weightsHiddenToOutput
        inputFinalLayer = Matrix.multiplyMatrices(weightsHiddenToOutput, outputHiddenLayer);

        //  biasOutputLayer zu Ergebnis der Multiplikation addieren
        inputFinalLayer = Matrix.addMatrices(inputFinalLayer, biasOutputLayer,"inputFinalLayer");

        // Aktivierungsfunktion anwenden
        double[][] finalOutput = activationFunction(inputFinalLayer);

        prediction = Arrays.stream(finalOutput)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        return prediction;
    }



    /**
     * Das ist eine Hilfsmethode, um die Kosten zu berechnen. Diese Methode berechnet die Länge eines Vektors
     *
     * @param vector Vektor, dessen Länge ermittelt werden soll
     * @return Länge des Vektors
     */
    public double computeVectorLength(double[] vector) {
        double length = 0.0;
        for (double v : vector) {
            length += Math.pow(v, 2);
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
        /*
         * 1. Schritt: Fehler(Gradient) in der Ausgabe-Ebene ermitteln:
         */
        computeOutputError(targetOutputs);


        /*
         * 2. Schritt: Fehler(Gradient) in der versteckten Schicht berechnen
         * Matrix der Gewichte Hidden To Output transponieren,
         * Matrix mit Gradientenmatrix der Ausgabeschicht multiplizieren
         * Matrix mit Ableitung der Aktivierungsfunktion multiplizieren
         */

        double[][] resultMatrix = Matrix.transposeMatrix(weightsHiddenToOutput);
        resultMatrix = Matrix.multiplyMatrices(resultMatrix, outputGradients);
        for (int i = 0; i < resultMatrix.length; i++) {
            hiddenGradients[i][0] = resultMatrix[i][0] * derivativeActivationFunction(outputHiddenLayer[i][0]);
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
            outputGradients[i][0] =targetOutputs[i] - prediction[i];
            outputGradients[i][0] = outputGradients[i][0] * derivativeActivationFunction(prediction[i]);
        }

    }

    /**
     * Das ist eine Hilfsmethode, um die Ableitung der aktuellen Aktivierungsfunktion zu berechnen
     *
     * @param z
     * @return Ableitung der Aktivierungsfunktion nach z
     */
    @SuppressWarnings("JavaDoc")
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
    public void batchUpdate(double divider) {
        for (int i = 0; i < hiddenLayerBatchCorrectionValues.length; i++) {
            for (int j = 0; j < hiddenLayerBatchCorrectionValues[0].length; j++) {
                hiddenLayerBatchCorrectionValues[i][j] = hiddenLayerBatchCorrectionValues[i][j] / divider;
            }
            hiddenLayerBatchCorrectionValues[i][hiddenLayerBatchCorrectionValues[0].length-1] =hiddenLayerBatchCorrectionValues[i][hiddenLayerBatchCorrectionValues[0].length-1]/divider;

        }
        for (int i = 0; i < outLayerBatchCorrectionValues.length; i++) {
            for (int j = 0; j < outLayerBatchCorrectionValues[0].length; j++) {
                outLayerBatchCorrectionValues[i][j] = outLayerBatchCorrectionValues[i][j] / divider;
            }
            outLayerBatchCorrectionValues[i][outLayerBatchCorrectionValues[0].length-1] = outLayerBatchCorrectionValues[i][outLayerBatchCorrectionValues[0].length-1]/divider;
        }

        updateWeights();
        updateBiases();
    }


    public void sumUpCorrectionValues(double[] input) {
        for (int i = 0; i < weightsInputToHidden.length; i++) {
            for (int j = 0; j < weightsInputToHidden[0].length-1; j++) {
                hiddenLayerBatchCorrectionValues[i][j] += learningRate * input[j] * weightsInputToHidden[i][j] * hiddenGradients[i][0];
            }
            hiddenLayerBatchCorrectionValues[i][hiddenLayerBatchCorrectionValues[0].length-1]+=learningRate * hiddenGradients[i][0];
        }
        for (int i = 0; i < weightsHiddenToOutput.length; i++) {
            for (int j = 0; j < weightsHiddenToOutput[0].length-1; j++) {
                outLayerBatchCorrectionValues[i][j] += learningRate * outputHiddenLayer[j][0] * weightsHiddenToOutput[i][j] * outputGradients[i][0];
            }
            outLayerBatchCorrectionValues[i][outLayerBatchCorrectionValues[0].length-1] += learningRate * outputGradients[i][0];
        }
    }

    public double getHiddenWeightCorrectionValue(int i, int j){
        return hiddenLayerBatchCorrectionValues[i][j];
    }
    public double getOutputWeightCorrectionValue(int i, int j){
        return outLayerBatchCorrectionValues[i][j];
    }
    public double getOutputBiasCorrectionValue(int i){
        return outLayerBatchCorrectionValues[i][outLayerBatchCorrectionValues[0].length-1];
    }
    public double getHiddenBiasCorrectionValue(int i){
        return hiddenLayerBatchCorrectionValues[i][hiddenLayerBatchCorrectionValues[0].length-1];
    }

    /**
     * Diese Methode aktualisiert die Gewichte und Biaswerte anhand der berechneten Gradienten
     * neues Gewicht von Neuron i zu Neuron j = altes Gewicht  - eta * (Eingabewert von Neuron i zu Neuron j - Gradient des Neurons)
     * neuer Biaswert = alter Biaswert - eta * Gradient des Neurons
     */
    public void updateWeights() {
        for (int i = 0; i < weightsInputToHidden.length; i++) {
            for (int j = 0; j < weightsInputToHidden[0].length; j++) {
                weightsInputToHidden[i][j] = weightsInputToHidden[i][j] - getHiddenWeightCorrectionValue(i, j);

            }
        }

        for (int i = 0; i < weightsHiddenToOutput.length; i++) {
            for (int j = 0; j < weightsHiddenToOutput[0].length; j++) {
                weightsHiddenToOutput[i][j] = weightsHiddenToOutput[i][j] - getOutputWeightCorrectionValue(i,j);
            }

        }
    }

    /**
     * Diese Methode aktulisiert die BiasWerte eines jeden Neurons anhand der zuvor berechneten Gradienten
     * neuer Biaswert = alter Biaswert - eta * Gradient
     */
    public void updateBiases() {
        for (int i = 0; i < biasHiddenLayer.length; i++) {
            biasHiddenLayer[i][0] = biasHiddenLayer[i][0] - getHiddenBiasCorrectionValue(i);
        }
        for (int i = 0; i < biasOutputLayer.length; i++) {
            biasOutputLayer[i][0] = biasOutputLayer[i][0] - getOutputBiasCorrectionValue(i);
        }
    }

    public void stochasticUpdate(double[] input){
        for (int i = 0; i < weightsInputToHidden.length; i++) {
            for (int j = 0; j < weightsInputToHidden[0].length; j++) {
                weightsInputToHidden[i][j] = weightsInputToHidden[i][j] - (learningRate * input[j] * weightsInputToHidden[i][j] * hiddenGradients[i][0]);
                if(Double.isInfinite(weightsInputToHidden[i][j])) {
                    weightsInputToHidden[i][j] = Double.MIN_VALUE;
                }


            }
        }

        for (int i = 0; i < weightsHiddenToOutput.length; i++) {
            for (int j = 0; j < weightsHiddenToOutput[0].length; j++) {
                weightsHiddenToOutput[i][j] = weightsHiddenToOutput[i][j] - (learningRate * outputHiddenLayer[j][0] * weightsHiddenToOutput[i][j] * outputGradients[i][0]);
                if(Double.isInfinite(weightsInputToHidden[i][j])) {
                    weightsInputToHidden[i][j] = Double.MIN_VALUE;
                }
            }

        }
        for (int i = 0; i < biasHiddenLayer.length; i++) {
            biasHiddenLayer[i][0] = biasHiddenLayer[i][0] - (learningRate * hiddenGradients[i][0]);
            if(Double.isInfinite(biasHiddenLayer[i][0])) {
                System.out.println("biasH is infinite");
            }
        }
        for (int i = 0; i < biasOutputLayer.length; i++) {
            biasOutputLayer[i][0] = biasOutputLayer[i][0] - (learningRate * outputGradients[i][0]);
        }
    }


}
