import java.util.Arrays;

public class Matrix {


    private static double multiplyMatricesCell(double[][] firstMatrix,
                                               double[][] secondMatrix, int row, int col) {
        double cell = 0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
            if(Double.isInfinite(cell)) {
                if(cell < Double.MAX_VALUE) cell = Double.MIN_VALUE;
                else cell = Double.MAX_VALUE;
            }
        }
        return cell;
    }
    public static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix[0].length != secondMatrix.length) {
            throw new IllegalArgumentException("Matrix Sizes not fit for multiplication");
        }
        double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
            }
        }

        return result;
    }



    public static double[][] transposeMatrix(double[][] matrix) {
        double[][] transpose = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transpose[j][i] = matrix[i][j];
            }
        }
        return transpose;
    }

    public static double[][] addMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix.length != secondMatrix.length || firstMatrix[0].length != secondMatrix[0].length) {
            throw new IllegalArgumentException("Matrix addition only possible if number of rows and columns are equal!");
        }
        int rows = firstMatrix.length;
        int columns = firstMatrix[0].length;
        double[][] sum = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sum[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
            }
        }
        return sum;
    }
    public static double[][] addMatrices(double[][] firstMatrix, double[][] secondMatrix, String aufrufer) {
        if (firstMatrix.length != secondMatrix.length || firstMatrix[0].length != secondMatrix[0].length) {
            throw new IllegalArgumentException("Matrix addition only possible if number of rows and columns are equal!");
        }
        int rows = firstMatrix.length;
        int columns = firstMatrix[0].length;
        double[][] sum = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sum[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
                if(Double.isInfinite(sum[i][j])) System.out.println("Infinite at AddMAtrices: Aufrufer: " + aufrufer +   "-" + firstMatrix[i][j] +","+secondMatrix[i][j]);
                if(Double.isNaN(sum[i][j])) System.out.println("NaN at AddMAtrices: Params - " + firstMatrix[i][j] +","+secondMatrix[i][j]);
            }
        }
        return sum;
    }

    public static double[] subtract(double[] A, double[] B) {
        if (A.length != B.length) {
            throw new IllegalArgumentException("Matrices must have equal size to be subtracted");
        }
        int i;
        int size = A.length;
        double[] C = new double[size];

            for (i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }


        return C;
    }
}
