package project.project2;

public class MatrixMultiplication {
    static int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = (int) (Math.random() * 10);
            }
        }
        return matrix;
    }

    static void multiplyMatrix1(int[][] matrix1, int[][] matrix2, int[][] result) {
        int n = matrix1.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    static void multiplyMatrix2(int[][] matrix1, int[][] matrix2, int[][] result) {
        int n = matrix1.length;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    public static void main(String[] args) {
        int n = 2048;
        int[][] matrix1 = generateMatrix(n);
        int[][] matrix2 = generateMatrix(n);
        int[][] result = new int[n][n];
        long startTime = System.nanoTime();
        multiplyMatrix1(matrix1, matrix2, result);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        System.out.println("Time: " + duration / 1000000 + " ms");
        startTime = System.nanoTime();
        multiplyMatrix2(matrix1, matrix2, result);
        endTime = System.nanoTime();
        duration = (endTime - startTime);
        System.out.println("Time: " + duration / 1000000 + " ms");

    }
}