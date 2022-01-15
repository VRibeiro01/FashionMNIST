import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MNISTDecoder {

    public static List<Fashion> loadDataSet(String pathOfImages, String pathOfLabels) throws IOException {
        Path imagePath = Paths.get(pathOfImages);
        Path labelPath = Paths.get(pathOfLabels);

        byte[] imageByte = Files.readAllBytes(imagePath);
        byte[] labelByte = Files.readAllBytes(labelPath);

        ArrayList<Fashion> fashions = new ArrayList<>();

        int readHeadImages = 16;
        int readHeadLabel = 8;
        while(readHeadImages < imageByte.length) {

            double[] image = new double[784];

            for(int i = 0; i < 784; i++) {
                image[i] = toUnsignedByte(imageByte[readHeadImages++])/255.0;
            }

            int label = toUnsignedByte(labelByte[readHeadLabel++]);
            double[] labelArray = new double[10];
            labelArray[label] = 1.0;

            fashions.add(new Fashion(labelArray, image));
        }

        return fashions;
    }

    public static int toUnsignedByte(byte b) {
        return b & 0xFF;
    }

    public static class Fashion {

        public double[] label;
        public double[] image;

        public Fashion(double[] label, double[] image) {
            this.label = label;
            this.image = image;


        }

        @Override
        public String toString() {
            return (String.format("Label: %s Image: %s \n", Arrays.toString(this.label), Arrays.toString(this.image)));
        }

    }

}
