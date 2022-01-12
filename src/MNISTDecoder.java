import java.io.IOException;
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

        List<Fashion> fashions = new ArrayList<>();

        int readHeadImages = 16;
        int readHeadLabel = 8;
        while(readHeadImages < imageByte.length) {

            byte[][] image = new byte[28][28];

            for(int i = 0; i < 28; i++) {

                for(int k = 0; k < 28; k++) {
                    image[i][k] = imageByte[readHeadImages++];
                }

            }
            int label = toUnsignedByte(labelByte[readHeadLabel++]);

            fashions.add(new Fashion(label, image));
        }

        return fashions;
    }

    public static int toUnsignedByte(byte b) {
        return b & 0xFF;
    }

    public static class Fashion {

        public int label;
        public byte[][] image;

        public Fashion(int label, byte[][] image) {
            this.label = label;
            this.image = image;
        }

        @Override
        public String toString() {
            return (String.format("Label: %s Image: %s \n", this.label, Arrays.deepToString(this.image)));
        }

    }

}
