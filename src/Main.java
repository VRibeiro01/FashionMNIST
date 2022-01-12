import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        List<MNISTDecoder.Fashion> fashionSet = MNISTDecoder.loadDataSet(
                System.getProperty("user.dir") + "/resources/train-images-idx3-ubyte",
                System.getProperty("user.dir") + "/resources/train-labels-idx1-ubyte"
        );
        System.out.println(fashionSet);

    }

}
