import com.google.common.primitives.Doubles;
import edu.stanford.math.plex4.api.Plex4;
import edu.stanford.math.plex4.examples.PointCloudExamples;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import edu.stanford.math.plex4.homology.chain_basis.Simplex;
import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.metric.impl.EuclideanMetricSpace;
import edu.stanford.math.plex4.streams.impl.ExplicitSimplexStream;
import edu.stanford.math.plex4.streams.impl.VietorisRipsStream;
import mmaioe.com.featureextraction.topology.PersistentHomology;

import java.util.ArrayList;
import java.util.List;


public class BasicHomology {

    public static void main(String[] args) {

//        System.out.println(Math.sqrt(8));
//        EuclideanMetricSpace test = new EuclideanMetricSpace(PointCloudExamples.getHouseExample());

        List<List<Double>> pointCloud = new ArrayList<List<Double>>();
        double[][] points = PointCloudExamples.getHouseExample();

        for(int i=0;i<points.length;i++){
            pointCloud.add(Doubles.asList(new double[]{points[i][0], points[i][1]}));
        }

        PersistentHomology homology = new PersistentHomology(pointCloud);

        int[] numbers = homology.getBettiNumbers(2);

        System.out.println(" size of numbers : "+numbers.length);
        System.out.println(numbers);
//
//        ExplicitSimplexStream stream = new ExplicitSimplexStream();
//
//        stream.addVertex(0);
//        stream.addVertex(1);
//        stream.addVertex(2);
//
//        stream.addElement(new int[] {0, 1});
//        stream.addElement(new int[]{0, 2});
//
//        stream.finalizeStream();
//
//        System.out.println("Size of complex: " + stream.getSize());
//
//
//        AbstractPersistenceAlgorithm<Simplex> persistence
//                = Plex4.getModularSimplicialAlgorithm(3, 2);
//
//        BarcodeCollection<Double> circle_intervals
//                = persistence.computeIntervals(stream);
//
//        System.out.println(circle_intervals);

    }

}