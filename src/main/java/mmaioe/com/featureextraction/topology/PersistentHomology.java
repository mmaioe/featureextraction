package mmaioe.com.featureextraction.topology;

import edu.stanford.math.plex4.api.Plex4;
import edu.stanford.math.plex4.examples.PointCloudExamples;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import edu.stanford.math.plex4.homology.chain_basis.Simplex;
import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.streams.impl.VietorisRipsStream;

import java.util.List;

/**
 * Created by ito_m on 7/4/16.
 */
public class PersistentHomology {
    List<List<Double>> pointCloud;
    public PersistentHomology(List<List<Double>> pointCloud){
        this.pointCloud = pointCloud;
    }

    /**
     * Informally, there are different kinds of interpretation for Betti Number:
     * 1. the number of holes in a two dimensional point cloud, e.g. English Letter
     *
     * @param dimension
     * @return
     */
    public int[] getBettiNumbers(int dimension){
        double[][] pointCloudArray = new double[pointCloud.size()][pointCloud.get(0).size()];
        for(int i=0;i<pointCloudArray.length;i++){
            pointCloudArray[i] = new double[]{
                pointCloud.get(i).get(0),
                pointCloud.get(i).get(1)
            };
        }

        VietorisRipsStream stream = Plex4.createVietorisRipsStream(pointCloudArray, 2, 8, 10);
        AbstractPersistenceAlgorithm<Simplex> persistence =Plex4.getModularSimplicialAlgorithm(dimension,2);

        //
        BarcodeCollection<Double> circle_intervals
                = persistence.computeIntervals(stream);

        return circle_intervals.getBettiSequence();
    }
}
