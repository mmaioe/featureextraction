package mmaioe.com.featureextraction

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Created by ito_m on 6/4/16.
 */
trait FeatureExtraction {

  def encode(rowdata:INDArray): INDArray
  def encodeAndDecode(rowdata:INDArray): INDArray
}
