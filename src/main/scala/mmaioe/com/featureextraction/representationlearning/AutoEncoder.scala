package mmaioe.com.featureextraction.representationlearning

import mmaioe.com.featureextraction.FeatureExtraction
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer, RBM}
import org.deeplearning4j.nn.conf.{GradientNormalization, Updater, NeuralNetConfiguration, MultiLayerConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Created by ito_m on 6/4/16.
 */
class AutoEncoder(numOfInputs: Int, numOfOutputs: Int, layerNumOfOutputs: List[Int], epoch:Int) extends FeatureExtraction {
  val seed = 123
  val iterations = 2

//  val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(iterations)
//                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
//                .list(6)
//                .layer(0, new RBM.Builder().nIn(numOfInputs).nOut((numOfInputs*1.5).toInt).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(1, new RBM.Builder().nIn((numOfInputs*1.5).toInt).nOut((numOfInputs/2).toInt).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(2, new RBM.Builder().nIn((numOfInputs/2).toInt).nOut(numOfOutputs).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
//                .layer(3, new RBM.Builder().nIn(numOfOutputs).nOut((numOfInputs/2).toInt).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
//                .layer(4, new RBM.Builder().nIn((numOfInputs/2).toInt).nOut((numOfInputs*1.5).toInt).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn((numOfInputs*1.5).toInt).nOut(numOfInputs).build())
//                .pretrain(true).backprop(true)
//                .build()

         var listBuilder =  new NeuralNetConfiguration.Builder()
                          .seed(seed)
                          .iterations(iterations)
                          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        //                  .learningRate(0.06)
                          .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .list(layerNumOfOutputs.size*2+2)


           listBuilder = listBuilder.layer(0, new DenseLayer.Builder().nIn(numOfInputs).nOut(layerNumOfOutputs(0)).activation("relu").weightInit(WeightInit.XAVIER).build())
          (1 until layerNumOfOutputs.size).foreach{
            index =>
              listBuilder = listBuilder.layer(index, new DenseLayer.Builder().nIn(layerNumOfOutputs(index-1)).nOut(layerNumOfOutputs(index)).activation("relu").weightInit(WeightInit.XAVIER).build())
          }
          listBuilder = listBuilder.layer(layerNumOfOutputs.size, new DenseLayer.Builder().nIn(layerNumOfOutputs.last).nOut(numOfOutputs).activation("relu").weightInit(WeightInit.XAVIER).build())
          listBuilder = listBuilder.layer(layerNumOfOutputs.size+1, new DenseLayer.Builder().nIn(numOfOutputs).nOut(layerNumOfOutputs.last).activation("relu").weightInit(WeightInit.XAVIER).build())
          (1 until layerNumOfOutputs.size).foreach{
            index =>
              listBuilder = listBuilder.layer(layerNumOfOutputs.size+index+1, new DenseLayer.Builder().nIn(layerNumOfOutputs(layerNumOfOutputs.size-index)).nOut(layerNumOfOutputs(layerNumOfOutputs.size-index-1)).activation("relu").weightInit(WeightInit.XAVIER).build())
          }
          listBuilder = listBuilder.layer(layerNumOfOutputs.size*2+1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(layerNumOfOutputs(0)).activation("sigmoid").weightInit(WeightInit.XAVIER).nOut(numOfInputs).build())

          var neuralConf = listBuilder
            .pretrain(true).backprop(true)
            .build()

//          val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
//                  .seed(seed)
//                  .iterations(iterations)
//                  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
////                  .learningRate(0.06)
//                  .updater(Updater.NESTEROVS).momentum(0.9)
//                  .regularization(true).l2(1e-4)
//                 .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                 .gradientNormalizationThreshold(1.0)
//                  .list(10)
////                .layer(0, new RBM.Builder().nIn(numOfInputs).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(4, new RBM.Builder().nIn(100).nOut(numOfOutputs).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
////                .layer(5, new RBM.Builder().nIn(numOfOutputs).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
////                .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
////                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numOfInputs).build())
////                   .layer(0, new DenseLayer.Builder().nIn(numOfInputs).nOut(1000).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(1, new DenseLayer.Builder().nIn(1000).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(2, new DenseLayer.Builder().nIn(500).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(3, new DenseLayer.Builder().nIn(250).nOut(100).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(4, new DenseLayer.Builder().nIn(100).nOut(numOfOutputs).activation("relu").weightInit(WeightInit.XAVIER).build()) //encoding stops
////                   .layer(5, new DenseLayer.Builder().nIn(numOfOutputs).nOut(100).activation("relu").weightInit(WeightInit.XAVIER).build()) //decoding starts
////                   .layer(6, new DenseLayer.Builder().nIn(100).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(7, new DenseLayer.Builder().nIn(250).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(8, new DenseLayer.Builder().nIn(500).nOut(1000).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(1000).activation("sigmoid").weightInit(WeightInit.XAVIER).nOut(numOfInputs).build())
//
//                    .layer(0, new DenseLayer.Builder().nIn(numOfInputs).nOut(layerNumOfOutputs(0)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                   .layer(1, new DenseLayer.Builder().nIn(layerNumOfOutputs(0)).nOut(layerNumOfOutputs(1)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                   .layer(2, new DenseLayer.Builder().nIn(layerNumOfOutputs(1)).nOut(layerNumOfOutputs(2)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                   .layer(3, new DenseLayer.Builder().nIn(layerNumOfOutputs(2)).nOut(layerNumOfOutputs(3)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                   .layer(4, new DenseLayer.Builder().nIn(layerNumOfOutputs(3)).nOut(numOfOutputs).activation("relu").weightInit(WeightInit.XAVIER).build()) //encoding stops
//                   .layer(5, new DenseLayer.Builder().nIn(numOfOutputs).nOut(layerNumOfOutputs(3)).activation("relu").weightInit(WeightInit.XAVIER).build()) //decoding starts
//                   .layer(6, new DenseLayer.Builder().nIn(layerNumOfOutputs(3)).nOut(layerNumOfOutputs(2)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                  .layer(7, new DenseLayer.Builder().nIn(layerNumOfOutputs(2)).nOut(layerNumOfOutputs(1)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                  .layer(8, new DenseLayer.Builder().nIn(layerNumOfOutputs(1)).nOut(layerNumOfOutputs(0)).activation("relu").weightInit(WeightInit.XAVIER).build())
//                  .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(layerNumOfOutputs(0)).activation("sigmoid").weightInit(WeightInit.XAVIER).nOut(numOfInputs).build())
//
////                    .layer(0, new DenseLayer.Builder().nIn(numOfInputs).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(1, new DenseLayer.Builder().nIn(500).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(2, new DenseLayer.Builder().nIn(250).nOut(125).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(3, new DenseLayer.Builder().nIn(125).nOut(62).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(4, new DenseLayer.Builder().nIn(62).nOut(numOfOutputs).activation("relu").weightInit(WeightInit.XAVIER).build()) //encoding stops
////                   .layer(5, new DenseLayer.Builder().nIn(numOfOutputs).nOut(62).activation("relu").weightInit(WeightInit.XAVIER).build()) //decoding starts
////                   .layer(6, new DenseLayer.Builder().nIn(58).nOut(125).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(7, new DenseLayer.Builder().nIn(125).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(8, new DenseLayer.Builder().nIn(250).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(500).activation("sigmoid").weightInit(WeightInit.XAVIER).nOut(numOfInputs).build())
//
//
////                    .layer(0, new DenseLayer.Builder().nIn(numOfInputs).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(1, new DenseLayer.Builder().nIn(500).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(2, new DenseLayer.Builder().nIn(250).nOut(170).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(3, new DenseLayer.Builder().nIn(170).nOut(50).activation("relu").weightInit(WeightInit.XAVIER).build())
////                   .layer(4, new DenseLayer.Builder().nIn(50).nOut(numOfOutputs).activation("relu").weightInit(WeightInit.XAVIER).build()) //encoding stops
////                   .layer(5, new DenseLayer.Builder().nIn(numOfOutputs).nOut(50).activation("relu").weightInit(WeightInit.XAVIER).build()) //decoding starts
////                   .layer(6, new DenseLayer.Builder().nIn(50).nOut(170).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(7, new DenseLayer.Builder().nIn(170).nOut(250).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(8, new DenseLayer.Builder().nIn(250).nOut(500).activation("relu").weightInit(WeightInit.XAVIER).build())
////                  .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(500).activation("sigmoid").weightInit(WeightInit.XAVIER).nOut(numOfInputs).build())
//
//            .pretrain(true).backprop(true)
//                  .build()

  val model = new MultiLayerNetwork(neuralConf)
  model.init()

  def training(dataset: DataSetIterator): Unit ={

    (0 until epoch).foreach{
      epochIndex=>
        model.fit(dataset)
        dataset.reset()
    }
//    while(dataset.hasNext) {
//      val next: DataSet = dataset.next()
//      model.fit(new DataSet(next.getFeatureMatrix, next.getFeatureMatrix))
//    }
  }

  override def encode(rowdata: INDArray): INDArray = {
    val layerResults = model.feedForward(rowdata)

//    println("extract results")
//    println("(5)="+layerResults.get(5).rows()+","+layerResults.get(5).columns())
//    println(layerResults)



    println(" extracted features size:"+layerResults.get(layerNumOfOutputs.size+1).length())

    return layerResults.get(layerNumOfOutputs.size+1)
//    return rowdata;
  }

  override def encodeAndDecode(rowdata: INDArray): INDArray = {
     model.output(rowdata)
  }
}
