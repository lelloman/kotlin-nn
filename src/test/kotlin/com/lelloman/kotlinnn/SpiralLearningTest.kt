package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.Activation
import com.lelloman.kotlinnn.layer.DenseLayer
import com.lelloman.kotlinnn.layer.GaussianWeightsInitializer
import com.lelloman.kotlinnn.layer.InputLayer
import com.lelloman.kotlinnn.optimizer.SGD
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class SpiralLearningTest {

    private val imgSizeI = 512
    private val imgSizeD = imgSizeI.toDouble()

    private val trainingSet = spiralDataSet(5000)
    private val validationSet = spiralDataSet(1000)

    private var saveImages = false

    private fun saveNetworkSampling(network: Network, dirName: String, fileName: String) {
        if (saveImages.not()) return

        val img = createImage(imgSizeD)
        for (x in 0 until imgSizeI) {
            val xd = x.toDouble() / imgSizeD
            for (y in 0 until imgSizeI) {
                val outSample = network.forwardPass(doubleArrayOf(xd, y / imgSizeD))

                val r = (255 * Math.max(0.0, Math.min(1.0, outSample[0]))).toInt().shl(16)
                val g = (255 * Math.max(0.0, Math.min(1.0, outSample[1]))).toInt().shl(8)
                val b = (255 * Math.max(0.0, Math.min(1.0, outSample[2]))).toInt()
                val p = 0xff000000 + r + g + b
                img.setRGB(x, y, p.toInt())
            }
        }
        img.save(dirName, fileName)
    }

    @Test
    fun `learns spiral branch classification with SGD`() {
        val folderName = "spiral_sgd"

        val dataSetImg = createImage(imgSizeD)

        trainingSet.forEach { inSample, outSample ->
            val x = (inSample[0] * imgSizeD).toInt()
            val y = (inSample[1] * imgSizeD).toInt()

            val r = (255 * outSample[0]).toInt().shl(16)
            val g = (255 * outSample[1]).toInt().shl(8)
            val b = (255 * outSample[2]).toInt()
            val p = 0xff000000 + r + g + b
            dataSetImg.setRGB(x, y, p.toInt())
        }

        dataSetImg.save(folderName, "dataset")


        val input = InputLayer(2)
        val hidden1 = DenseLayer.Builder(16)
                .activation(Activation.LOGISTIC)
                .prevLayer(input)
                .weightsInitializer(GaussianWeightsInitializer(0.0, 0.2))
                .build()
        val hidden2 = DenseLayer.Builder(16)
                .activation(Activation.LOGISTIC)
                .prevLayer(hidden1)
                .weightsInitializer(GaussianWeightsInitializer(0.0, 0.2))
                .build()
        val output = DenseLayer.Builder(3)
                .activation(Activation.LOGISTIC)
                .prevLayer(hidden2)
                .weightsInitializer(GaussianWeightsInitializer(0.0, 0.2))
                .build()

        val network = Network.Builder()
                .addLayer(input)
                .addLayer(hidden1)
                .addLayer(hidden2)
                .addLayer(output)
                .build()

        val epochs = 1000
        val callback = object : Training.PrintEpochCallback() {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                super.onEpoch(epoch, trainingLoss, validationLoss, finished)
                if (epoch % 50 == 0) {
                    saveNetworkSampling(network, folderName, "epoch_$epoch")
                }
            }

            override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = validationLoss < 0.04
        }

        val optimizer = SGD(0.01)
        val batchSize = 10

        val training = Training(network, trainingSet, validationSet, callback, epochs, optimizer = optimizer, batchSize = batchSize)

        saveNetworkSampling(network, folderName, "before")
        training.perform()
        saveNetworkSampling(network, folderName, "trained")

        assertThat(training.validationLoss()).isLessThan(0.04)
    }
}