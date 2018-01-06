package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.training.BatchTraining
import com.lelloman.kotlinnn.training.Training
import org.junit.Test
import java.util.*

class AutoEncoderTest {

    @Test
    fun `learns frequency with one neuron`() {

        val waveSampleSize = 16
        val random = Random(1)
        val activationFactory = ::LogisticActivation
        val weightsInitialiser = GaussianWeightsInitializer(0.5, 0.3, random)

        val input = InputLayer(waveSampleSize)
        val encodedLayer = DenseLayer.Builder()
                .prevLayer(input)
                .size(4)
                .activation(activationFactory)
                .weightsInitializer(weightsInitialiser)
                .build()
        val output = DenseLayer.Builder()
                .size(waveSampleSize)
                .prevLayer(encodedLayer)
                .activation(activationFactory)
                .weightsInitializer(weightsInitialiser)
                .build()

        val network = Network.Builder()
                .addLayer(input)
                .addLayer(encodedLayer)
                .addLayer(output)
                .build()

        val sample = { _: Int ->
            val k = 0.2 + random.nextDouble()
            val wave = DoubleArray(waveSampleSize, { Math.sin(it * k) })
            wave to wave
        }
        val trainingSet = DataSet.Builder(100000)
                .add(sample)
                .build()

        val validationSet = DataSet.Builder(100)
                .add(sample)
                .build()

        val epochs = 1000
        val callback = Training.PrintEpochCallback()
        val eta = 0.01
        val batchSize = 100
        val training = BatchTraining(network, trainingSet, validationSet,epochs, callback, eta, batchSize)
        training.perform()
    }
}