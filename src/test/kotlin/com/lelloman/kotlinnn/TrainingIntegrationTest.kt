package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.optimizer.SGD
import org.assertj.core.api.Assertions
import org.junit.Test
import java.util.*

class TrainingIntegrationTest {

    private val random = Random()

    private val lossThreshold = 0.001

    private val trainingSet by lazy { xorDataSet(10000) }
    private val validationSet by lazy { xorDataSet(100) }

    private val logisticNetwork = makeNetwork(Activation.LOGISTIC, GaussianWeightsInitializer(0.0, 0.3))
    private val leakyReluNetwork = makeNetwork(Activation.LEAKY_RELU, GaussianWeightsInitializer(0.4, 0.3))

    private val callback = object : Training.PrintEpochCallback() {
        override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = validationLoss < lossThreshold
    }

    private fun makeNetwork(activation: Activation, weightsInitializer: WeightsInitializer)
            : Network {
        val inputLayer = InputLayer(2)
        val hiddenLayer = DenseLayer.Builder()
                .size(8)
                .activation(activation)
                .prevLayer(inputLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        val outputLayer = DenseLayer.Builder()
                .size(1)
                .activation(activation)
                .prevLayer(hiddenLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        return Network.Builder()
                .addLayer(inputLayer)
                .addLayer(hiddenLayer)
                .addLayer(outputLayer)
                .build()
    }

    private val epochs = 10000

    @Test
    fun `batch size 10 XOR with logistic activation`() {
        println("Training XOR batch size 10 logistic activation...")
        val training = Training(logisticNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 100 XOR with logistic activation`() {
        println("Training XOR batch size 100 logistic activation...")
        val training = Training(logisticNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 100)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch full size XOR with logistic activation`() {
        println("Training XOR batch size 100 logistic activation...")
        val training = Training(logisticNetwork, trainingSet, validationSet, epochs, callback, SGD(0.005))
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 10 XOR with leaky ReLU activation`() {
        println("Training XOR batch size 10 leaky ReLU activation...")
        val training = Training(leakyReluNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 100 XOR with leaky ReLU activation`() {
        println("Training XOR batch size 100 leaky ReLU activation...")
        val training = Training(leakyReluNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 100)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch full size XOR with leaky ReLU activation`() {
        println("Training XOR batch full size leaky ReLU activation...")
        val training = Training(leakyReluNetwork, trainingSet, validationSet, epochs, callback, SGD(0.00005))
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }
}