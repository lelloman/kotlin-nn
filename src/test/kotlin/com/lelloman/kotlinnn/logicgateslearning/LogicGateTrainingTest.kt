package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.Network
import com.lelloman.kotlinnn.Training
import com.lelloman.kotlinnn.activation.Activation
import com.lelloman.kotlinnn.layer.DenseLayer
import com.lelloman.kotlinnn.layer.GaussianWeightsInitializer
import com.lelloman.kotlinnn.layer.InputLayer
import com.lelloman.kotlinnn.layer.WeightsInitializer
import com.lelloman.kotlinnn.logicGateDataSet
import com.lelloman.kotlinnn.optimizer.SGD
import org.assertj.core.api.Assertions
import org.junit.Test

abstract class LogicGateTrainingTest {

    abstract fun f(a: Double, b: Double): Double
    abstract val label: String

    private val lossThreshold = 0.001

    private val trainingSet by lazy { logicGateDataSet(10000, ::f) }
    private val validationSet by lazy { logicGateDataSet(100, ::f) }

    private val callback = object : Training.PrintEpochCallback() {
        override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = validationLoss < lossThreshold
    }

    private val epochs = 1000

    private val logisticNetwork = makeNetwork(Activation.LOGISTIC)
    private val tanhNetwork = makeNetwork(Activation.TANH)
    private val reluNetwork = makeNetwork(Activation.RELU, GaussianWeightsInitializer(0.3, 0.2))
    private val leakyReluNetwork = makeNetwork(Activation.LEAKY_RELU, GaussianWeightsInitializer(0.3, 0.2))

    private fun makeNetwork(activation: Activation,
                            weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.3, 0.3))
            : Network {
        val inputLayer = InputLayer(2)
        val hiddenLayer = DenseLayer.Builder(8)
                .activation(activation)
                .prevLayer(inputLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        val outputLayer = DenseLayer.Builder(1)
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

    @Test
    fun `online learns logic gate with logistic activation multilayer`() {
        println("Training $label logistic activation...")
        val training = Training(logisticNetwork, trainingSet, validationSet, callback, epochs, optimizer = SGD(0.01), batchSize = 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with tanh activation multilayer`() {
        println("Training $label tanh activation...")
        val training = Training(tanhNetwork, trainingSet, validationSet, callback, epochs, optimizer = SGD(0.01), batchSize = 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with ReLU activation multilayer`() {
        println("Training $label ReLU activation...")
        val training = Training(reluNetwork, trainingSet, validationSet, callback, epochs, optimizer = SGD(0.01), batchSize = 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with leaky ReLU activation multilayer`() {
        println("Training $label leaky ReLU activation...")
        val training = Training(leakyReluNetwork, trainingSet, validationSet, callback, epochs, optimizer = SGD(0.01), batchSize = 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

}