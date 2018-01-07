package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.Network
import com.lelloman.kotlinnn.Training
import com.lelloman.kotlinnn.layer.*
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

    private val logisticNetwork = makeNetwork({ size -> LogisticActivation(size) })
    private val tanhNetwork = makeNetwork({ size -> TanhActivation(size) })
    private val reluNetwork = makeNetwork({ size -> ReluActivation(size) }, GaussianWeightsInitializer(0.3, 0.2))
    private val leakyReluNetwork = makeNetwork({ size -> LeakyReluActivation(size) }, GaussianWeightsInitializer(0.3, 0.2))

    private fun makeNetwork(activationFactory: (Int) -> LayerActivation,
                            weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.3, 0.3))
            : Network {
        val inputLayer = InputLayer(2)
        val hiddenLayer = DenseLayer.Builder()
                .size(8)
                .activation(activationFactory)
                .prevLayer(inputLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        val outputLayer = DenseLayer.Builder()
                .size(1)
                .activation(activationFactory)
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
        val training = Training(logisticNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with tanh activation multilayer`() {
        println("Training $label tanh activation...")
        val training = Training(tanhNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with ReLU activation multilayer`() {
        println("Training $label ReLU activation...")
        val training = Training(reluNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with leaky ReLU activation multilayer`() {
        println("Training $label leaky ReLU activation...")
        val training = Training(leakyReluNetwork, trainingSet, validationSet, epochs, callback, SGD(0.01), 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

}