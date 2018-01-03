package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.training.OnlineTraining
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions
import org.junit.Test
import java.util.*
import kotlin.math.roundToInt

abstract class LogicGateTrainingTest {

    private val random = Random()

    protected fun Double.toBoolean() = roundToInt() == 1
    private fun Boolean.toDouble() = if (this) 1.0 else 0.0

    abstract val f: (Double, Double) -> Boolean
    abstract val label: String

    private val sample = { _: Int ->
        val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
        val y = f(x[0], x[1])
        x to doubleArrayOf(if (y) 1.0 else 0.0)
    }

    private val trainingSet by lazy {
        DataSet.Builder(1000)
                .add(sample)
                .build()
    }
    private val validationSet by lazy {
        DataSet.Builder(1000)
                .add(sample)
                .build()
    }

    private val callback = object : Training.EpochCallback {
        override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
            println("epoch $epoch training loss $trainingLoss validation loss $validationLoss")
        }

    }
    private val epochs = 100

    private val logisticNetwork = makeNetwork(LogisticActivation)
    private val tanhNetwork = makeNetwork(TanhActivation)
    private val reluNetwork = makeNetwork(ReluActivation, GaussianWeightsInitializer(0.4, 0.2))
    private val leakyReluNetwork = makeNetwork(LeakyReluActivation)

    private fun makeNetwork(activation: ActivationFunction,
                            weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
            : Network {
        val inputLayer = Layer.Builder()
                .size(2)
                .activation(activation)
                .build()
        val hiddenLayer = Layer.Builder()
                .size(10)
                .activation(activation)
                .prevLayer(inputLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        val outputLayer = Layer.Builder()
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

    @Test
    fun `learns OR with logistic activation multilayer`() {
        println("Training $label logistic activation...")
        val training = OnlineTraining(logisticNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns OR with tanh activation multilayer`() {
        println("Training $label tanh activation...")
        val training = OnlineTraining(tanhNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns OR with ReLU activation multilayer`() {
        println("Training $label ReLU activation...")
        val training = OnlineTraining(reluNetwork, trainingSet, validationSet, epochs, callback, 0.01)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns OR with leaky ReLU activation multilayer`() {
        println("Training $label leaky ReLU activation...")
        val training = OnlineTraining(leakyReluNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }
}