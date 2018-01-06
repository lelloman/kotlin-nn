package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.toDouble
import com.lelloman.kotlinnn.training.OnlineTraining
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions
import org.junit.Test
import java.util.*

abstract class LogicGateTrainingTest {

    private val random = Random()

    abstract fun f(a: Double, b: Double): Boolean
    abstract val label: String

    private val lossThreshold = 0.001

    private fun sample(index: Int): Pair<DoubleArray, DoubleArray> {
        val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
        val y = f(x[0], x[1])
        return x to doubleArrayOf(if (y) 1.0 else 0.0)
    }

    private val trainingSet by lazy {
        DataSet.Builder(1000)
                .add(::sample)
                .build()
    }
    private val validationSet by lazy {
        DataSet.Builder(1000)
                .add(::sample)
                .build()
    }

    private val callback = object : Training.EpochCallback {
        override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
            println("epoch $epoch training loss $trainingLoss validation loss $validationLoss")
        }

        override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = validationLoss < lossThreshold

    }

    private val epochs = 10000

    private val logisticNetwork = makeNetwork({ size -> LogisticActivation(size) })
    private val tanhNetwork = makeNetwork({ size -> TanhActivation(size) })
    private val reluNetwork = makeNetwork({ size -> ReluActivation(size) }, GaussianWeightsInitializer(0.5, 0.2))
    private val leakyReluNetwork = makeNetwork({ size -> LeakyReluActivation(size) })

    private fun makeNetwork(activationFactory: (Int) -> LayerActivation,
                            weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
            : Network {
        val inputLayer = InputLayer(2)
        val hiddenLayer = DenseLayer.Builder()
                .size(10)
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
        val training = OnlineTraining(logisticNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with tanh activation multilayer`() {
        println("Training $label tanh activation...")
        val training = OnlineTraining(tanhNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with ReLU activation multilayer`() {
        println("Training $label ReLU activation...")
        val training = OnlineTraining(reluNetwork, trainingSet, validationSet, epochs, callback, 0.01)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `online learns logic gate with leaky ReLU activation multilayer`() {
        println("Training $label leaky ReLU activation...")
        val training = OnlineTraining(leakyReluNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

}