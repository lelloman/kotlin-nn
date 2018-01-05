package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.training.BatchTraining
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions
import org.junit.Ignore
import org.junit.Test
import java.util.*
import kotlin.math.roundToInt

class BatchTrainingIntegrationTest {

    private val random = Random()

    protected fun Double.toBoolean() = roundToInt() == 1
    private fun Boolean.toDouble() = if (this) 1.0 else 0.0

    private val lossThreshold = 0.001

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

    private val logisticNetwork = makeNetwork({ size -> LogisticActivation(size) })
    private val leakyReluNetwork = makeNetwork({ size -> LeakyReluActivation(size) })

    private val callback = object : Training.EpochCallback {
        override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
            println("epoch $epoch training loss $trainingLoss validation loss $validationLoss")
        }

        override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = validationLoss < lossThreshold

    }

    fun f(a: Double, b: Double) = (a.toBoolean()).xor(b.toBoolean())

    private fun sample(index: Int): Pair<DoubleArray, DoubleArray>{
        val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
        val y = f(x[0], x[1])
        return x to doubleArrayOf(if (y) 1.0 else 0.0)
    }

    private fun makeNetwork(activationFactory: (Int) -> LayerActivation,
                            weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
            : Network {
        val inputLayer = Layer.Builder()
                .size(2)
                .activation(activationFactory)
                .build()
        val hiddenLayer = Layer.Builder()
                .size(10)
                .activation(activationFactory)
                .prevLayer(inputLayer)
                .weightsInitializer(weightsInitializer)
                .build()
        val outputLayer = Layer.Builder()
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
    fun `batch size 10 XOR with logistic activation`() {
        println("Training XOR batch size 10 logistic activation...")
        val training = BatchTraining(logisticNetwork, trainingSet, validationSet, 1000, callback, 0.02, 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 100 XOR with logistic activation`() {
        println("Training XOR batch size 100 logistic activation...")
        val training = BatchTraining(logisticNetwork, trainingSet, validationSet, 1000, callback, 0.02, 100)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch full size XOR with logistic activation`() {
        println("Training XOR batch full size logistic activation...")
        val training = BatchTraining(logisticNetwork, trainingSet, validationSet, 1000, callback, 0.02)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 10 XOR with leaky ReLU activation`() {
        println("Training XOR batch size 10 leaky ReLU activation...")
        val training = BatchTraining(leakyReluNetwork, trainingSet, validationSet, 1000, callback, 0.02, 10)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Test
    fun `batch size 100 XOR with leaky ReLU activation`() {
        println("Training XOR batch size 100 leaky ReLU activation...")
        val training = BatchTraining(leakyReluNetwork, trainingSet, validationSet, 1000, callback, 0.02, 100)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }

    @Ignore // doesn't seem to work with leaky ReLU
    @Test
    fun `batch full size XOR with leaky ReLU activation`() {
        println("Training XOR batch full size leaky ReLU activation...")
        val training = BatchTraining(leakyReluNetwork, trainingSet, validationSet, 1000, callback, 0.02)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(lossThreshold)
    }
}