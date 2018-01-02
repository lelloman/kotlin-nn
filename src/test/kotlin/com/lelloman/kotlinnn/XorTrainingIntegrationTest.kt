package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.*
import com.lelloman.kotlinnn.training.OnlineTraining
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions
import org.junit.Test
import java.util.*
import kotlin.math.roundToInt

class XorTrainingIntegrationTest {

    private val random = Random()

    private fun Double.toBoolean() = roundToInt() == 1
    private fun Boolean.toDouble() = if (this) 1.0 else 0.0

    private val f = { a: Double, b: Double -> (a.toBoolean()).xor(b.toBoolean()) }
    private val sample = { _: Int ->
        val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
        val y = f(x[0], x[1])
        x to doubleArrayOf(if (y) 1.0 else 0.0)
    }

    private val trainingSet = DataSet.Builder(1000)
            .add(sample)
            .build()
    private val validationSet = DataSet.Builder(1000)
            .add(sample)
            .build()

    private val callback = object : Training.EpochCallback {
        override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
            println("epoch $epoch training loss $trainingLoss validation loss $validationLoss")
        }
    }
    private val epochs = 100

    private val logisticNetwork = makeNetwork(LogisticActivation)
    private val tanhNetwork = makeNetwork(TanhActivation)
    private val reluNetwork = makeNetwork(ReluActivation)
    private val leakyReluNetwork = makeNetwork(LeakyReluActivation)

    private fun makeNetwork(activation: ActivationFunction): Network {
        val inputLayer = Layer.Builder()
                .size(2)
                .activation(activation)
                .build()
        val hiddenLayer = Layer.Builder()
                .size(10)
                .activation(activation)
                .prevLayer(inputLayer)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .activation(activation)
                .prevLayer(hiddenLayer)
                .build()
        return Network.Builder()
                .addLayer(inputLayer)
                .addLayer(hiddenLayer)
                .addLayer(outputLayer)
                .build()
    }

    @Test
    fun `learns XOR with logistic activation multilayer`() {
        println("Training XOR logistic activation...")
        val training = OnlineTraining(logisticNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns XOR with tanh activation multilayer`() {
        println("Training tanh logistic activation...")
        val training = OnlineTraining(tanhNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns XOR with ReLU activation multilayer`() {
        println("Training ReLU logistic activation...")
        val training = OnlineTraining(reluNetwork, trainingSet, validationSet, epochs, callback, 0.01)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }

    @Test
    fun `learns XOR with leaky ReLU activation multilayer`() {
        println("Training leaky ReLU logistic activation...")
        val training = OnlineTraining(leakyReluNetwork, trainingSet, validationSet, epochs, callback, 0.1)
        training.perform()

        val loss = training.validationLoss()
        Assertions.assertThat(loss).isLessThan(0.001)
    }
}