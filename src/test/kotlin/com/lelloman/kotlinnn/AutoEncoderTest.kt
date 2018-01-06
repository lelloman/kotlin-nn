package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.DenseLayer
import com.lelloman.kotlinnn.layer.GaussianWeightsInitializer
import com.lelloman.kotlinnn.layer.InputLayer
import com.lelloman.kotlinnn.layer.TanhActivation
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import java.util.*

class AutoEncoderTest {

    @Test
    fun `learns frequency with one neuron`() {

        val waveSampleSize = 16
        val random = Random(1)
        val weightsInitializer = GaussianWeightsInitializer(0.0, 0.2, random)

        val input = InputLayer(waveSampleSize)
        val encodedLayer = DenseLayer.Builder()
                .prevLayer(input)
                .size(8)
                .activation(::TanhActivation)
                .weightsInitializer(weightsInitializer)
                .build()
        val output = DenseLayer.Builder()
                .size(waveSampleSize)
                .prevLayer(encodedLayer)
                .activation(::TanhActivation)
                .weightsInitializer(weightsInitializer)
                .build()

        val network = Network.Builder()
                .addLayer(input)
                .addLayer(encodedLayer)
                .addLayer(output)
                .build()

        val ks = doubleArrayOf(0.1, 0.2, 0.4, 0.8, 1.6)
        val sample = { _: Int ->
            val k = ks[random.nextInt(ks.size)] + random.nextDouble() * 0.05
            val wave = DoubleArray(waveSampleSize, { Math.sin(it * k) })
            wave to wave
        }
        ks.forEach {  k ->
            val s = DoubleArray(waveSampleSize, { Math.sin(it * k) })
            println("k $k ${s.joinToString("")}")
        }
        val trainingSet = DataSet.Builder(10000)
                .add(sample)
                .build()

        val validationSet = DataSet.Builder(100)
                .add(sample)
                .build()

        val epochs = 1000
        var success = false
        val callback = object : Training.PrintEpochCallback() {
            override fun shouldEndTraining(trainingLoss: Double, validationLoss: Double): Boolean {
                success = true
                return validationLoss < 0.01
            }
        }
        val eta = 0.001
        val batchSize = 10
        val training = Training(network, trainingSet, validationSet,epochs, callback, eta, batchSize)
        training.perform()

        ks.forEach { k ->
            val wave = DoubleArray(waveSampleSize, { Math.sin(it * k) })
            val reconstructed = network.forwardPass(wave)
            val a = Array(wave.size, {"%+.2f".format(wave[it])})
            val b = Array(wave.size, {"%+.2f".format(reconstructed[it])})
            println("original: ${a.joinToString(",")}")
            println("network:  ${b.joinToString(",")}")
            println("")
        }
        assertThat(success).isTrue()
    }
}