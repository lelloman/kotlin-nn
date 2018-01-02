package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.training.NaiveTraining
import com.lelloman.kotlinnn.training.Training
import org.junit.Test
import java.util.*

class NaiveTrainingIntegrationTest {

    @Test
    fun `trains simplest network`() {
        val inputLayer = Layer.Builder()
                .size(1)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(inputLayer)
                .build()

        val network = Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()

        val f = { x: Double -> .9 * x + 0.2 }
        val random = Random()
        val trainingSet = DataSet.Builder(20)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val validationSet = DataSet.Builder(20)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val epochs = 100
        val eta = 0.05
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")
            }
        }
        val training = NaiveTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }
}