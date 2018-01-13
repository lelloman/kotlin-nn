package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.Activation
import com.lelloman.kotlinnn.layer.DenseLayer
import com.lelloman.kotlinnn.layer.GaussianWeightsInitializer
import com.lelloman.kotlinnn.layer.InputLayer
import com.lelloman.kotlinnn.loss.Loss
import com.lelloman.kotlinnn.optimizer.SGD
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import java.util.*

class TrainingLoss {

    fun makeNetwork(random: Random): Network {
        val input = InputLayer(2)
        val hidden = DenseLayer.Builder(16)
                .weightsInitializer(GaussianWeightsInitializer(random = random))
                .prevLayer(input)
                .activation(Activation.LOGISTIC)
                .build()
        val output = DenseLayer.Builder(1)
                .weightsInitializer(GaussianWeightsInitializer(random = random))
                .prevLayer(hidden)
                .activation(Activation.LOGISTIC)
                .build()
        return Network.Builder()
                .addLayer(input)
                .addLayer(hidden)
                .addLayer(output)
                .build()
    }

    @Test
    fun asd() {
        val random1 = Random(123)
        val random2 = Random(123)
        val random = Random()
        val dataSetSize = 1000
        val samples = Array(dataSetSize, {
            val a = random.nextBoolean()
            val b = random.nextBoolean()
            doubleArrayOf(a.toDouble(), b.toDouble()) to doubleArrayOf(a.or(b).toDouble())
        })
        val dataSet1 = DataSet.Builder(dataSetSize)
                .random(random1)
                .add { samples[it] }
                .build()
        val dataSet2 = DataSet.Builder(dataSetSize)
                .random(random2)
                .add { samples[it] }
                .build()
        val validationSet = DataSet.Builder(100)
                .add {
                    val a = random.nextBoolean()
                    val b = random.nextBoolean()
                    doubleArrayOf(a.toDouble(), b.toDouble()) to doubleArrayOf(a.or(b).toDouble())
                }
                .build()

        val network1 = makeNetwork(random1)
        val network2 = makeNetwork(random2)

        val batchSize = 10
        val epochs = 10

        val errors1 = mutableListOf<Double>()
        val errors2 = mutableListOf<Double>()
        val callback1 = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                errors1.add(trainingLoss)
            }
        }

        val callback2 = object : Training2.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                errors2.add(trainingLoss)
            }
        }

        val training1 = Training(network1, dataSet1, validationSet, epochs, callback1, SGD(), batchSize)
        val training2 = Training2(network2, dataSet2, validationSet, callback2, epochs, Loss.MSE, batchSize = batchSize)

        training1.perform()
        training2.perform()

        assertThat(errors1).hasSameElementsAs(errors2)

        val hiddenSize = network1.layerAt(1).weightsSize
        for (i in 0 until hiddenSize) {
            assertThat(network1.layerAt(1).weightAt(i)).isEqualTo(network2.layerAt(1).weightAt(i))
        }
    }
}