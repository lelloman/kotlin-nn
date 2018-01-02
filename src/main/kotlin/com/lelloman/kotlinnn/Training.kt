package com.lelloman.kotlinnn

import java.util.*

class Training(private val network: Network,
               private val trainingSet: DataSet,
               private val validationSet: DataSet,
               private val epochs: Int,
               private val callBack: EpochCallback) {

    interface EpochCallback {
        fun onEpoch(epoch: Int, loss: Double, accuracy: Double, finished: Boolean)
    }

    // holds the error of each neuron for each layer, that is BP1 and BP2 from
    // http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
    private val neuronErrors: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).size) })

    init {
        if (trainingSet.sameDimensionAs(validationSet).not()) {
            throw IllegalArgumentException("Training set and validation set must have same dimensions, training " +
                    "input/output dimensions are ${trainingSet.inputDimension}/${trainingSet.outputDimension} while" +
                    "validation ones are ${validationSet.inputDimension}/${validationSet.outputDimension}")
        }
    }

    fun perform() {
        (1..epochs).forEachIndexed { index, epoch ->
            val loss = trainEpoch()
            val accuracy = computeAccuracy()

            callBack.onEpoch(epoch, loss, accuracy, epoch == epochs)
        }
    }

    private fun trainEpoch(): Double {

        neuronErrors.forEach { Arrays.fill(it, 0.0) }

        trainingSet.forEach { input, output ->

        }

        return 0.0
    }

    private fun computeAccuracy(): Double {
        return 0.0
    }
}