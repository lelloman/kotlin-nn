package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import java.util.*

class NaiveTraining(network: Network,
                    trainingSet: DataSet,
                    validationSet: DataSet,
                    epochs: Int,
                    callback: Training.EpochCallback,
                    private val eta: Double = 0.01)
    : Training(network, trainingSet, validationSet, epochs, callback) {

    // holds the error of each neuron for each layer, that is BP1 and BP2 from
    // http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
    private val neuronErrors: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).size) })

    private val weightGradients: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })

    override fun perform() = (1..epochs).forEach { epoch ->
        val loss = trainEpoch()
        val accuracy = computeAccuracy()

        callback.onEpoch(epoch, loss, accuracy, epoch == epochs)
    }

    private fun trainEpoch(): Double {

        var loss = 0.0

        trainingSet.forEach { input, output ->
            val activation = network.forwardPass(input).clone()
            loss += activation.mapIndexed { index, v -> Math.pow(output[index] - v, 2.0) }.sum() / trainingSet.size

            val outputLayerIndex = network.size - 1
            val outputLayer = network.layerAt(outputLayerIndex)
            val outputLayerError = neuronErrors[outputLayerIndex]
            val outputLayerGradients = weightGradients[outputLayerIndex]
            val prevActivation = outputLayer.prevLayer!!.activation

            var outputWeightOffset = 0
            for (i in 0 until activation.size) {
                val deltaError = (output[i] - activation[i]) * outputLayer.activationDerivative(i)
                outputLayerError[i] = deltaError
                for (j in 0 until outputLayer.prevLayer!!.size) {
                    outputLayerGradients[outputWeightOffset++] = eta * deltaError * prevActivation[j]
                }
                if (outputLayer.hasBias) {
                    outputLayerGradients[outputWeightOffset++] = eta * deltaError
                }
            }

            outputLayer.deltaWeights(outputLayerGradients)
        }

        return loss
    }
}