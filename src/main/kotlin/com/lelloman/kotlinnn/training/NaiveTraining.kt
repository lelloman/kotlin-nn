package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

class NaiveTraining(network: Network,
                    trainingSet: DataSet,
                    validationSet: DataSet,
                    epochs: Int,
                    callback: Training.EpochCallback,
                    private val eta: Double = 0.01)
    : Training(network, trainingSet, validationSet, epochs, callback) {

    private val neuronErrors: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).size) })
    private val weightGradients: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })

    override fun perform() = (1..epochs).forEach { epoch ->
        val validationLoss = validationLoss()
        val trainingLoss = trainEpoch()

        callback.onEpoch(epoch, trainingLoss, validationLoss, epoch == epochs)
    }

    private fun trainEpoch(): Double {

        var loss = 0.0

        trainingSet.forEach { input, output ->
            val activation = network.forwardPass(input).clone()
            loss += activation.mapIndexed { index, v -> Math.pow(v - output[index], 2.0) }.sum() / trainingSet.size

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