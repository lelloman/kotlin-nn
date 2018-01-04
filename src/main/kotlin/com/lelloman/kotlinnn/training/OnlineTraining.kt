package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

class OnlineTraining(network: Network,
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

        trainingSet.forEach { input, targetOutput ->
            val outputActivation = network.forwardPass(input)
            loss += outputActivation.mapIndexed { index, v -> Math.pow(v - targetOutput[index], 2.0) }.sum() / trainingSet.size

            val outputLayerIndex = network.size - 1
            val outputLayer = network.layerAt(outputLayerIndex)
            val outputLayerError = neuronErrors[outputLayerIndex]
            val outputLayerGradients = weightGradients[outputLayerIndex]
            var prevActivation = outputLayer.prevLayer!!.output

            var outputWeightOffset = 0
            for (i in 0 until outputActivation.size) {
                val deltaError = (targetOutput[i] - outputActivation[i]) * outputLayer.activationDerivative(i)
                outputLayerError[i] = deltaError
                for (j in 0 until outputLayer.prevLayer.size) {
                    outputLayerGradients[outputWeightOffset++] = eta * deltaError * prevActivation[j]
                }
                if (outputLayer.hasBias) {
                    outputLayerGradients[outputWeightOffset++] = eta * deltaError
                }
            }

            outputLayer.deltaWeights(outputLayerGradients)

            for (layerIndex in network.size - 2 downTo 0) {
                val layer = network.layerAt(layerIndex)
                if (layer.isTrainable().not()) {
                    continue
                }

                val activation = layer.output
                val nextLayer = network.layerAt(layerIndex + 1)
                val nextLayerError = neuronErrors[layerIndex+1]
                val nextWeightStep = activation.size + (if (nextLayer.hasBias) 1 else 0)
                val layerError = neuronErrors[layerIndex]
                val layerGradients = weightGradients[layerIndex]
                prevActivation = layer.prevLayer!!.output

                var weightOffset = 0
                for (i in 0 until activation.size) {
                    var deltaError = 0.0
                    var nextWeightIndex = i
                    for (j in 0 until nextLayerError.size) {
                        deltaError += nextLayerError[j] * nextLayer.weightAt(nextWeightIndex)
                        nextWeightIndex += nextWeightStep
                    }
                    deltaError *= layer.activationDerivative(i)
                    layerError[i] = deltaError

                    for (j in 0 until layer.prevLayer.size) {
                        layerGradients[weightOffset++] = eta * deltaError * prevActivation[j]
                    }
                    if (layer.hasBias) {
                        layerGradients[weightOffset++] = eta * deltaError
                    }
                }
                layer.deltaWeights(layerGradients)
            }
        }

        return loss
    }
}