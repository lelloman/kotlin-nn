package com.lelloman.kotlinnn.optimizer

import java.util.*

class SGD(eta: Double = 0.01) : Optimizer(eta) {

    private val weightGradients by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })
    }

    private val neuronErrors by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).size) })
    }

    override fun onStartEpoch() {
        weightGradients.forEach { Arrays.fill(it, 0.0) }
    }

    override fun trainOnSample(outputActivation: DoubleArray, targetOutput: DoubleArray) {
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
                outputLayerGradients[outputWeightOffset++] += eta * deltaError * prevActivation[j]
            }
            if (outputLayer.hasBias) {
                outputLayerGradients[outputWeightOffset++] += eta * deltaError
            }
        }

        for (layerIndex in network.size - 2 downTo 1) {
            val layer = network.layerAt(layerIndex)
            if (layer.isTrainable().not()) {
                continue
            }

            val activation = layer.output
            val nextLayer = network.layerAt(layerIndex + 1)
            val nextLayerError = neuronErrors[layerIndex + 1]
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
                layerError[i] += deltaError

                for (j in 0 until layer.prevLayer.size) {
                    layerGradients[weightOffset++] += eta * deltaError * prevActivation[j]
                }
                if (layer.hasBias) {
                    layerGradients[weightOffset++] += eta * deltaError
                }
            }
        }


    }

    override fun updateWeights() {
        (1 until network.size).forEach {
            network.layerAt(it).deltaWeights(weightGradients[it])
            Arrays.fill(weightGradients[it], 0.0)
        }
    }
}