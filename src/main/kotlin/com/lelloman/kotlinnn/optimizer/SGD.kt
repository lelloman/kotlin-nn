package com.lelloman.kotlinnn.optimizer

import java.util.*

class SGD(eta: Double = 0.01, private val momentum: Double? = null) : Optimizer(eta) {

    private val weightGradients by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })
    }

    private val prevWeightGradients by lazy {
        Array(network.size, { DoubleArray(weightGradients[it].size) })
    }

    private val neuronErrors by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).size) })
    }

    override fun onStartEpoch() {
        weightGradients.forEach { Arrays.fill(it, 0.0) }
    }

    override fun trainOnSample(outputActivation: DoubleArray, outputError: DoubleArray) {

        for (layerIndex in network.size - 1 downTo 1) {
            val layer = network.layerAt(layerIndex)

            val layerGradients = weightGradients[layerIndex]
            val layerError = neuronErrors[layerIndex]
            var weightOffset = 0
            val activation = layer.output
            val prevActivation = layer.prevLayer!!.output

            val isOutputLayer = layerIndex == network.size - 1

            val nextLayer = if (!isOutputLayer) network.layerAt(layerIndex + 1) else null
            val nextLayerError = if (!isOutputLayer) neuronErrors[layerIndex + 1] else null
            val nextWeightStep = activation.size + (if (nextLayer?.hasBias == true) 1 else 0)

            for (i in 0 until activation.size) {
                var deltaError = if (isOutputLayer) {
                    outputError[i]
                } else {
                    var offset = i
                    (0 until nextLayerError!!.size).sumByDouble {
                        val v = nextLayerError[it] * nextLayer!!.weightAt(offset)
                        offset += nextWeightStep
                        v
                    }
                }

                deltaError *= layer.activationDerivative(i)
                layerError[i] = deltaError
                if (layer.isTrainable()) {
                    for (j in 0 until layer.prevLayer.size) {
                        layerGradients[weightOffset++] += eta * deltaError * prevActivation[j]
                    }
                    if (layer.hasBias) {
                        layerGradients[weightOffset++] += eta * deltaError
                    }
                }
            }
        }
    }

    override fun updateWeights() = (1 until network.size).forEach {

        val gradients = weightGradients[it]
        momentum?.let { m ->
            val prevGradients = prevWeightGradients[it]
            prevGradients.forEachIndexed {gradientIndex, prevGradient ->
                val currentGradient = gradients[gradientIndex] + prevGradient * m
                gradients[gradientIndex] = currentGradient
                prevGradients[gradientIndex] = currentGradient
            }
        }
        network.layerAt(it).deltaWeights(gradients)

        Arrays.fill(weightGradients[it], 0.0)
    }

}