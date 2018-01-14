package com.lelloman.kotlinnn.optimizer

import com.lelloman.kotlinnn.Network
import java.util.*

open class SGD(private var eta: Double = 0.01) {

    protected lateinit var network: Network

    protected val weightGradients by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })
    }

    private val neuronErrors by lazy {
        Array(network.size, { DoubleArray(network.layerAt(it).size) })
    }

    fun setup(network: Network) {
        this.network = network
    }

    fun onStartEpoch() {
        weightGradients.forEach { Arrays.fill(it, 0.0) }
    }

    open fun trainOnSample(outputError: DoubleArray) {

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

    open fun updateWeights() = (1 until network.size).forEach {
        network.layerAt(it).deltaWeights(weightGradients[it])

        Arrays.fill(weightGradients[it], 0.0)
    }
}
