package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.Layer

class Network private constructor(private val layers: Array<Layer>) {

    val size: Int = layers.size

    private val forwardLayers: Array<Layer> = layers.sliceArray(IntRange(1, layers.size - 1))

    fun forwardPass(input: DoubleArray): DoubleArray {
        layers[0].setActivation(input)

        forwardLayers.forEach { it.computeActivation() }

        return layers.last().activation
    }

    fun layerAt(index: Int) = layers[index]

    class Builder {
        private val layers = mutableListOf<Layer>()

        fun addLayer(layer: Layer) = apply {
            layers.add(layer)
        }

        fun build(): Network {
            if (layers.size < 2) {
                throw IllegalStateException("A network must have at least an input and an output layer")
            }

            if (layers[0].isInput.not()) {
                throw IllegalStateException("The first layer of the network must be input")
            }

            (1 until layers.size)
                    .filter { layers[it].isInput }
                    .forEach { throw IllegalStateException("Layer $it is an input layer, only the first layer can be input") }

            // TODO check for circular network

            layers.forEach {
                it.initializeWeights()
            }

            return Network(layers.toTypedArray())
        }
    }
}