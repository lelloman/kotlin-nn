package com.lelloman.kotlinnn

class Layer private constructor(val size: Int, val prevLayer: Layer?, val hasBias: Boolean) {

    val isInput = prevLayer == null
    val activation = DoubleArray(size, { 0.0 })
    val weightsSize: Int by lazy { weights.size }

    private val neuronConnections: Int
    private val weights: DoubleArray

    init {
        if (isInput) {
            weights = doubleArrayOf()
            neuronConnections = 0
        } else {
            weights = DoubleArray(size * prevLayer!!.size + (if (hasBias) size else 0), { 0.0 })
            neuronConnections = size + (if (hasBias) 1 else 0)
        }
    }

    fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weights.size) {
            throw IllegalArgumentException("Weights size is supposed to be ${this.weights.size} for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weights, 0, weights.size)
    }

    fun setActivation(activation: DoubleArray) {
        System.arraycopy(activation, 0, this.activation, 0, activation.size)
    }

    fun computeActivation() {
        val prevActivation = prevLayer!!.activation
        val prevSize = prevActivation.size

        var weightOffset = 0

        for (i in 0 until size) {
            var v = 0.0
            for (j in 0 until prevSize) {
                v += prevActivation[j] * weights[weightOffset++]
            }
            if (hasBias) {
                v += weights[weightOffset++]
            }

            activation[i] = activationFunction(v)
        }
    }

    private fun activationFunction(x: Double) = Math.tanh(x)

    class Builder() {
        private var size: Int? = null
        private var prevLayer: Layer? = null
        private var hasBias = true

        fun size(size: Int): Builder {
            this.size = size
            return this
        }

        fun prevLayer(layer: Layer): Builder {
            prevLayer = layer
            return this
        }

        fun noBias(): Builder {
            hasBias = false
            return this
        }

        fun build(): Layer {
            if (size == null) {
                throw IllegalStateException("Must set layer size")
            }

            return Layer(size!!, prevLayer, hasBias)
        }
    }

}