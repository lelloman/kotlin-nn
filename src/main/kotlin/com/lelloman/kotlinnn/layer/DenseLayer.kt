package com.lelloman.kotlinnn.layer

import com.lelloman.kotlinnn.activation.Activation

open class DenseLayer(size: Int,
                      inputLayer: Layer,
                      hasBias: Boolean = true,
                      activation: Activation = Activation.LOGISTIC,
                      private val weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
    : Layer(size, inputLayer, hasBias, activation.factory) {

    private val z = DoubleArray(size)

    override val weightsSize: Int by lazy { weights.size }
    private val weights: DoubleArray = DoubleArray(size * inputLayer!!.size + (if (hasBias) size else 0), { 0.0 })

    override fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weightsSize) {
            throw IllegalArgumentException("Weights size is supposed to be $weightsSize for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weights, 0, weights.size)
    }

    override fun initializeWeights() = weightsInitializer.initialize(this.weights)

    override fun deltaWeights(delta: DoubleArray) {
        if (weightsSize != delta.size) {
            throw IllegalArgumentException("Weight updates size is supposed to be $weightsSize for this layer but" +
                    "argument has size ${delta.size}")
        }

        delta.forEachIndexed { index, d -> weights[index] += d }
    }

    override fun weightAt(index: Int) = weights[index]

    override fun computeActivation() {
        val input = inputLayer!!.output
        val inputSize = input.size

        var weightOffset = 0

        for (i in 0 until size) {
            var v = (0 until inputSize).sumByDouble { input[it] * weights[weightOffset++] }
            if (hasBias) {
                v += weights[weightOffset++]
            }
            z[i] = v
        }

        if (isTraining) {
            activation.performWithDerivative(z)
        } else {
            activation.perform(z)
        }
    }

    override fun activationDerivative(index: Int) = activation.derivative(index)
}