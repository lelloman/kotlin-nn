package com.lelloman.kotlinnn.layer

import com.lelloman.kotlinnn.activation.Activation
import com.lelloman.kotlinnn.activation.LayerActivation
import com.lelloman.kotlinnn.activation.LogisticActivation

open class DenseLayer(size: Int,
                      prevLayer: Layer,
                      hasBias: Boolean = true,
                      activation: Activation = Activation.LOGISTIC,
                      private val weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
    : Layer(size, prevLayer, hasBias, activation.factory) {

    private val z = DoubleArray(size)

    override val weightsSize: Int by lazy { weights.size }
    private val weights: DoubleArray = DoubleArray(size * prevLayer!!.size + (if (hasBias) size else 0), { 0.0 })

    override fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weights.size) {
            throw IllegalArgumentException("Weights size is supposed to be ${this.weights.size} for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weights, 0, weights.size)
    }

    override fun initializeWeights() = weightsInitializer.initialize(this.weights)

    override fun deltaWeights(delta: DoubleArray) {
        if (weights.size != delta.size) {
            throw IllegalArgumentException("Weight updates size is supposed to be ${weights.size} for this layer but" +
                    "argument has size ${delta.size}")
        }

        delta.forEachIndexed { index, d -> weights[index] += d }
    }

    override fun weightAt(index: Int) = weights[index]

    override fun computeActivation() {
        val prevActivation = prevLayer!!.output
        val prevSize = prevActivation.size

        var weightOffset = 0

        for (i in 0 until size) {
            var v = (0 until prevSize).sumByDouble { prevActivation[it] * weights[weightOffset++] }
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