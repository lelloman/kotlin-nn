package com.lelloman.kotlinnn.layer

import com.lelloman.kotlinnn.activation.InputActivation

class InputLayer(size: Int) : Layer(size, null, false, { size: Int -> InputActivation(size) }) {

    override val weightsSize = 0

    override fun setWeights(weights: DoubleArray) {
        throw RuntimeException("Cannot set weights of an InputLayer")
    }

    override fun initializeWeights() {
        throw RuntimeException("Cannot initialize weights of an InputLayer")
    }

    override fun deltaWeights(delta: DoubleArray) {
        throw RuntimeException("Cannot modify weights of an InputLayer")
    }

    override fun weightAt(index: Int): Double {
        throw RuntimeException("Cannot get weights from an InputLayer")
    }

    override fun computeActivation() {
        throw RuntimeException("Cannot compute activation of an InputLayer")
    }

    override fun activationDerivative(index: Int): Double {
        throw RuntimeException("Cannot compute derivative of the activation of an InputLayer")
    }
}