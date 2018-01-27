package com.lelloman.kotlinnn.layer

import com.lelloman.kotlinnn.activation.Activation

class RecurrentLayer(size: Int,
                     inputLayer: Layer,
                     private val returnSequence: Boolean = true,
                     hasBias: Boolean = true,
                     activation: Activation = Activation.LOGISTIC,
                     private val weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3))
    : Layer(size, inputLayer, hasBias, activation.factory) {

    private val z = DoubleArray(size)
    private val prevActivation = DoubleArray(size)

    private val weightsW: DoubleArray = DoubleArray(size * inputLayer!!.size + (if (hasBias) size else 0), { 0.0 })
    private val weightsU: DoubleArray = DoubleArray(size * size, { 0.0 })

    override val weightsSize: Int = weightsW.size + weightsU.size

    override fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weightsSize) {
            throw IllegalArgumentException("Weights size is supposed to be ${this.weightsSize} for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weightsW, 0, weightsW.size)
        System.arraycopy(weights, weightsW.size, this.weightsU, 0, weightsU.size)
    }

    override fun initializeWeights() {
        weightsInitializer.initialize(weightsW)
        weightsInitializer.initialize(weightsU)
    }

    override fun deltaWeights(delta: DoubleArray) {
        if (weightsSize != delta.size) {
            throw IllegalArgumentException("Weight updates size is supposed to be $weightsSize for this layer but" +
                    "argument has size ${delta.size}")
        }

        (0 until weightsW.size).forEach { weightsW[it] += delta[it] }
        (0 until weightsU.size).forEach { weightsU[it] += delta[it + weightsW.size] }
    }

    override fun weightAt(index: Int) = if (index < weightsW.size) {
        weightsW[index]
    } else {
        weightsU[index - weightsW.size]
    }

    override fun computeActivation() {
        if (!returnSequence) TODO("RecurrentLayer not returning sequences is not implemented yet")

        val input = inputLayer!!.output
        val inputSize = input.size

        var weightOffsetW = 0

        for (i in 0 until size) {
            var v = (0 until inputSize).sumByDouble { input[it] * weightsW[weightOffsetW++] }
            if (hasBias) {
                v += weightsW[weightOffsetW++]
            }
            z[i] = v
        }

        var weightOffsetU = 0
        for (i in 0 until size) {
            z[i] += (0 until size).sumByDouble { prevActivation[it] * weightsU[weightOffsetU++] }
        }

        if (isTraining) {
            activation.performWithDerivative(z)
        } else {
            activation.perform(z)
        }

        System.arraycopy(output, 0, prevActivation, 0, size)
    }

    override fun activationDerivative(index: Int) = activation.derivative(index)
}