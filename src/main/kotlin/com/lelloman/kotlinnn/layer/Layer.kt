package com.lelloman.kotlinnn.layer

import com.lelloman.kotlinnn.activation.LayerActivation

abstract class Layer(val size: Int,
                     val prevLayer: Layer?,
                     val hasBias: Boolean,
                     activationFactory: (Int) -> LayerActivation) {

    protected val activation = activationFactory.invoke(size)

    var isTraining = false

    val output: DoubleArray
        get() = activation.output

    abstract val weightsSize: Int

    abstract fun setWeights(weights: DoubleArray)
    abstract fun initializeWeights()
    abstract fun deltaWeights(delta: DoubleArray)
    abstract fun weightAt(index: Int): Double

    fun setActivation(activation: DoubleArray) {
        System.arraycopy(activation, 0, this.activation.output, 0, activation.size)
    }

    fun isTrainable() = true

    abstract fun computeActivation()
    abstract fun activationDerivative(index: Int): Double
}