package com.lelloman.kotlinnn.activation

abstract class LayerActivation(val size: Int) {

    val output = DoubleArray(size)
    protected val derivatives = DoubleArray(size)

    open fun perform(z: DoubleArray) = (0 until size).forEach {
        output[it] = func(z[it])
    }

    open fun performWithDerivative(z: DoubleArray) = (0 until size).forEach {
        val zi = z[it]
        val v = func(zi)
        output[it] = v
        derivatives[it] = funcPrime(v)
    }

    abstract protected fun func(z: Double): Double

    fun derivative(outputIndex: Int) = derivatives[outputIndex]

    abstract protected fun funcPrime(y: Double): Double
}
