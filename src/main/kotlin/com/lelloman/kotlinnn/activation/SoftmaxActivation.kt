package com.lelloman.kotlinnn.activation

class SoftmaxActivation(size: Int) : LayerActivation(size) {

    private val exp = DoubleArray(size)

    override fun perform(z: DoubleArray) {
        var sum = 0.0
        (0 until size).forEach {
            val v = Math.exp(z[it])
            exp[it] = v
            sum += v
        }

        (0 until size).forEach {
            output[it] = exp[it] / sum
        }
    }

    override fun performWithDerivative(z: DoubleArray) {
        var sum = 0.0
        (0 until size).forEach {
            val v = Math.exp(z[it])
            exp[it] = v
            sum += v
        }

        (0 until size).forEach {
            val a = exp[it]
            val v = a / sum
            output[it] = v
            derivatives[it] = v * (1 - v) * a
        }
    }

    override fun func(z: Double) = Math.exp(z)
    override fun funcPrime(y: Double) = y * (1 - y)
}