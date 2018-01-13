package com.lelloman.kotlinnn.layer

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

class InputActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = throw RuntimeException("cannot compute InputActivation")
    override fun funcPrime(y: Double) = throw RuntimeException("cannot differentiate InputActivation")
}

class LogisticActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = 1.0 / (1.0 + Math.exp(-z))
    override fun funcPrime(y: Double) = y * (1 - y)
}

class TanhActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = Math.tanh(z)
    override fun funcPrime(y: Double) = 1.0 - Math.pow(y, 2.0)
}

class ReluActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = Math.min(1.0, Math.max(0.0, z))
    override fun funcPrime(y: Double) = if (y <= 0.0) 0.0 else 1.0
}

class LeakyReluActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = Math.min(1.0, if (z < 0.0) z * 0.001 else z)
    override fun funcPrime(y: Double) = if (y <= 0.0) 0.001 else 1.0
}

class SoftMaxActivation(size: Int) : LayerActivation(size) {

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

enum class Activation(val factory: (Int) -> LayerActivation) {
    LOGISTIC({ LogisticActivation(it) }),
    TANH({ TanhActivation(it) }),
    RELU({ ReluActivation(it) }),
    LEAKY_RELU({ LeakyReluActivation(it) }),
    SOFTMAX({ SoftMaxActivation(it) })
}