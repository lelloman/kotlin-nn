package com.lelloman.kotlinnn.layer

interface ActivationFunction {
    fun perform(z: Double): Double
    fun derivative(z: Double): Double
}

object LogisticActivation : ActivationFunction {
    override fun perform(z: Double) = 1.0 / (1.0 + Math.exp(-z))
    override fun derivative(z: Double) = z * (1 - z)
}

object TanhActivation : ActivationFunction {
    override fun perform(z: Double) = Math.tanh(z)
    override fun derivative(z: Double) = 1.0 - Math.pow(perform(z), 2.0)
}

object ReluActivation : ActivationFunction {
    override fun perform(z: Double) = Math.min(1.0, Math.max(0.0, z))
    override fun derivative(z: Double) = if (z <= 0.0) 0.0 else 1.0
}

object LeakyReluActivation : ActivationFunction {
    override fun perform(z: Double) = Math.min(1.0, if (z < 0.0) z * 0.001 else z)
    override fun derivative(z: Double) = if (z <= 0.0) 0.001 else 1.0
}