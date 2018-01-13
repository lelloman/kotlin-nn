package com.lelloman.kotlinnn.activation

class LogisticActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = 1.0 / (1.0 + Math.exp(-z))
    override fun funcPrime(y: Double) = y * (1 - y)
}