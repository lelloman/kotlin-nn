package com.lelloman.kotlinnn.activation

class TanhActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = Math.tanh(z)
    override fun funcPrime(y: Double) = 1.0 - Math.pow(y, 2.0)
}