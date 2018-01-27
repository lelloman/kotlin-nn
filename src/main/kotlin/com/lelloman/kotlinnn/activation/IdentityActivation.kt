package com.lelloman.kotlinnn.activation

class IdentityActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = z
    override fun funcPrime(y: Double) = 0.0
}