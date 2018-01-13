package com.lelloman.kotlinnn.activation

class ReluActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = Math.min(1.0, Math.max(0.0, z))
    override fun funcPrime(y: Double) = if (y <= 0.0) 0.0 else 1.0
}