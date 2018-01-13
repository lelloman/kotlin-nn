package com.lelloman.kotlinnn.activation

class InputActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = throw RuntimeException("cannot compute InputActivation")
    override fun funcPrime(y: Double) = throw RuntimeException("cannot differentiate InputActivation")
}