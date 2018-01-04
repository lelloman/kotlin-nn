package com.lelloman.kotlinnn.layer

abstract class LayerActivation(val size: Int) {
    val output = DoubleArray(size)
    
    fun perform(z: DoubleArray) = (0 until size).forEach {
        output[it] = func(z[it])
    }
    
    abstract protected fun func(z: Double) : Double
    
    fun derivative(outputIndex: Int) = funcPrime(output[outputIndex])
    
    abstract protected fun funcPrime(z: Double): Double
}

class LogisticActivation(size: Int) : LayerActivation(size) {
    override fun func(z: Double) = 1.0 / (1.0 + Math.exp(-z))
    override fun funcPrime(z: Double) = z * (1 - z)
}

class TanhActivation(size: Int) : LayerActivation (size){
    override fun func(z: Double) = Math.tanh(z)
    override fun funcPrime(z: Double) = 1.0 - Math.pow(func(z), 2.0)
}

class ReluActivation(size: Int) : LayerActivation (size){
    override fun func(z: Double) = Math.min(1.0, Math.max(0.0, z))
    override fun funcPrime(z: Double) = if (z <= 0.0) 0.0 else 1.0
}

class LeakyReluActivation(size: Int) : LayerActivation (size){
    override fun func(z: Double) = Math.min(1.0, if (z < 0.0) z * 0.001 else z)
    override fun funcPrime(z: Double) = if (z <= 0.0) 0.001 else 1.0
}