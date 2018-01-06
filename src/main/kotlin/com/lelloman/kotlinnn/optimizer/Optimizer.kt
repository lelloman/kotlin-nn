package com.lelloman.kotlinnn.optimizer

import com.lelloman.kotlinnn.Network

abstract class Optimizer(var eta: Double = 0.01) {

    lateinit var network: Network

    fun setup(network: Network) {
        this.network = network
    }

    abstract fun onStartEpoch()
    abstract fun trainOnSample(outputActivation: DoubleArray, targetOutput: DoubleArray)
    abstract fun updateWeights()
}