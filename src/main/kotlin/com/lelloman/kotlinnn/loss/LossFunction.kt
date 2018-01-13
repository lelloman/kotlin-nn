package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

interface LossFunction {

    fun onEpochStarted(outputSize: Int, dataSetSize: Int)

    /**
     * returns the error gradients
     */
    fun onEpochSample(activation: DoubleArray, target: DoubleArray): DoubleArray
    fun getEpochLoss(): Double
    fun compute(network: Network, dataSet: DataSet): Double
}

enum class Loss(val factory: () -> LossFunction) {
    MSE(::MseLoss),
    CROSS_ENTROPY(::CrossEntropyLoss)
}