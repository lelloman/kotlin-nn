package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

interface LossFunction {
    fun onEpochStarted(dataSetSize: Int)
    fun onEpochSample(activation: DoubleArray, target: DoubleArray)
    fun getEpochLoss(): Double
    fun compute(network: Network, dataSet: DataSet): Double
}

enum class Loss(val factory: () -> LossFunction) {
    MSE(::MseLoss)
}