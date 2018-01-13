package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

internal class MseLoss : LossFunction {

    private var loss = 0.0
    private var dataSetSize = 0

    override fun onEpochStarted(dataSetSize: Int) {
        this.dataSetSize = dataSetSize
        loss = 0.0
    }

    override fun onEpochSample(activation: DoubleArray, target: DoubleArray) {
        loss += activation.mapIndexed { index, v -> Math.pow(v - target[index], 2.0) }.sum() / dataSetSize
    }

    override fun getEpochLoss(): Double = loss

    override fun compute(network: Network, dataSet: DataSet): Double {
        onEpochStarted(dataSet.size)
        dataSet.map { inSample, outSample ->
            onEpochSample(network.forwardPass(inSample), outSample)
        }
        return loss
    }
}