package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

internal class CrossEntropyLoss : LossFunction {

    private var loss = 0.0
    private var dataSetSize = 0
    private lateinit var gradients: DoubleArray

    override fun onEpochStarted(outputSize: Int, dataSetSize: Int) {
        this.dataSetSize = dataSetSize
        gradients = DoubleArray(outputSize)
        loss = 0.0
    }

    override fun onEpochSample(activation: DoubleArray, target: DoubleArray): DoubleArray {
        loss += activation.mapIndexed { index, y ->
            val t = target[index]
            val oneMinusY = 1 - y

            gradients[index] = -(y - t) / (y * oneMinusY)
            -t * Math.log(y) - (1 - t) * Math.log(oneMinusY)
        }.sum() / dataSetSize
        return gradients
    }

    override fun getEpochLoss(): Double = loss

    override fun compute(network: Network, dataSet: DataSet): Double {
        onEpochStarted(network.output.size, dataSet.size)
        dataSet.map { inSample, outSample ->
            onEpochSample(network.forwardPass(inSample), outSample)
        }
        return loss
    }
}