package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.dataset.DataSet
import com.lelloman.kotlinnn.Network

internal class CrossEntropyLoss : LossFunction {

    internal var loss = 0.0
    internal var dataSetSize = 0
    internal lateinit var gradients: DoubleArray

    override fun onEpochStarted(outputSize: Int, dataSetSize: Int) {
        this.dataSetSize = dataSetSize
        gradients = DoubleArray(outputSize)
        loss = 0.0
    }

    override fun onEpochSample(activation: DoubleArray, target: DoubleArray): DoubleArray {
        val sum = activation.mapIndexed { index, y ->
            val t = target[index]
            val oneMinusY = (1 - y)

            gradients[index] = -(y - t) / ((y * oneMinusY) + EPSILON)
            val output = -t * Math.log(y + EPSILON) - (1 - t) * Math.log(oneMinusY + EPSILON)
            output
        }.sum()
        loss += sum / dataSetSize
        return gradients
    }

    override fun getEpochLoss(): Double = loss

    override fun compute(network: Network, dataSet: DataSet): Double {
        onEpochStarted(network.output.size, dataSet.size)
        dataSet.samples.map { (inSample, outSample) ->
            onEpochSample(network.forwardPass(inSample[0]), outSample[0])
        }
        return loss
    }

    companion object {
        private const val EPSILON = 0.00000000000000000000000001
    }
}