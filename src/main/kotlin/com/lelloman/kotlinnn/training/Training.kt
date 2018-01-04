package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

abstract class Training(protected val network: Network,
                        protected val trainingSet: DataSet,
                        protected val validationSet: DataSet,
                        protected val epochs: Int,
                        protected val callback: EpochCallback) {

    interface EpochCallback {
        fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean)
        fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = false
    }

    init {
        if (trainingSet.sameDimensionAs(validationSet).not()) {
            throw IllegalArgumentException("Training set and validation set must have same dimensions, training " +
                    "input/output dimensions are ${trainingSet.inputDimension}/${trainingSet.outputDimension} while" +
                    "validation ones are ${validationSet.inputDimension}/${validationSet.outputDimension}")
        }
    }

    abstract fun perform()

    fun validationLoss() = validationSet.map { inSample, outSample ->
        network.forwardPass(inSample)
                .mapIndexed { index, y ->
                    Math.pow(y - outSample[index], 2.0)
                }
                .sum()
    }.average()

}