package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.optimizer.Optimizer

open class Training(private val network: Network,
                    private val trainingSet: DataSet,
                    private val validationSet: DataSet,
                    private val epochs: Int,
                    private val callback: EpochCallback,
                    private val optimizer: Optimizer,
                    private val batchSize: Int = trainingSet.size) {

    interface EpochCallback {
        fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean)
        fun shouldEndTraining(trainingLoss: Double, validationLoss: Double) = false
    }

    open class PrintEpochCallback : EpochCallback {
        override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
            println("epoch $epoch training loss $trainingLoss validation loss $validationLoss")
        }
    }


    init {
        if (trainingSet.sameDimensionAs(validationSet).not()) {
            throw IllegalArgumentException("Training set and validation set must have same dimensions, training " +
                    "input/output dimensions are ${trainingSet.inputDimension}/${trainingSet.outputDimension} while" +
                    "validation ones are ${validationSet.inputDimension}/${validationSet.outputDimension}")
        }

        optimizer.setup(network)
    }

    fun perform() = (1..epochs).forEach { epoch ->
        network.setTraining(true)

        val trainingLoss = trainEpoch()
        val validationLoss = validationLoss()
        var end = epoch == epochs

        if (callback.shouldEndTraining(trainingLoss, validationLoss)) {
            end = true
        }
        callback.onEpoch(epoch, trainingLoss, validationLoss, end)

        if (end) {
            network.setTraining(false)
            return
        }
    }

    private fun trainEpoch(): Double {

        var loss = 0.0
        trainingSet.shuffle()
        optimizer.onStartEpoch()

        var sampleIndex = 0

        trainingSet.forEach { input, targetOutput ->
            val outputActivation = network.forwardPass(input)
            loss += outputActivation.mapIndexed { index, v -> Math.pow(v - targetOutput[index], 2.0) }.sum() / trainingSet.size

            optimizer.trainOnSample(outputActivation, targetOutput)
            if (++sampleIndex >= batchSize) {
                optimizer.updateWeights()
                sampleIndex = 0
            }
        }

        if (sampleIndex > 0) {
            optimizer.updateWeights()
        }

        return loss
    }

    fun validationLoss() = validationSet.map { inSample, outSample ->
        network.forwardPass(inSample)
                .mapIndexed { index, y ->
                    Math.pow(y - outSample[index], 2.0)
                }
                .sum()
    }.average()
}