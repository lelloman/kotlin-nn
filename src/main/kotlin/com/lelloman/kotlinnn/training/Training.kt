package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import java.util.*

open class Training(private val network: Network,
                    private val trainingSet: DataSet,
                    private val validationSet: DataSet,
                    private val epochs: Int,
                    private val callback: EpochCallback,
                    private val eta: Double = 0.01,
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

    private val neuronErrors: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).size) })
    private val weightGradients: Array<DoubleArray> = Array(network.size, { DoubleArray(network.layerAt(it).weightsSize) })

    init {
        if (trainingSet.sameDimensionAs(validationSet).not()) {
            throw IllegalArgumentException("Training set and validation set must have same dimensions, training " +
                    "input/output dimensions are ${trainingSet.inputDimension}/${trainingSet.outputDimension} while" +
                    "validation ones are ${validationSet.inputDimension}/${validationSet.outputDimension}")
        }
    }

    fun perform() = (1..epochs).forEach { epoch ->
        val trainingLoss = trainEpoch()
        val validationLoss = validationLoss()
        var end = epoch == epochs

        if (callback.shouldEndTraining(trainingLoss, validationLoss)) {
            end = true
        }
        callback.onEpoch(epoch, trainingLoss, validationLoss, end)

        if (end) {
            return
        }
    }

    private fun trainEpoch(): Double {

        var loss = 0.0
        trainingSet.shuffle()

        weightGradients.forEach { Arrays.fill(it, 0.0) }

        var sampleIndex = 0

        trainingSet.forEach { input, targetOutput ->
            val outputActivation = network.forwardPass(input)
            loss += outputActivation.mapIndexed { index, v -> Math.pow(v - targetOutput[index], 2.0) }.sum() / trainingSet.size

            val outputLayerIndex = network.size - 1
            val outputLayer = network.layerAt(outputLayerIndex)
            val outputLayerError = neuronErrors[outputLayerIndex]
            val outputLayerGradients = weightGradients[outputLayerIndex]
            var prevActivation = outputLayer.prevLayer!!.output

            var outputWeightOffset = 0
            for (i in 0 until outputActivation.size) {
                val deltaError = (targetOutput[i] - outputActivation[i]) * outputLayer.activationDerivative(i)
                outputLayerError[i] = deltaError
                for (j in 0 until outputLayer.prevLayer.size) {
                    outputLayerGradients[outputWeightOffset++] += eta * deltaError * prevActivation[j]
                }
                if (outputLayer.hasBias) {
                    outputLayerGradients[outputWeightOffset++] += eta * deltaError
                }
            }

            for (layerIndex in network.size - 2 downTo 1) {
                val layer = network.layerAt(layerIndex)
                if (layer.isTrainable().not()) {
                    continue
                }

                val activation = layer.output
                val nextLayer = network.layerAt(layerIndex + 1)
                val nextLayerError = neuronErrors[layerIndex + 1]
                val nextWeightStep = activation.size + (if (nextLayer.hasBias) 1 else 0)
                val layerError = neuronErrors[layerIndex]
                val layerGradients = weightGradients[layerIndex]
                prevActivation = layer.prevLayer!!.output

                var weightOffset = 0
                for (i in 0 until activation.size) {
                    var deltaError = 0.0
                    var nextWeightIndex = i
                    for (j in 0 until nextLayerError.size) {
                        deltaError += nextLayerError[j] * nextLayer.weightAt(nextWeightIndex)
                        nextWeightIndex += nextWeightStep
                    }
                    deltaError *= layer.activationDerivative(i)
                    layerError[i] += deltaError

                    for (j in 0 until layer.prevLayer.size) {
                        layerGradients[weightOffset++] += eta * deltaError * prevActivation[j]
                    }
                    if (layer.hasBias) {
                        layerGradients[weightOffset++] += eta * deltaError
                    }
                }
            }

            if (++sampleIndex >= batchSize) {
                (1 until network.size).forEach {
                    network.layerAt(it).deltaWeights(weightGradients[it])
                    Arrays.fill(weightGradients[it], 0.0)
                }
                sampleIndex = 0
            }
        }

        if (sampleIndex > 0) {
            (1 until network.size).forEach {
                network.layerAt(it).deltaWeights(weightGradients[it])
            }
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