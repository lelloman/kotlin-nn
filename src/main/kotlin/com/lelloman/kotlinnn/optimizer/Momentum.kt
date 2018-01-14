package com.lelloman.kotlinnn.optimizer

import java.util.*

class Momentum(eta: Double = 0.01, private val momentum: Double? = null) : SGD(eta) {

    private val prevWeightGradients by lazy {
        Array(network.size, { DoubleArray(weightGradients[it].size) })
    }

    override fun updateWeights() = (1 until network.size).forEach {

        val gradients = weightGradients[it]
        momentum?.let { m ->
            val prevGradients = prevWeightGradients[it]
            prevGradients.forEachIndexed { gradientIndex, prevGradient ->
                val currentGradient = gradients[gradientIndex] + prevGradient * m
                gradients[gradientIndex] = currentGradient
                prevGradients[gradientIndex] = currentGradient
            }
        }
        network.layerAt(it).deltaWeights(gradients)

        Arrays.fill(weightGradients[it], 0.0)
    }

}