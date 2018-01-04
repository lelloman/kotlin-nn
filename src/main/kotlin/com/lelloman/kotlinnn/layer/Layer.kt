package com.lelloman.kotlinnn.layer

class Layer private constructor(val size: Int,
                                val prevLayer: Layer?,
                                val hasBias: Boolean,
                                activationFactory: (Int) -> LayerActivation,
                                private val weightsInitializer: WeightsInitializer) {

    val isInput = prevLayer == null
    val weightsSize: Int by lazy { weights.size }

    val output: DoubleArray
        get() = activation.output

    private val z = DoubleArray(size)

    private val activation = activationFactory.invoke(size)
    private val neuronConnections: Int
    private val weights: DoubleArray

    init {
        if (isInput) {
            weights = doubleArrayOf()
            neuronConnections = 0
        } else {
            weights = DoubleArray(size * prevLayer!!.size + (if (hasBias) size else 0), { 0.0 })
            neuronConnections = size + (if (hasBias) 1 else 0)
        }
    }

    fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weights.size) {
            throw IllegalArgumentException("Weights size is supposed to be ${this.weights.size} for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weights, 0, weights.size)
    }

    fun initializeWeights() = weightsInitializer.initialize(this.weights)

    fun deltaWeights(delta: DoubleArray) {
        if (weights.size != delta.size) {
            throw IllegalArgumentException("Weight updates size is supposed to be ${weights.size} for this layer but" +
                    "argument has size ${delta.size}")
        }

        delta.forEachIndexed { index, d -> weights[index] += d }
    }

    fun weightAt(index: Int) = weights[index]

    fun setActivation(activation: DoubleArray) {
        System.arraycopy(activation, 0, this.activation.output, 0, activation.size)
    }

    fun isTrainable() = isInput.not()

    fun computeActivation() {
        val prevActivation = prevLayer!!.output
        val prevSize = prevActivation.size

        var weightOffset = 0

        for (i in 0 until size) {
            var v = (0 until prevSize).sumByDouble { prevActivation[it] * weights[weightOffset++] }
            if (hasBias) {
                v += weights[weightOffset++]
            }
            z[i] = v
        }

        activation.perform(z)
    }

    fun activationDerivative(index: Int) = activation.derivative(index)

    class Builder {
        private var size: Int? = null
        private var prevLayer: Layer? = null
        private var hasBias = true
        private var activationFactory: (Int) -> LayerActivation = { size -> LogisticActivation(size) }
        private var weightsInitializer: WeightsInitializer = GaussianWeightsInitializer(0.0, 0.3)

        fun size(size: Int) = apply {
            this.size = size
        }

        fun prevLayer(layer: Layer) = apply {
            prevLayer = layer
        }

        fun noBias() = apply {
            hasBias = false
        }

        fun activation(activationFactory: (Int) -> LayerActivation) = apply {
            this.activationFactory = activationFactory
        }

        fun weightsInitializer(weightsInitializer: WeightsInitializer) = apply {
            this.weightsInitializer = weightsInitializer
        }

        fun build(): Layer {
            if (size == null) {
                throw IllegalStateException("Must set layer size")
            }

            return Layer(size!!, prevLayer, hasBias, activationFactory, weightsInitializer)
        }
    }

}