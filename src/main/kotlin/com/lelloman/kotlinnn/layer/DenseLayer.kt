package com.lelloman.kotlinnn.layer

open class DenseLayer internal constructor(size: Int,
                                           prevLayer: Layer?,
                                           hasBias: Boolean,
                                           activationFactory: (Int) -> LayerActivation,
                                           private val weightsInitializer: WeightsInitializer)
    : Layer(size, prevLayer, hasBias, activationFactory) {

    private val z = DoubleArray(size)

    override val weightsSize: Int by lazy { weights.size }
    private val weights: DoubleArray = DoubleArray(size * prevLayer!!.size + (if (hasBias) size else 0), { 0.0 })

    override fun setWeights(weights: DoubleArray) {
        if (weights.size != this.weights.size) {
            throw IllegalArgumentException("Weights size is supposed to be ${this.weights.size} for this layer but" +
                    "argument has size ${weights.size}")
        }

        System.arraycopy(weights, 0, this.weights, 0, weights.size)
    }

    override fun initializeWeights() = weightsInitializer.initialize(this.weights)

    override fun deltaWeights(delta: DoubleArray) {
        if (weights.size != delta.size) {
            throw IllegalArgumentException("Weight updates size is supposed to be ${weights.size} for this layer but" +
                    "argument has size ${delta.size}")
        }

        delta.forEachIndexed { index, d -> weights[index] += d }
    }

    override fun weightAt(index: Int) = weights[index]

    override fun computeActivation() {
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

    override fun activationDerivative(index: Int) = activation.derivative(index)

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

        fun build(): DenseLayer {
            if (size == null) {
                throw IllegalStateException("Must set layer size")
            }

            return DenseLayer(size!!, prevLayer, hasBias, activationFactory, weightsInitializer)
        }
    }

}