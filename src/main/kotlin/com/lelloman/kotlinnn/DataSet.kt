package com.lelloman.kotlinnn

import java.util.*

class DataSet(val input: Array<DoubleArray>,
              val output: Array<DoubleArray>,
              private val random: Random) {

    val inputDimension: Int
    val outputDimension: Int
    val size = input.size
    val randomiser = IntArray(size, { it })

    init {
        if (input.size != output.size) {
            throw IllegalArgumentException("Dataset input and output must have equal size, input has ${input.size} samples while ouptut has ${output.size}")
        }

        if (input.isEmpty()) {
            throw IllegalArgumentException("Dataset must have data in it, input and output are empty")
        }

        inputDimension = input[0].size
        input.forEachIndexed { index, value ->
            if (inputDimension != value.size) {
                throw IllegalArgumentException("All input samples must have the same dimension, input sample at index 0 has size $inputDimension while input sample at index $index has ${value.size}")
            }
        }

        outputDimension = output[0].size
        output.forEachIndexed { index, value ->
            if (outputDimension != value.size) {
                throw IllegalArgumentException("All output samples must have the same dimension, output sample at index 0 has size $outputDimension while output sample at index $index has ${value.size}")
            }
        }
    }

    fun shuffle() {
        val indices = MutableList(size, {it})
        (0 until size).forEach {
            randomiser[it] = indices.removeAt(random.nextInt(indices.size))
        }
    }

    fun sameDimensionAs(other: DataSet)
            = other.inputDimension == this.inputDimension && other.outputDimension == this.outputDimension

    inline fun forEach(action: (inSample: DoubleArray, outSample: DoubleArray) -> Unit) = (0 until size).forEach {
        val index = randomiser[it]
        action(input[index], output[index])
    }

    inline fun <reified T> map(action: (inSample: DoubleArray, outSample: DoubleArray) -> T)
            = Array(size, { action(input[it], output[it]) })

    class Builder(private val size: Int) {
        private val input = mutableListOf<DoubleArray>()
        private val output = mutableListOf<DoubleArray>()
        private var random = Random()

        fun add(action: (index: Int) -> Pair<DoubleArray, DoubleArray>) = apply{
            (0 until size).forEach {
                val sample = action(it)
                input.add(sample.first)
                output.add(sample.second)
            }
        }

        fun random(random: Random) = apply {
            this.random = random
        }

        fun build() = DataSet(input.toTypedArray(), output.toTypedArray(), random)
    }
}