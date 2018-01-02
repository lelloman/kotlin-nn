package com.lelloman.kotlinnn

class DataSet(val input: Array<DoubleArray>, val output: Array<DoubleArray>) {

    val inputDimension: Int
    val outputDimension: Int
    val size = input.size

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


    fun sameDimensionAs(other: DataSet)
            = other.inputDimension == this.inputDimension && other.outputDimension == this.outputDimension

    inline fun forEach(action: (inSample: DoubleArray, outSample: DoubleArray) -> Unit) = (0 until size).forEach {
        action(input[it], output[it])
    }

}