package com.lelloman.kotlinnn

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test


class NetworkIntegrationTest {

    /*
        Network 1 ->
        I O
     */
    private val network1: Network by lazy {
        val inputLayer = Layer.Builder()
                .size(1)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .noBias()
                .prevLayer(inputLayer)
                .build()
        Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()
    }
    private val networkWeights1 = arrayOf(
            doubleArrayOf(0.3),
            doubleArrayOf(0.0),
            doubleArrayOf(-1.0),
            doubleArrayOf(-0.25)
    )

    private fun forwardPass1(input: Double, weights: DoubleArray) = doubleArrayOf(Math.tanh(0.0 + input * weights[0]))

    @Test
    fun `check network 1 forward pass`() {
        networkWeights1.forEach { weights ->
            network1.layerAt(1).setWeights(weights)
            doubleArrayOf(0.0, -1.0, -.5, .5, 1.0, 100.0, -100.0).forEach { input ->
                assertThat(network1.forwardPass(doubleArrayOf(input)))
                        .isEqualTo(forwardPass1(input, weights))
            }
        }
    }

    /*
        Network 2 ->
        I O
        B
     */
    private val network2: Network by lazy {
        val inputLayer = Layer.Builder()
                .size(1)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(inputLayer)
                .build()
        Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()
    }
    private val networkWeights2 = arrayOf(
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.0, 0.0),
            doubleArrayOf(-1.0, 0.5),
            doubleArrayOf(-0.25, -0.5)
    )

    private fun forwardPass2(input: Double, weights: DoubleArray) =
            doubleArrayOf(Math.tanh(input * weights[0] + weights[1]))

    @Test
    fun `check network 2 forward pass`() {
        networkWeights2.forEach { weights ->
            network2.layerAt(1).setWeights(weights)
            doubleArrayOf(0.0, -1.0, -.5, .5, 1.0, 100.0, -100.0).forEach { input ->
                assertThat(network2.forwardPass(doubleArrayOf(input)))
                        .isEqualTo(forwardPass2(input, weights))
            }
        }
    }

    /*
        Network 3 ->
        I
        I O
        B
     */
    private val network3: Network by lazy {
        val inputLayer = Layer.Builder()
                .size(2)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(inputLayer)
                .build()
        Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()
    }
    private val networkWeights3 = arrayOf(
            doubleArrayOf(0.3, 0.5, 0.1),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(-1.0, .2, .34),
            doubleArrayOf(-0.25, -0.5, -.3)
    )

    private fun forwardPass3(input: DoubleArray, weights: DoubleArray): DoubleArray {
        var v = 0.0
        v += input[0] * weights[0]
        v += input[1] * weights[1]
        v += weights[2]
        return doubleArrayOf(Math.tanh(v))
    }

    @Test
    fun `check network 3 forward pass`() {
        networkWeights3.forEach { weights ->
            network3.layerAt(1).setWeights(weights)
            arrayOf(doubleArrayOf(0.0, 0.0),
                    doubleArrayOf(-1.0, -0.2),
                    doubleArrayOf(-.5, .5),
                    doubleArrayOf(.5, -.5),
                    doubleArrayOf(1.0, 1.0),
                    doubleArrayOf(100.0, -23.0),
                    doubleArrayOf(-100.0, 0.0)
            ).forEach { input ->
                assertThat(network3.forwardPass(input))
                        .isEqualTo(forwardPass3(input, weights))
            }
        }
    }

    /*
        Network 4 ->
        I O
        I O
        B
     */
    private val network4: Network by lazy {
        val inputLayer = Layer.Builder()
                .size(2)
                .build()
        val outputLayer = Layer.Builder()
                .size(2)
                .prevLayer(inputLayer)
                .build()
        Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()
    }
    private val networkWeights4 = arrayOf(
            doubleArrayOf(0.3, 0.5, 0.1, -.3, .012, -.4),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(-1.0, .2, .34, -1.0, .2, .34),
            doubleArrayOf(-0.25, -0.5, -.3, .023, -1.0, .3)
    )

    private fun forwardPass4(input: DoubleArray, weights: DoubleArray): DoubleArray {
        var v1 = 0.0
        v1 += input[0] * weights[0]
        v1 += input[1] * weights[1]
        v1 += weights[2]

        var v2 = 0.0
        v2 += input[0] * weights[3]
        v2 += input[1] * weights[4]
        v2 += weights[5]

        return doubleArrayOf(Math.tanh(v1), Math.tanh(v2))
    }

    @Test
    fun `check network 4 forward pass`() {
        networkWeights4.forEach { weights ->
            network4.layerAt(1).setWeights(weights)
            arrayOf(doubleArrayOf(0.0, 0.0),
                    doubleArrayOf(-1.0, -0.2),
                    doubleArrayOf(-.5, .5),
                    doubleArrayOf(.5, -.5),
                    doubleArrayOf(1.0, 1.0),
                    doubleArrayOf(100.0, -23.0),
                    doubleArrayOf(-100.0, 0.0)
            ).forEach { input ->
                assertThat(network4.forwardPass(input))
                        .isEqualTo(forwardPass4(input, weights))
            }
        }
    }

    /*
       Network 5 ->
       I H O
       I H O
       B
    */
    private val network5: Network by lazy {
        val inputLayer = Layer.Builder()
                .size(2)
                .build()
        val hiddenLayer = Layer.Builder()
                .size(2)
                .prevLayer(inputLayer)
                .build()
        val outputLayer = Layer.Builder()
                .size(2)
                .prevLayer(hiddenLayer)
                .noBias()
                .build()
        Network.Builder()
                .addLayer(inputLayer)
                .addLayer(hiddenLayer)
                .addLayer(outputLayer)
                .build()
    }
    private val networkWeights5 = arrayOf(
            doubleArrayOf(0.3, 0.5, 0.1, -.3, .012, -.4) to doubleArrayOf(0.2, -0.2, .123, -.123),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) to doubleArrayOf(0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(-1.0, .2, .34, -1.0, .2, .34) to doubleArrayOf(1.0, -1.0, 0.2, 0.22),
            doubleArrayOf(-0.25, -0.5, -.3, .023, -1.0, .3) to doubleArrayOf(-.1, -.1, .1, 0.0)
    )

    private fun forwardPass5(input: DoubleArray, weights: Pair<DoubleArray, DoubleArray>): DoubleArray {
        val w1 = weights.first
        val w2 = weights.second

        var v11 = 0.0
        v11 += input[0] * w1[0]
        v11 += input[1] * w1[1]
        v11 += w1[2]

        var v12 = 0.0
        v12 += input[0] * w1[3]
        v12 += input[1] * w1[4]
        v12 += w1[5]

        val h1 = Math.tanh(v11)
        val h2 = Math.tanh(v12)

        var v21 = 0.0
        v21 += h1 * w2[0]
        v21 += h2 * w2[1]

        var v22 = 0.0
        v22 += h1 * w2[2]
        v22 += h2 * w2[3]

        return doubleArrayOf(Math.tanh(v21), Math.tanh(v22))
    }

    @Test
    fun `check network 5 forward pass`() {
        networkWeights5.forEach { weights ->
            network5.layerAt(1).setWeights(weights.first)
            network5.layerAt(2).setWeights(weights.second)
            arrayOf(doubleArrayOf(0.0, 0.0),
                    doubleArrayOf(-1.0, -0.2),
                    doubleArrayOf(-.5, .5),
                    doubleArrayOf(.5, -.5),
                    doubleArrayOf(1.0, 1.0),
                    doubleArrayOf(100.0, -23.0),
                    doubleArrayOf(-100.0, 0.0)
            ).forEach { input ->
                assertThat(network5.forwardPass(input))
                        .isEqualTo(forwardPass5(input, weights))
            }
        }
    }


}