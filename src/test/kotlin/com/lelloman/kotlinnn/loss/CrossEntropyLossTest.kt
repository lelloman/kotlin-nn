package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import com.nhaarman.mockito_kotlin.any
import com.nhaarman.mockito_kotlin.mock
import org.assertj.core.api.Assertions
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class CrossEntropyLossTest {

    private val crossEntropyLoss = CrossEntropyLoss()

    @Test
    fun `reinitializes values on epoch start`() {
        crossEntropyLoss.onEpochStarted(10, 10)

        Assertions.assertThat(crossEntropyLoss.loss).isEqualTo(0.0)
        Assertions.assertThat(crossEntropyLoss.dataSetSize).isEqualTo(10)
        Assertions.assertThat(crossEntropyLoss.gradients).hasSize(10)

        crossEntropyLoss.loss = 10.0
        crossEntropyLoss.onEpochStarted(20, 20)

        Assertions.assertThat(crossEntropyLoss.loss).isEqualTo(0.0)
        Assertions.assertThat(crossEntropyLoss.dataSetSize).isEqualTo(20)
        Assertions.assertThat(crossEntropyLoss.gradients).hasSize(20)
    }

    @Test
    fun `computes loss on single sample`() {
        val activation = doubleArrayOf(0.0, 0.0, 1.0)
        val target = doubleArrayOf(0.0, 0.0, 1.0)
        crossEntropyLoss.onEpochStarted(3, 1)

        crossEntropyLoss.onEpochSample(activation, target)

        assertThat(crossEntropyLoss.loss).isEqualTo(0.0)
    }

    @Test
    fun `accumulates loss and stores gradients for each epoch sample`() {
        val outputSize = 2
        val dataSetSize = 3

        crossEntropyLoss.onEpochStarted(outputSize, dataSetSize)

        var gradients = crossEntropyLoss.onEpochSample(doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 1.0))
        assertThat(crossEntropyLoss.getEpochLoss()).isBetween(-0.000001, 0.000001)
        assertThat(gradients).isEqualTo(doubleArrayOf(-0.0, -0.0))

        gradients = crossEntropyLoss.onEpochSample(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 0.0))
        assertThat(crossEntropyLoss.getEpochLoss()).isBetween(39.911470, 39.911479)
        assertThat(gradients).isEqualTo(doubleArrayOf(9.999999999999999E25, -9.999999999999999E25))

        gradients = crossEntropyLoss.onEpochSample(doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 1.0))
        assertThat(crossEntropyLoss.getEpochLoss()).isBetween(39.911470, 39.911479)
        assertThat(gradients).isEqualTo(doubleArrayOf(-0.0, -0.0))
    }

    @Test
    fun `computes zero loss on entire dataset`() {
        val dataSet = DataSet.Builder(3)
                .add { doubleArrayOf(0.0, 0.0, 1.0) to doubleArrayOf(0.0, 0.0, 1.0) }
                .build()

        val network: Network = mock {
            on { forwardPass(any()) }.thenReturn(doubleArrayOf(0.0, 0.0, 1.0))
            on { output }.thenReturn(DoubleArray(3))
        }

        val loss = crossEntropyLoss.compute(network, dataSet)
        Assertions.assertThat(loss).isBetween(-0.000001, 0.000001)
    }

    @Test
    fun `computes non zero loss on entire dataset`() {
        val dataSet = DataSet.Builder(3)
                .add { doubleArrayOf(0.0, 0.0, 1.0) to doubleArrayOf(0.0, 0.0, 1.0) }
                .build()

        var i = 0
        val values = arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0),
                doubleArrayOf(0.0, 1.0, 0.0),
                doubleArrayOf(0.0, 0.0, 1.0)
        )

        val network: Network = mock {
            on { forwardPass(any()) }.thenAnswer { values[i++] }
            on { output }.thenReturn(DoubleArray(3))
        }

        val loss = crossEntropyLoss.compute(network, dataSet)
        Assertions.assertThat(loss).isBetween(79.822949890, 79.822949899)
    }

}