package com.lelloman.kotlinnn.loss

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network
import com.nhaarman.mockito_kotlin.any
import com.nhaarman.mockito_kotlin.mock
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class MseLossTest {

    private val mseLoss = MseLoss()

    @Test
    fun `reinitializes values on epoch start`() {
        mseLoss.onEpochStarted(10, 10)

        assertThat(mseLoss.loss).isEqualTo(0.0)
        assertThat(mseLoss.dataSetSize).isEqualTo(10)
        assertThat(mseLoss.gradients).hasSize(10)

        mseLoss.loss = 10.0
        mseLoss.onEpochStarted(20, 20)

        assertThat(mseLoss.loss).isEqualTo(0.0)
        assertThat(mseLoss.dataSetSize).isEqualTo(20)
        assertThat(mseLoss.gradients).hasSize(20)
    }

    @Test
    fun `accumulates loss and stores gradients for each epoch sample`() {
        val outputSize = 1
        val dataSetSize = 3

        mseLoss.onEpochStarted(outputSize, dataSetSize)

        var gradients = mseLoss.onEpochSample(doubleArrayOf(0.0), doubleArrayOf(1.0))
        assertThat(mseLoss.getEpochLoss()).isBetween(0.33333, 0.333334)
        assertThat(gradients).isEqualTo(doubleArrayOf(1.0000))

        gradients = mseLoss.onEpochSample(doubleArrayOf(0.0), doubleArrayOf(2.0))
        assertThat(mseLoss.getEpochLoss()).isBetween(1.66666, 1.666669)
        assertThat(gradients).isEqualTo(doubleArrayOf(2.0000))

        gradients = mseLoss.onEpochSample(doubleArrayOf(0.0), doubleArrayOf(3.0))
        assertThat(mseLoss.getEpochLoss()).isBetween(4.666666, 4.6666667)
        assertThat(gradients).isEqualTo(doubleArrayOf(3.0000))
    }

    @Test
    fun `computes loss on entire dataset`() {
        val dataSet = DataSet.Builder(3)
                .add { doubleArrayOf(0.0) to doubleArrayOf(0.0) }
                .build()

        var i = 0
        val values = Array(3, { doubleArrayOf(it + 1.0) })

        val network: Network = mock {
            on { forwardPass(any()) }.thenAnswer { values[i++] }
            on { output }.thenReturn(doubleArrayOf(0.0))
        }

        val loss = mseLoss.compute(network, dataSet)
        assertThat(loss).isBetween(4.66666, 4.666669)
    }
}