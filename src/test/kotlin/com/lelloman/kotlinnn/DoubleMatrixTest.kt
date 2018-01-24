package com.lelloman.kotlinnn

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.Test

class DoubleMatrixTest {

    @Test
    fun `has basic get method`() {
        val matrix = DoubleMatrix(4, 4, { row -> DoubleArray(4, { row * 10.0 + it }) })

        assertThat(matrix[0, 0]).isEqualTo(0.0)
        assertThat(matrix[0, 1]).isEqualTo(1.0)
        assertThat(matrix[0, 2]).isEqualTo(2.0)
        assertThat(matrix[0, 3]).isEqualTo(3.0)

        assertThat(matrix[2, 0]).isEqualTo(20.0)
        assertThat(matrix[2, 1]).isEqualTo(21.0)
        assertThat(matrix[2, 2]).isEqualTo(22.0)
        assertThat(matrix[2, 3]).isEqualTo(23.0)

        assertThat(matrix[3, 3]).isEqualTo(33.0)
    }

    @Test
    fun `throws exception if initializer returns DoubleArray with invalid size`() {
        assertThatThrownBy { DoubleMatrix(1, 1, { DoubleArray(100) }) }
                .isInstanceOf(IllegalArgumentException::class.java)
                .hasMessageContaining("declared matrix width is")
    }

    @Test
    fun `throws exception if dimensions are smaller than 1`() {
        assertThatThrownBy { DoubleMatrix(0, 1) }
                .isInstanceOf(IllegalArgumentException::class.java)
                .hasMessageContaining("greater than 0")

        assertThatThrownBy { DoubleMatrix(1, -1) }
                .isInstanceOf(IllegalArgumentException::class.java)
                .hasMessageContaining("greater than 0")
    }

    @Test
    fun `returns shape`() {
        assertThat(DoubleMatrix(1, 1).shape).isEqualTo(1 to 1)
        assertThat(DoubleMatrix(10, 10).shape).isEqualTo(10 to 10)
    }

}