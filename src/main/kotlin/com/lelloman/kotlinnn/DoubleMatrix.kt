package com.lelloman.kotlinnn

typealias DoubleMatrixInitializer = (row: Int) -> DoubleArray

class DoubleMatrix(val rows: Int,
                   val columns: Int,
                   initializer: DoubleMatrixInitializer) {

    constructor(rows: Int, columns: Int) : this(rows, columns, if (columns > 0) {
        { _: Int -> DoubleArray(columns) }
    } else {
        { _: Int -> throw IllegalArgumentException("Dimensions size must be greater than 0, got $rows and $columns instead") }
    })

    val shape = rows to columns

    private val data = Array(rows, {
        initializer.invoke(it).apply {
            if (size != columns) {
                throw IllegalArgumentException("DoubleArray at index $it has size $size but declared matrix width is $columns")
            }
        }
    })

    init {
        if (rows < 1 || columns < 1) {
            throw IllegalArgumentException("Dimensions size must be greater than 0, got $rows and $columns instead")
        }
    }

    operator fun get(row: Int, column: Int) = data[row][column]

}