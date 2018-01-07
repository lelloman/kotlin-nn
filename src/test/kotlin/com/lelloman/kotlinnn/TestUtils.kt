package com.lelloman.kotlinnn

import java.util.*
import kotlin.math.roundToInt


fun Double.toBoolean() = roundToInt() == 1
fun Boolean.toDouble() = if (this) 1.0 else 0.0

private val random = Random()

fun xorSample(a: Double, b: Double) = a.toBoolean().xor(b.toBoolean()).toDouble()
fun orSample(a: Double, b: Double) = a.toBoolean().or(b.toBoolean()).toDouble()

fun logicGateDataSet(size: Int, f: (a: Double, b: Double) -> Double) = DataSet.Builder(size)
        .add {
            val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
            x to doubleArrayOf(f(x[0], x[1]))
        }
        .build()

fun xorDataSet(size: Int) = DataSet.Builder(size)
        .add {
            val x = doubleArrayOf(random.nextBoolean().toDouble(), random.nextBoolean().toDouble())
            x to doubleArrayOf(xorSample(x[0], x[1]))
        }
        .build()