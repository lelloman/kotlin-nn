package com.lelloman.kotlinnn

import kotlin.math.roundToInt


fun Double.toBoolean() = roundToInt() == 1
fun Boolean.toDouble() = if (this) 1.0 else 0.0