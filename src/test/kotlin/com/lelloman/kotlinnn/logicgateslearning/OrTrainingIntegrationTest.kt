package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.toBoolean

class OrTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = (a.toBoolean()).or(b.toBoolean())

    override val label = "OR"
}