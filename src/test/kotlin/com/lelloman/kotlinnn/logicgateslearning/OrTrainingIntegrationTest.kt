package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.orSample

class OrTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = orSample(a, b)

    override val label = "OR"
}