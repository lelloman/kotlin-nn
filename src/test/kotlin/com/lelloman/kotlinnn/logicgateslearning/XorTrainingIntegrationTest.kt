package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.xorSample

class XorTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = xorSample(a, b)

    override val label = "XOR"
}