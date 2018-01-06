package com.lelloman.kotlinnn.logicgateslearning

import com.lelloman.kotlinnn.toBoolean

class XorTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = (a.toBoolean()).xor(b.toBoolean())

    override val label = "XOR"
}