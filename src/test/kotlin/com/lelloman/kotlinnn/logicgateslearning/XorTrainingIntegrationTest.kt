package com.lelloman.kotlinnn.logicgateslearning

class XorTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = (a.toBoolean()).xor(b.toBoolean())

    override val label = "XOR"
}