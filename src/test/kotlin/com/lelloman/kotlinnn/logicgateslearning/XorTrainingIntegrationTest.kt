package com.lelloman.kotlinnn.logicgateslearning

class XorTrainingIntegrationTest : LogicGateTrainingTest() {

    override val f = { a: Double, b: Double -> (a.toBoolean()).xor(b.toBoolean()) }

    override val label = "XOR"
}