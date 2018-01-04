package com.lelloman.kotlinnn.logicgateslearning

class OrTrainingIntegrationTest : LogicGateTrainingTest() {

    override fun f(a: Double, b: Double) = (a.toBoolean()).or(b.toBoolean())

    override val label = "OR"
}