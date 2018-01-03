package com.lelloman.kotlinnn.logicgateslearning

class OrTrainingIntegrationTest : LogicGateTrainingTest() {

    override val f = { a: Double, b: Double -> (a.toBoolean()).or(b.toBoolean()) }

    override val label = "OR"
}