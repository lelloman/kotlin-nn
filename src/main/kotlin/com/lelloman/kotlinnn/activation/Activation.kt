package com.lelloman.kotlinnn.activation

enum class Activation(val factory: (Int) -> LayerActivation) {
    LOGISTIC({ LogisticActivation(it) }),
    TANH({ TanhActivation(it) }),
    RELU({ ReluActivation(it) }),
    LEAKY_RELU({ LeakyReluActivation(it) }),
    SOFTMAX({ SoftmaxActivation(it) })
}