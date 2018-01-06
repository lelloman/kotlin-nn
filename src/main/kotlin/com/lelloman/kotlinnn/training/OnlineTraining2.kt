package com.lelloman.kotlinnn.training

import com.lelloman.kotlinnn.DataSet
import com.lelloman.kotlinnn.Network

class OnlineTraining2(network: Network,
                      trainingSet: DataSet,
                      validationSet: DataSet,
                      epochs: Int,
                      callback: Training.EpochCallback,
                      private val eta: Double = 0.01) : BatchTraining(network, trainingSet, validationSet, epochs, callback, eta, batchSize = 1){

}