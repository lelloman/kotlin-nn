package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.training.NaiveTraining
import com.lelloman.kotlinnn.training.Training
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import javax.imageio.ImageIO

class NaiveTrainingIntegrationTest {

    private val random = Random()
    private val imgSize = 1024.0

    private fun createImg() = BufferedImage(imgSize.toInt(), imgSize.toInt(), BufferedImage.TYPE_4BYTE_ABGR).apply {
        val graphics = createGraphics()
        graphics.paint = Color.BLACK
        graphics.fillRect(0, 0, width, height)
    }

    @Test
    fun `trains simplest network`() {
        println("\n\nTRAINS SIMPLEST NETWORK 1 - 1")
        val dir = File("src/test/resources", "simplest_train")
        if (dir.exists()) {
            dir.deleteRecursively()
        }
        dir.mkdir()
        println("simplest dir ${dir.absolutePath}")

        val inputLayer = Layer.Builder()
                .size(1)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(inputLayer)
                .build()

        val network = Network.Builder()
                .addLayer(inputLayer)
                .addLayer(outputLayer)
                .build()

        val f = { x: Double -> .5 * x + 0.2 }
        val trainingSet = DataSet.Builder(20)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val validationSet = DataSet.Builder(20)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val epochs = 50
        val eta = 0.05
        var prevLoss = Double.MAX_VALUE
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")
                assertThat(prevLoss).isGreaterThan(trainingLoss)
                prevLoss = trainingLoss

                val img = createImg()
                for (x in 0 until imgSize.toInt()) {
                    val actual = network.forwardPass(doubleArrayOf(x / imgSize))
                    val imageY = (actual[0] * imgSize).toInt()
                    img.setRGB(x, imageY, 0xff0000ff.toInt())
                }

                validationSet.forEach { inSample, outSample ->
                    val imageX = (inSample[0] * imgSize).toInt()
                    val targetY = (outSample[0] * imgSize).toInt()
                    val b = img.getRGB(imageX, targetY).and(0xff)
                    val p = 0xff000000.toInt() + 0xff0000 + b
                    img.setRGB(imageX, targetY, p)
                }

                val file = File(dir, "epoch_$epoch.png")
                ImageIO.write(img, "png", file)
            }
        }
        val training = NaiveTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }

    @Test
    fun `trains multilayer network on sin function`() {
        println("\n\nTRAINS NETWORK ON SIN FUN 1 - 20 - 1")
        val dir = File("src/test/resources", "sin_train")
        if (dir.exists()) {
            dir.deleteRecursively()
        }
        dir.mkdir()
        println("sin train dir ${dir.absolutePath}")
        val inputLayer = Layer.Builder()
                .size(1)
                .build()
        val hiddenLayer1 = Layer.Builder()
                .size(20)
                .prevLayer(inputLayer)
                .build()
        val hiddenLayer2 = Layer.Builder()
                .size(20)
                .prevLayer(hiddenLayer1)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(hiddenLayer2)
                .build()
        val network = Network.Builder()
                .addLayer(inputLayer)
                .addLayer(hiddenLayer1)
                .addLayer(hiddenLayer2)
                .addLayer(outputLayer)
                .build()
//        val f = { x: Double -> .9 * x + 0.2 }
        val f = { x: Double -> Math.abs(Math.sin(x * Math.PI)) }
        val trainingSet = DataSet.Builder(200)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val validationSet = DataSet.Builder(200)
                .add {
                    val x = random.nextDouble()
                    val y = f(x)
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val epochs = 100
        val eta = 0.1
        var prevLoss = Double.MAX_VALUE
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")
//                assertThat(prevLoss).isGreaterThan(trainingLoss)
                prevLoss = trainingLoss

                val img = createImg()
                for (x in 0 until imgSize.toInt()) {
                    val actual = network.forwardPass(doubleArrayOf(x / imgSize))
                    val imageY = (actual[0] * imgSize).toInt()
                    img.setRGB(x, imageY, 0xff0000ff.toInt())
                }

                validationSet.forEach { inSample, outSample ->
                    val imageX = (inSample[0] * imgSize).toInt()
                    val targetY = (outSample[0] * imgSize).toInt()
                    val b = img.getRGB(imageX, targetY).and(0xff)
                    val p = 0xff000000.toInt() + 0xff0000 + b
                    img.setRGB(imageX, targetY, p)
                }

                val file = File(dir, "epoch_$epoch.png")
                ImageIO.write(img, "png", file)
            }
        }
        val training = NaiveTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }
}