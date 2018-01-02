package com.lelloman.kotlinnn

import com.lelloman.kotlinnn.layer.Layer
import com.lelloman.kotlinnn.training.OnlineTraining
import com.lelloman.kotlinnn.training.Training
import org.junit.Ignore
import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import javax.imageio.ImageIO

class OnlineTrainingIntegrationTest {

    private val random = Random()
    private val imgSize = 1024.0

    private fun createImg() = BufferedImage(imgSize.toInt(), imgSize.toInt(), BufferedImage.TYPE_4BYTE_ABGR).apply {
        val graphics = createGraphics()
        graphics.paint = Color.BLACK
        graphics.fillRect(0, 0, width, height)
    }

    @Ignore
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
        val epochs = 100
        val eta = 0.5
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")

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
        val training = OnlineTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }

    @Ignore
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
        val hiddenLayer = Layer.Builder()
                .size(25)
                .prevLayer(inputLayer)
                .build()
        val outputLayer = Layer.Builder()
                .size(1)
                .prevLayer(hiddenLayer)
                .build()
        val network = Network.Builder()
                .addLayer(inputLayer)
                .addLayer(hiddenLayer)
                .addLayer(outputLayer)
                .build()
        val f = { x: Double -> (1.0 + Math.sin(x * Math.PI * 3)) / 2.0 }
        val trainingSet = DataSet.Builder(1000)
                .add {
                    val x = random.nextDouble() * .8 + .1
                    val y = f(x) * .8 + .1
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val validationSet = DataSet.Builder(200)
                .add {
                    val x = random.nextDouble() * .8 + .1
                    val y = f(x) * .8 + .1
                    doubleArrayOf(x) to doubleArrayOf(y)
                }
                .build()
        val epochs = 50
        val eta = 0.1
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")
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
        val training = OnlineTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }

    @Ignore
    @Test
    fun `train 2d input with 1d output multilayer`() {
        val f = { x1: Double, x2: Double ->
            val s1 = Math.pow(x1, 2.0) + x1 * .5 + 2.0
            val s2 = 1.0 + Math.sin(x2 * Math.PI * 3) / 2
            (s1 + s2) / 2.0
        }
        val sample = { _: Int ->
            val x1 = random.nextDouble() * .8 + .1
            val x2 = random.nextDouble() * .8 + .1
            val y = f(x1, x2)
            doubleArrayOf(x1, x2) to doubleArrayOf(y)
        }
        val dir = File("src/test/resources", "2d_train")
        if (dir.exists()) {
            dir.deleteRecursively()
        }
        dir.mkdir()
        val img = createImg()
        for (x in 0 until imgSize.toInt()) {
            for (y in 0 until imgSize.toInt()) {
                val p = 0xff000000.toInt() + (f(x / imgSize, y / imgSize) * 256.0).toInt()
                img.setRGB(x, y, p)
            }
        }
        ImageIO.write(img, "png", File(dir, "data.png"))

        val trainingSet = DataSet.Builder(50000)
                .add(sample)
                .build()
        val validationSet = DataSet.Builder(200)
                .add(sample)
                .build()
        println("\n\nTRAINS NETWORK ON 2D FUN")

        val inputLayer = Layer.Builder()
                .size(2)
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
        val epochs = 50
        val eta = 0.01
        val callback = object : Training.EpochCallback {
            override fun onEpoch(epoch: Int, trainingLoss: Double, validationLoss: Double, finished: Boolean) {
                println("epoch $epoch training loss $trainingLoss valid loss $validationLoss")
                if (finished) {
                    val img = createImg()
                    for (x in 0 until imgSize.toInt()) {
                        for (y in 0 until imgSize.toInt()) {
                            val out = network.forwardPass(doubleArrayOf(x / imgSize, y / imgSize))
                            val p = 0xff000000.toInt() + (out[0] * 256.0).toInt()
                            img.setRGB(x, y, p)
                        }
                    }

                    val file = File(dir, "epoch_$epoch.png")
                    ImageIO.write(img, "png", file)
                }
            }
        }
        val training = OnlineTraining(network, trainingSet, validationSet, epochs, callback, eta)
        training.perform()
    }
}