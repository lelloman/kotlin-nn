package com.lelloman.kotlinnn

import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import javax.imageio.ImageIO

class SpiralLearningTest {

    private val random = Random()
    private val imgSize = 1024.0

    private val sample = { index: Int ->
        val j = index % 3

        val rIndex = random.nextInt(100)
        val r = rIndex / 100.0
        val tStep = (4) / 100.0

        val t = j * 4 + rIndex * tStep + (random.nextGaussian() + 1.0) * 0.25
        val x = 0.5 + r * Math.sin(t) / 2.0
        val y = 0.5 + r * Math.cos(t) / 2.0

        doubleArrayOf(x, y) to DoubleArray(3, { (it == j).toDouble() })
    }

    private val trainingSet = DataSet.Builder(1000)
            .add(sample)
            .build()

    private val validationSet = DataSet.Builder(100)
            .add(sample)
            .build()

    private fun createImg() = BufferedImage(imgSize.toInt(), imgSize.toInt(), BufferedImage.TYPE_4BYTE_ABGR).apply {
        val graphics = createGraphics()
        graphics.paint = Color.BLACK
        graphics.fillRect(0, 0, width, height)
    }

    @Test
    fun `spiral image`() {
        val img = createImg()

        val imgScale = (imgSize.toInt() - 1) * 0.8
        val imgBase = imgSize * 0.1
        trainingSet.forEach { inSample, outSample ->
            val x = (imgBase + inSample[0] * imgScale).toInt()
            val y = (imgBase + inSample[1] * imgScale).toInt()
            println("coordinates $x $y")

            val r = (255 * outSample[0]).toInt().shl(16)
            val g = (255 * outSample[1]).toInt().shl(8)
            val b = (255 * outSample[2]).toInt()
            val p = 0xff000000 + r + g + b
            img.setRGB(x, y, p.toInt())
        }

        val dir = File("src/test/resources", "spiral")
        if (dir.exists()) {
            dir.deleteRecursively()
        }
        dir.mkdir()
        val file = File(dir, "spiral.png")
        ImageIO.write(img, "png", file)
    }
}