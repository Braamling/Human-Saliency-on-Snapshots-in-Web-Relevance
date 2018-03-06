package main

import org.apache.spark.ml.feature._
import org.apache.spark.sql.{SaveMode, SparkSession}
import nl.surfsara.warcutils.WarcInputFormat
import org.apache.hadoop.io.LongWritable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.jwat.warc.WarcRecord
import utils.Records.getWarcRecord
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.functions._

import scala.util.{Success, Try}

object FitModels {
  def main(args: Array[String]) {


    // Prepares test data
    val spark: SparkSession = SparkSession.builder.getOrCreate
    val sc = spark.sparkContext

//    val path = "/data/private/clueweb12/Disk1/ClueWeb12_00/0012wb/0012wb-99.warc.gz"
    val path = "/data/private/clueweb12/Disk[0-4]*/*/*/*.warc.gz"

    // Read all Warc records that have a TREC-ID.
    val warcRdd = sc.newAPIHadoopFile[LongWritable, WarcRecord, WarcInputFormat](path).
      filter(x => (null != x._2.getHeader("WARC-TREC-ID"))).
      map(x => Try(getWarcRecord(x._2))).collect{ case Success(df) => df }

    val warcDf = spark.createDataFrame(warcRdd).toDF()

    // Use a tokenizer on the warc content.
    val contentTokenizer = new Tokenizer().setInputCol("content").setOutputCol("contentWords")
    val titleTokenizer = new Tokenizer().setInputCol("title").setOutputCol("titleWords")

    // Fit a tf model that can be reused for lookup
    val contentVectorizerModel = new CountVectorizer()
      .setInputCol("contentWords").setOutputCol("contentTF")

    // Fit a tf model that can be reused for lookup
    val titleVectorizerModel = new CountVectorizer()
      .setInputCol("titleWords").setOutputCol("titleTF")

    // Use the term frequencies to calculate a idf model
    val ContentIdfModel = new IDF().setInputCol("contentTF").setOutputCol("contentIDF")
    val TitleIdfModel = new IDF().setInputCol("titleTF").setOutputCol("titleIDF")


    val pipeline = new Pipeline()
      .setStages(Array(contentTokenizer, titleTokenizer, contentVectorizerModel, titleVectorizerModel, ContentIdfModel, TitleIdfModel))

    val model = pipeline.fit(warcDf)
//    val model = PipelineModel.read.load("pipeline-model.parquet")

    var output = model.transform(warcDf)

    model.write.overwrite.save("pipeline-model.parquet")

    // Calculate meanDocumentLength
    output = output.withColumn("contentSize", size(output("contentWords")))
    output = output.withColumn("titleSize", size(output("titleWords")))
//    output.select(Seq("contentSize", "titleSize").map(mean(_)): _*).show()
    val meanDocumentLength = output.select(Seq("contentSize", "titleSize").map(mean(_)): _*)

    meanDocumentLength.withColumnRenamed("avg(contentSize)", "avgContentSize").
      withColumnRenamed("avg(titleSize)", "avgTitleSize").write.mode(SaveMode.Overwrite).save("meanDocument.parquet")

    // Store the tf and idf model for later usage.
  }
}