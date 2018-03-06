package main

import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.{CountVectorizerModel, IDFModel}
import org.apache.spark.sql.{SQLContext, SaveMode, SparkSession}
import nl.surfsara.warcutils.WarcInputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.jwat.warc.WarcRecord
import org.apache.spark.sql.functions._

import scala.io.Source
import utils.Records.{getPagerankRecord, getWarcRecord}

import scala.util.{Success, Try}

object GetFeatures {
  def main(args: Array[String]) {

    // Prepares test data
    val spark: SparkSession = SparkSession.builder.getOrCreate
    val sc = spark.sparkContext
    val docIDs = sc.textFile("all_ids").collect().toList
    
//    val path = "/data/private/clueweb12/Disk1/ClueWeb12_00/0012wb/0012wb-99.warc.gz"
//    val path = "/data/private/clueweb12/Disk1/ClueWeb12_00/*/*.warc.gz"
    val path = "/data/private/clueweb12/Disk[0-4]*/*/*/*.warc.gz"

//    val total_cores = sc.hadoopConfiguration.get("spark.executor.instances").toInt *
//      sc.hadoopConfiguration.get("spark.executor.cores").toInt
//    val partitions = total_cores * 3

//    val conf = sc.hadoopConfiguration.set("mapred.max.split.size", (3 * total_cores).toString())

    // Read all Warc records that have a TREC-ID.
    val warcRdd = sc.newAPIHadoopFile[LongWritable, WarcRecord, WarcInputFormat](path).
      filter(x => (null != x._2.getHeader("WARC-TREC-ID"))).
      map(x => getWarcRecord(x._2))

    // Create a dataframe with only the required Warc Files.
    var warcDf = spark.createDataFrame(warcRdd).toDF()
    var filteredWarcDf = warcDf.filter(warcDf("docID").isin(docIDs: _*))

    // Load the pre-trained tf and idf models
    val model = PipelineModel.read.load("pipeline-model.parquet")

    var output = model.transform(filteredWarcDf)

    val countTokens = udf { (words: Seq[String]) => words.length }

    output = output.withColumn("contentLength", countTokens(col("contentWords")))
    output = output.withColumn("titleLength", countTokens(col("titleWords")))

    // TODO add document length to each entry
    output.write.mode(SaveMode.Overwrite).save("all_ids.parquet")
    //select("docID", "url", "contentIDF", "titleIDF", "titleTF", "contentTF",
//      "contentLength", "titleLength").

    // Add pagerank scores TODO make seperate parquet file for this.
    val pagerank = "/data/private/clueweb12/pagerank/full/pagerank.docNameOrder.bz2"
    val pagerankRdd = sc.textFile(pagerank).map(x => getPagerankRecord(x))
    var pagerankDf = spark.createDataFrame(pagerankRdd).toDF()
    var filteredPagerankDf = pagerankDf.filter(pagerankDf("docID").isin(docIDs: _*))

    val joinedDf = filteredPagerankDf.join(output, "docID")

    // TODO add document length to each entry
    joinedDf.write.mode(SaveMode.Overwrite).save("spark ")
//      .select("docID", "url", "contentIDF", "titleIDF", "titleTF", "contentTF",
//        "contentLength", "titleLength", "pagerank").

//    joinedDf.select("docID", "url", "tf", "idf", "pagerank").show(5)
  }
}
