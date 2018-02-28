package main

import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.{CountVectorizerModel, IDFModel}
import org.apache.spark.sql.{SQLContext, SaveMode, SparkSession}
import nl.surfsara.warcutils.WarcInputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.jwat.warc.WarcRecord

import scala.io.Source
import utils.Records.{getPagerankRecord, getWarcRecord}

import scala.util.{Success, Try}

object GetFeatures {
  def main(args: Array[String]) {

    // Prepares test data
    val spark: SparkSession = SparkSession.builder.getOrCreate
    val sc = spark.sparkContext
    val docIDs = sc.textFile("all_ids").collect().toList
    
    val path = "/data/private/clueweb12/Disk1/ClueWeb12_00/0012wb/0012wb-99.warc.gz"
//    val path = "/data/private/clueweb12/Disk1/ClueWeb12_00/*/"

    // Read all Warc records that have a TREC-ID.
    val warcRdd = sc.newAPIHadoopFile[LongWritable, WarcRecord, WarcInputFormat](path).
      filter(x => (null != x._2.getHeader("WARC-TREC-ID"))).
      map(x => Try(getWarcRecord(x._2))).collect{ case Success(df) => df }

    // Create a dataframe with only the required Warc Files.
    var warcDf = spark.createDataFrame(warcRdd).toDF()
    var filteredWarcDf = warcDf.filter(warcDf("docID").isin(docIDs:_*))

    // Load the pre-trained tf and idf models
    val model = PipelineModel.read.load("pipeline-model.parquet")

    val output = model.transform(filteredWarcDf)

    // TODO add document length to each entry
    output.select("docID", "url", "contentTF", "contentIDF", "titleTF", "contentTF").
      write.mode(SaveMode.Overwrite).save("all_ids.parquet")

    // Add pagerank scores TODO make seperate parquet file for this.
//    val pagerank = "/data/private/clueweb12/pagerank/full/pagerank.docNameOrder.bz2"
//    val pagerankRdd = sc.textFile(pagerank).map(x => getPagerankRecord(x))
//    var pagerankDf = spark.createDataFrame(pagerankRdd).toDF()
//    var filteredPagerankDf = pagerankDf.filter(pagerankDf("docID").isin(docIDs:_*))
//
//    val joinedDf = filteredPagerankDf.join(idf, "docID")
//
//    joinedDf.select("docID", "url", "tf", "idf", "pagerank").show(5)
  }
}
