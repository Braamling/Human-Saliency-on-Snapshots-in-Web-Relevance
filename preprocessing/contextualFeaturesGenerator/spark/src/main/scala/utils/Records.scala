package utils

import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger
import org.jwat.warc.WarcRecord
import org.jsoup.Jsoup
import org.jsoup.nodes.{Document, Element}

object Records {
  @transient lazy val log = Logger.getLogger(getClass.getName)

  def getWarcRecord(in: WarcRecord, docIDs: Set[String]): warcSparkRecord = {
    // scalastyle:off
    println("Parsing document")
    // Filter out all documents that are to long.
    if(in.header.contentLength > 50000 && !docIDs.contains(in.getHeader("WARC-TREC-ID").toString)){
      println(in.header.contentLength)
      throw new IllegalStateException()
    }

    val doc: Document = Jsoup.parse(IOUtils.toString(in.getPayload().getInputStreamComplete()))
    warcSparkRecord(in.header.warcTargetUriStr, in.getHeader("WARC-TREC-ID").value,
      doc.text(), doc.title())
  }

  def getPagerankRecord(in: String): pagerankSparkRecord = {
    val splitted = in.split(" ")
    pagerankSparkRecord(splitted(0), splitted(1).toFloat)
  }

  case class warcSparkRecord(url: String, docID: String, content: String, title: String)
  case class pagerankSparkRecord(docID: String, pagerank: Float)
}
