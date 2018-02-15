from highlighter import Highlighter
from urllib.parse import urlsplit
from time import sleep
import requests
import os
import time
import subprocess
import argparse

def createSnapshots(highlighter, webpage, query, query_id, name):
    highlighter.prepare(webpage, wayback=True)
    highlighter.storeSnapshot("storage/snapshots/{}.png".format(name))
    highlighter.setHighlights(query)
    highlighter.storeSnapshot("storage/highlights/{}-{}.png".format(query_id, name))
    highlighter.removeContent()
    highlighter.storeSnapshot("storage/masks/{}-{}.png".format(query_id, name), grayscale=True)
    highlighter.close()

def getWebLink(url, date):
    # Try to get the actual link
    avail, waybackUrl = checkWaybackAvail(url, date)
    if avail:
        print(waybackUrl)
        return waybackUrl

    domain = "{0.scheme}://{0.netloc}/".format(urlsplit(url))

    # Attempt to get the root domain instead.
    avail, waybackUrl = checkWaybackAvail(domain, date)
    if avail:
        print(waybackUrl)
        return waybackUrl

    print(url)
    return url

def checkWaybackAvail(url, date):
    waybackUrl = "http://archive.org/wayback/available?url={}&timestamp={}"

    url = waybackUrl.format(url, date)

    json = requests.get(url).json()

    snapshots = json["archived_snapshots"]

    if "closest" in snapshots:
        print("available:", snapshots["closest"]["available"])
        return True, snapshots["closest"]["url"]

    return False, ""

def make_queries_dict():
    queries = {}
    with open("storage/prep/queries", 'r') as f:
        for line in f:
            query_id, query = line.rstrip().split(":", 1)
            queries[query_id] = query

    return queries

def documentGenerator():
    with open("storage/prep/{}_docs".format(FLAGS.query), 'r') as fd:
        with open("storage/prep/{}_urls".format(FLAGS.query), 'r') as fu:
            for doc_id, url in zip(fd, fu):
                yield doc_id.strip(), url.strip()


def main():
    highlighter = Highlighter() 
    date = "20120202"
    
    queries = make_queries_dict()

    query = queries[FLAGS.query]

    global_start = time.time()
    for i, (doc_id, url) in enumerate(documentGenerator()):
        try:
            start = time.time()
            url = getWebLink(url, date)
            createSnapshots(highlighter, url, query, FLAGS.query, doc_id)
        except Exception as e:
            print("failed to retrieve", doc_id, "from url", url)
        sleep(max(0, 90 - (time.time() - start)))
        print("Elapsed time", time.time() - global_start, "average time", (time.time() - global_start)/(i+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='storage/TREC/',
                        help='The location of the trec files')
    parser.add_argument('--query', type=str, default='207',
                        help='The location of the trec file with urls, query and doc_ids.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
