from highlighter import Highlighter
from urllib.parse import urlsplit
from time import sleep
import requests
import os
import time
import subprocess

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

def urlGenerator(path):
    with open(path, 'r') as f:
        for line in f:
            query_id, doc_id, url, query = line.rstrip().split(" ", 3)
            print(query_id, doc_id, url, query)
            yield (query_id, doc_id, url, query)


def main():
    highlighter = Highlighter() 
    date = "20120202"
    path = "storage/TREC/trec_201-204"

    global_start = time.time()
    for i, (query_id, doc_id, url, query) in enumerate(urlGenerator(path)):
        try:
            start = time.time()
            url = getWebLink(url, date)
            createSnapshots(highlighter, url, query, query_id, doc_id)
        except Exception as e:
            print("failed to retrieve", doc_id, "from url", url)
        sleep(max(0, 30 - (time.time() - start)))
        print("Elapsed time", time.time() - global_start, "average time", (time.time() - global_start)/(i+1))

if __name__ == '__main__':
    main()
