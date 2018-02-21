from highlighter import Highlighter
from urllib.parse import urlsplit
from time import sleep
import requests
import os
import time
import subprocess
import argparse
import random

"""
Create a set of three snapshots from a url.

highligher: Highligher object containing a selenium driver
url: url to use for snapshots
query: string containing the query to highlight
query_id: query id used for storing snapshots
doc_id: document id to use for storing snapshots
"""
def create_snapshots(highlighter, url, query, query_id, doc_id):
    highlighter.prepare(url, wayback=True)
    highlighter.store_snapshot("storage/snapshots/{}.png".format(doc_id))
    highlighter.set_highlights(query)
    highlighter.store_snapshot("storage/highlights/{}-{}.png".format(query_id, doc_id))
    highlighter.remove_content()
    highlighter.store_snapshot("storage/masks/{}-{}.png".format(query_id, doc_id), grayscale=True)
    highlighter.close()

"""
Convert a url and date to a wayback url.
The closed available wayback machine snapshot url is returned.
If the full path is not available, the url will be stripped to just the domain root.
If the domain root is not available, the url is returned as is.

url: url check at wayback
date: YYYYMMDD string that should be used for retrieving the snapshot
"""
def get_web_link(url, date):
    # Try to get the actual link
    avail, waybackUrl = check_wayback_avail(url, date)
    if avail:
        print(waybackUrl)
        return waybackUrl

    domain = "{0.scheme}://{0.netloc}/".format(urlsplit(url))

    # Attempt to get the root domain instead.
    avail, waybackUrl = check_wayback_avail(domain, date)
    if avail:
        print(waybackUrl)
        return waybackUrl

    print(url)
    return url


"""
Check whether a url/date combination is available in the wayback machine.

url: url check at wayback
date: YYYYMMDD string that should be used for retrieving the snapshot
""" 
def check_wayback_avail(url, date):
    waybackUrl = "http://archive.org/wayback/available?url={}&timestamp={}"

    url = waybackUrl.format(url, date)

    json = requests.get(url).json()

    snapshots = json["archived_snapshots"]

    if "closest" in snapshots:
        print("available:", snapshots["closest"]["available"])
        return True, snapshots["closest"]["url"]

    return False, ""

""" 
Create a dict with all query id's and their corresponding queries.
"""
def make_queries_dict():
    queries = {}
    with open("storage/TREC/queries", 'r') as f:
        for line in f:
            query_id, query = line.rstrip().split(":", 1)
            queries[query_id] = query

    return queries

"""
Yield doc_id, url pairs for the configured query.
"""
def document_generator():
    with open("storage/TREC/{}_docs".format(FLAGS.query), 'r') as fd:
        with open("storage/TREC/{}_urls".format(FLAGS.query), 'r') as fu:
            for doc_id, url in zip(fd, fu):
                yield doc_id.strip(), url.strip()


def main():
    highlighter = Highlighter() 
    queries = make_queries_dict()

    query = queries[FLAGS.query]

    global_start = time.time()
    for i, (doc_id, url) in enumerate(document_generator()):
        try:
            start = time.time()
            url = get_web_link(url, FLAGS.date)
            create_snapshots(highlighter, url, query, FLAGS.query, doc_id)
        except Exception as e:
            highlighter.close(driver=False)
            print(e)
            print("failed to retrieve", doc_id, "from url", url)
        sleep(max(0, random.randint(60, 75) - (time.time() - start)))
        print("Elapsed time", time.time() - global_start, "average time", (time.time() - global_start)/(i+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--query', type=str, default='207',
                        help='The query id to retrieve.')
    parser.add_argument('--date', type=str, default='20120202',
                        help='The date (YYYYMMDD) to aim for while scraping.')


    FLAGS, unparsed = parser.parse_known_args()

    main()
