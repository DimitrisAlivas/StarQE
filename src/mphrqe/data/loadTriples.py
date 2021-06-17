"""Load the train, validation, and test sets into the knowledge base."""

import copy
import logging
import ssl
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore

from .config import sparql_endpoint_address as default_sparql_endpoint
from .config import sparql_endpoint_options as default_spaqrl_endpoint_options

__all__ = [
    "load_data",
]

logger = logging.getLogger(__name__)


# TODO: Remove commented out code?

def load_data(
    source_directory: Path,
    sparql_endpoint: Optional[str] = None,
    sparql_endpoint_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute the insert queries in the source directory to populate the triple store.
    """

    sparql_endpoint = sparql_endpoint or default_sparql_endpoint
    # we make a deepcopy because we will modify the headers and do not want to modify the config itself
    sparql_endpoint_options = copy.deepcopy(sparql_endpoint_options or default_spaqrl_endpoint_options)
    assert isinstance(sparql_endpoint_options, dict)

    headers = sparql_endpoint_options.get('headers') or {}
    assert isinstance(headers, dict)
    assert headers.get('Content-Type') is None
    headers['Content-Type'] = 'application/sparql-query'
    sparql_endpoint_options['headers'] = headers

    store = SPARQLUpdateStore(sparql_endpoint, update_endpoint=sparql_endpoint, method="POST", autocommit=True, **sparql_endpoint_options)
    for query_file_path in source_directory.rglob("*.sparql"):
        query = query_file_path.read_text()
        # run_query(sparql_endpoint, query)
        store.update(query)


def run_query(sparql_endpoint, sparql_query, fmt=None):
    """
    Using this implementation from the anzograph documentation because a normal POST request seems to fail.
    """

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # create HTTP connection to SPARQL endpoint
    conn = HTTPSConnection(sparql_endpoint, context=ctx, timeout=100)  # may throw HTTPConnection exception
    # urlencode query for sending
    # docbody = urlencode({'query': sparql_query})
    # request result in json
    hdrs = {
        'Host': 'Anon',
        'Accept': 'application/sparql-results+csv',
        'User-Agent': 'someAgent',
        'Content-Type': 'application/sparql-query',
    }

    # send post request
    conn.request(method='POST', url='/sparql', body=sparql_query, headers=hdrs)  # may throw exception

    # read response
    resp = conn.getresponse()
    if resp.status != 200:
        errmsg = resp.read()
        conn.close()
        raise Exception('Query Error', errmsg)  # query processing errors - syntax errors, etc.

    # content-type header, and actual response data
    result = resp.read().lstrip()
    conn.close()

    logger.info(result)
    # check response content-type header
    # if raw or ctype.find('json') < 0:
    #    return result      # not a SELECT?

    #    # convert result in JSON string into python dict
    #    return json.loads(result)


def run_query_OLD(sparql_endpoint, sparql_query, fmt=None):
    """
    Using this implementation from the anzograph documentation because a normal POST request seems to fail.
    """
    # create HTTP connection to SPARQL endpoint
    conn = HTTPConnection(sparql_endpoint, timeout=100)  # may throw HTTPConnection exception
    # urlencode query for sending
    docbody = urlencode({'query': sparql_query})
    # request result in json
    hdrs = {'Accept': 'application/sparql-results+json',
            'Content-type': 'application/x-www-form-urlencoded'}
    raw = False
    if fmt is not None:
        raw = True
        if fmt in ('xml', 'XML'):
            hdrs['Accept'] = 'application/sparql-results+xml'
        elif fmt in ('csv', 'CSV'):
            hdrs['Accept'] = 'text/csv, application/sparql-results+csv'

    # send post request
    conn.request('POST', '/sparql', docbody, hdrs)  # may throw exception

    # read response
    resp = conn.getresponse()
    if 200 != resp.status:
        errmsg = resp.read()
        conn.close()
        raise Exception('Query Error', errmsg)  # query processing errors - syntax errors, etc.

    # content-type header, and actual response data
    ctype = resp.getheader('content-type', 'text/html').lower()
    result = resp.read().lstrip()
    conn.close()

    # check response content-type header
    if raw or ctype.find('json') < 0:
        return result  # not a SELECT?

#    # convert result in JSON string into python dict
#    return json.loads(result)
