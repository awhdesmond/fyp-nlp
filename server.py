import pydash

import logging
logging.basicConfig(level="INFO")

from bottle import Bottle, run, request, response
from nlpengine import NLPEngine

app = Bottle()
nlpEngine = NLPEngine()

def createErrorResponse(errMsg):
    return {
        "error": "Invalid Request: %s" % errMsg
    }

def createJSONResponse(data):
    return {
        "data": data # python dict
    }

@app.get('/')
def baseHandler():
    return "NLP Python Engine"

@app.post('/api/analyse')
def analyseHandler():
    body = dict(request.json)

    if len(pydash.intersection(["text", "relatedArticles"], body.keys())) != 2:
        return createErrorResponse("Missing parameters.")

    text            = body["text"]
    relatedArticles = body["relatedArticles"]
    analysis = nlpEngine.handleAnalyseQuery(text, relatedArticles)
    
    if analysis == None:
        result = {
            "queryHasClaims": False,
            "articlesWithEvidence": []
        }
        return createJSONResponse(result)
    result = {
        "queryHasClaims": True,
        "articlesWithEvidence": analysis
    }
    return createJSONResponse(result)

logging.info(f"Server listening on port: {8080}")
run(app, host='0.0.0.0', port=8080)
