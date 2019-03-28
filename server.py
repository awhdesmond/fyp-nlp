import pydash
import utils

from bottle import Bottle, run, request, response
from nlpengine import NLPEngine

app = Bottle()
nlpEngine = NLPEngine()

@app.get('/')
def baseHandler():
    return "NLP Python Engine"

@app.post('/api/analyse')
def analyseHandler():
    body = dict(request.json)

    if len(pydash.intersection(["text", "relatedArticles"], body.keys())) != 2:
        return utils.createErrorResponse("Missing parameters.")

    text            = body["text"]
    relatedArticles = body["relatedArticles"]
    analysis = nlpEngine.handleAnalyseQuery(text, relatedArticles)
    
    if analysis == None:
        result = {
            "queryHasClaims": False
            "articlesWithEvidence": []
        }
        return utils.createJSONResponse(result)
    else:
        result = {
            "queryHasClaims": True
            "articlesWithEvidence": analysis
        }
        return utils.createJSONResponse(result)

run(app, host='0.0.0.0', port=8080)
