def createErrorResponse(errMsg):
    return {
        "error": "Invalid Request: %s" % errMsg
    }

def createJSONResponse(data):
    return {
        "data": data # python dict
    }