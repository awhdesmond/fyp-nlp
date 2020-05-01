from entailment.nlp import model


class EntailmentEngine:

    ENTAILMENT_INDEX = 0
    CONTRADICTION_INDEX = 1
    NEUTRAL_INDEX = 2

    def __init__(self, model: model.TextualEntailmentModel):
        self.model = model
        # self.textEntModel.createModel()

    def predict(self, hypothesis: str, premise: str):
        """Predict the entailment between hypothesis and premise

        Returns (prediction, score/confidence) or (None, None) if score
        is not above 0.5
        """
        entailment_result = self.model.predict(premise, hypothesis)

        if entailment_result[EntailmentEngine.ENTAILMENT_INDEX] > 0.5:
            return (
                EntailmentEngine.ENTAILMENT_INDEX,
                entailment_result[EntailmentEngine.ENTAILMENT_INDEX]
            )

        if entailment_result[EntailmentEngine.CONTRADICTION_INDEX] > 0.5:
            return (
                EntailmentEngine.CONTRADICTION_INDEX,
                entailment_result[EntailmentEngine.CONTRADICTION_INDEX]
            )

        if entailment_result[EntailmentEngine.NEUTRAL_INDEX] > 0.5:
            return (
                EntailmentEngine.NEUTRAL_INDEX,
                entailment_result[EntailmentEngine.NEUTRAL_INDEX]
            )

        return None, None
