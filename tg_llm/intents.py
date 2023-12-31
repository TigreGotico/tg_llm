from txtai.pipeline import Sequences, Labels
from txtai.workflow import Workflow, TemplateTask


class LLMIntent:
    def __init__(self, model):
        # Create sequences pipeline
        self.sequences = Sequences(model)
        self.steps = {}

    def load_steps_from_file(self, name, file_path):
        with open(file_path) as f:
            steps = [s for s in f.read().split("\n")
                     if s.strip() and not s.startswith("#")]

        self.load_steps(name, steps)

    def load_steps(self, name, steps):
        self.steps[name] = Workflow([
            TemplateTask(
                template=s,
                action=self.sequences
            ) for s in steps
        ])

    def run_steps(self, name: str, inputs: list):
        return list(self.steps[name](inputs))


class ZeroShotIntentParser:
    def __init__(self, labels=None, model="facebook/bart-large-mnli"):
        """Alternate models can be used via passing the model
        eg, "roberta-large-mnli"
         """
        # Create labels model
        self.labels = labels or []
        self.parser = Labels(model)

    def register_intent(self, name):
        if name not in self.labels:
            self.labels.append(name)

    def classify(self, utterance):
        preds = self.parser(utterance, self.labels)
        label = self.labels[preds[0][0]]
        conf = preds[0][1]
        return label, conf

