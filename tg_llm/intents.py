from txtai.pipeline import Sequences, Labels
from txtai.workflow import Workflow, TemplateTask


class LLMIntent:
    def __init__(self, name, model=None):
        self.name = name
        self.sequences = Sequences(model)
        self.steps = {}

    def load_steps_from_file(self, file_path):
        with open(file_path) as f:
            steps = [s for s in f.read().split("\n")
                     if s.strip() and not s.startswith("#")]

        self.load_steps(self.name, steps)

    def load_steps(self, steps):
        self.steps[self.name] = Workflow([
            TemplateTask(
                template=s,
                action=self.sequences
            ) for s in steps
        ])

    def run_steps(self, inputs: list):
        return list(self.steps[self.name](inputs))


class ZeroShotIntentParser:
    def __init__(self, labels=None, model="facebook/bart-large-mnli", fallback=None):
        """Alternate models can be used via passing the model
        eg, "roberta-large-mnli"
         """
        # Create labels model
        self.labels = labels or []
        self.parser = Labels(model)
        self.handlers = {}
        self.fallback = fallback

    def register_intent(self, name, handler: LLMIntent):
        if name not in self.labels:
            self.labels.append(name)
        if name in self.handlers:
            print(f"Warning - replacing previous handled for {name}")
        self.handlers[name] = handler

    def classify(self, utterance):
        if not self.labels:
            return None, 0.0
        preds = self.parser(utterance, self.labels)
        label = self.labels[preds[0][0]]
        conf = preds[0][1]
        return label, conf

    def execute(self, utterance):
        label, conf = self.classify(utterance)
        if label in self.handlers:
            try:
                self.handlers[label].run_steps([utterance])[0]
            except Exception as e:
                print("ERROR - failed to handle intent")
                print(e)
        else:
            print("ERROR - unregistered intent")
        if self.fallback:
            return self.fallback.run_steps([utterance])[0]


