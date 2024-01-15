from txtai.pipeline import Sequences, Labels
from txtai.workflow import Workflow, TemplateTask


class LLMIntent:
    def __init__(self):
        self.steps = []
        self.sequences = None

    def load_model(self, model):
        if isinstance(model, str):
            self.sequences = Sequences(model)
        else:
            self.sequences = model

    def load_steps_from_file(self, file_path):
        with open(file_path) as f:
            steps = [s for s in f.read().split("\n")
                     if s.strip() and not s.startswith("#")]
        self.load_steps(steps)

    def load_steps(self, steps):
        self.steps = steps

    @property
    def pipeline(self):
        return Workflow([
            TemplateTask(
                template=s,
                action=self.sequences
            ) for s in self.steps
        ])

    def run_steps(self, inputs: list):
        return list(self.pipeline(inputs))


class ZeroShotIntentParser:
    def __init__(self, labels=None,
                 model="facebook/bart-large-mnli",
                 model2="google/flan-t5-large",
                 fallback: LLMIntent=None):
        """Alternate models can be used via passing the model
        eg, "roberta-large-mnli"
         """
        # Create labels model
        self.labels = labels or []
        self.labeler = Labels(model)
        self.llm = Sequences(model2)
        self.handlers = {}
        self.keywords = {}
        if fallback:
            fallback.load_model(self.llm)
        self.fallback = fallback

    def register_intent(self, name,
                        handler: LLMIntent,
                        keyword_extractor: dict):
        handler.load_model(self.llm)
        if name not in self.labels:
            self.labels.append(name)
        if keyword_extractor and name not in self.keywords:
            self.keywords[name] = keyword_extractor
        if name in self.handlers:
            print(f"Warning - replacing previous handled for {name}")
        self.handlers[name] = handler

    def classify(self, utterance):
        if not self.labels:
            return None, 0.0
        preds = self.labeler(utterance, self.labels)
        label = self.labels[preds[0][0]]
        conf = preds[0][1]
        return label, conf

    def execute(self, utterance):
        label, conf = self.classify(utterance)
        if label in self.handlers:
            inputs = [{"text": utterance}]

            # extract any input keywords
            if label in self.keywords:
                keywords = {}
                xtractor = self.keywords[label]
                for l2, handler in xtractor.items():
                    if isinstance(handler, tuple):
                        handler[0].load_model(self.llm)
                        arg = handler[0].run_steps(inputs)[0]
                        val = handler[1](arg)
                    elif isinstance(handler, LLMIntent):
                        handler.load_model(self.llm)
                        val = handler.run_steps(utterance)[0]
                    else:
                        val = handler(utterance)
                    keywords[l2] = val

                inputs = [{"text": utterance, **keywords}]

            return self.handlers[label].run_steps(inputs)[0]
        elif label is not None:
            print(f"ERROR - unregistered intent: {label}")
        if self.fallback:
            print("using catch all fallback handler")
            return self.fallback.run_steps([utterance])[0]


