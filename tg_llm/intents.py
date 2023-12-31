from txtai.pipeline import Sequences
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


