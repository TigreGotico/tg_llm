from tg_llm.intents import LLMIntent

s = LLMIntent("google/flan-t5-large")
s.load_steps_from_file("demo", "demo.steps")

# demo.steps is a text file of natural language commands
# each line is a LLM prompt executed sequentially, the result is sent to the next prompt
#   Translate '{statement}' to {language} if it's English
#   What language is the following text? {text}

inputs = [
    {"statement": "Hello, how are you", "language": "French"},
    {"statement": "Hallo, wie geht's dir", "language": "French"}
]

print(s.run_steps("demo", inputs))
# ['French', 'German']
