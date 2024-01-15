Library to build intents powered by LLMs

By putting an LLM behind an intent parser anyone can build a natural language interface

Solve problems with natural language, program intents by adding a sequence of steps to a `.steps` file

```text
Translate '{statement}' to {language} if it's English
What language is the following text? {text}
```

## Usage

### LLM Intents

Handle intents with LLMs

```python
from os.path import dirname

from tg_llm.intents import LLMIntent, Sequences

llm = Sequences("google/flan-t5-large")

s = LLMIntent() 
s.load_model(llm) # choose a LLM
s.load_steps(["extract person names from {text}"])
print(s.run_steps(["My name is Casimiro"]))
# ['Casimiro']


s.load_steps_from_file(f"{dirname(__file__)}/demo.steps")

# demo.steps is a text file of natural language commands
# each line is a LLM prompt executed sequentially, the result is sent to the next prompt as {text}
#   Translate '{statement}' to {language} if it's English
#   What language is the following text? {text}

inputs = [
    {"statement": "Hello, how are you", "language": "French"},
    {"statement": "Hallo, wie geht's dir", "language": "French"}
]

print(s.run_steps(inputs))
# ['French', 'German']
```

### Intent Parsing

Tag intents without any training data

```python
from tg_llm.intents import ZeroShotIntentParser

p = ZeroShotIntentParser(labels=["time", "weather", "math",
                                 "person_info", "joke", "get_data",
                                 "unknown"])

data = ["what time is it",
        "is it going to snow",
        "who was isaac newton",
        "what is the speed of light",
        "10+5",
        "tell me a joke"]

for utt in data:
    label, conf = p.classify(utt)
    print(utt, "-", label, conf)
    # what time is it - time 0.6037998795509338
    # is it going to snow - weather 0.4515065848827362
    # who was isaac newton - person_info 0.3793475925922394
    # what is the speed of light - unknown 0.6040631532669067
    # 10+5 - math 0.3564682900905609
    # tell me a joke - joke 0.6909916996955872

```


handle intents

```python
qa = LLMIntent("fallback")
qa.load_steps(["answer the question, if you don't know the answer say 'I don't know', question: {text}"])


p = ZeroShotIntentParser(fallback=qa)
ans = p.execute("in what language is this text")
print(ans)
# English
```


extract keywords

```python

p = ZeroShotIntentParser()


def get_wiki(query: str):
    import requests
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    res = requests.get(url).json()["query"]["search"]
    for r in res:
        pid = str(r["pageid"])
        results_url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts|pageimages&exintro&explaintext&redirects=1&pageids=" + pid
        r = requests.get(results_url).json()
        return r['query']['pages'][pid]['extract'].split("\n")[0]
    ans = f"I don't know anything about {query}"
    return ans



# use LLMs to pre-process the input
kwrod = LLMIntent()
kwrod.load_steps(["Extract the subject for a wikipedia search from the question: '{text}'"])
# accepts LLMIntent, callable, or tuple (LLMIntent, callable)
keywords = {
    "wiki_summary": (kwrod, get_wiki)  # (LLMIntent, handler) -> LLM provides input for handler
}

wiki = LLMIntent()
wiki.load_steps(["Answer the question '{text}' given this text from wikipedia: {wiki_summary}"])
p.register_intent("knowledge", wiki, keywords)

ans = p.execute("when was Isaac newton born")
print(ans)  # 25 December 1642
ans = p.execute("what did newton invent")
print(ans)  # law of universal gravitation
ans = p.execute("what was newton's profession")
print(ans)  # physicist

```