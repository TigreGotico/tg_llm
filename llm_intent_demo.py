from os.path import dirname

from tg_llm.intents import LLMIntent, ZeroShotIntentParser, Sequences

# Intent classifier
qa = LLMIntent()
qa.load_steps(["answer the question, if you don't know the answer say 'I don't know', question: {text}"])

p = ZeroShotIntentParser(fallback=qa)


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


wiki = LLMIntent()
wiki.load_steps(["Answer the question '{text}' given this text from wikipedia: {wiki_summary}"])

# use LLMs to pre-process the input
kwrod = LLMIntent()
kwrod.load_steps(["Extract the subject for a wikipedia search from the question: '{text}'"])

# accepts LLMIntent, callable, or tuple (LLMIntent, callable)
keywords = {
    "wiki_summary": (kwrod, get_wiki)  # (LLMIntent, handler) -> LLM provides input for handler
}
p.register_intent("knowledge", wiki, keywords)

ans = p.execute("when was Isaac newton born")
print(ans)  # 25 December 1642
ans = p.execute("what books did newton write")
print(ans)  # Newton's Laws
ans = p.execute("what did newton invent")
print(ans)  # law of universal gravitation
ans = p.execute("when did newton die")
print(ans)  # Newton (1560–1610) was an English mathematician and physicist.
ans = p.execute("what was newton's profession")
print(ans)  # physicist


# standalone usage of intents
# llm = Sequences("google/flan-t5-large")
llm = p.llm

s = LLMIntent()
s.load_model(llm)
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

