from google.genai.models import Models

counters = {"embedding_calls": 0, "generation_calls": 0}
_instrumented = False


def reset_counters():
    counters["embedding_calls"] = 0
    counters["generation_calls"] = 0


def instrument():
    global _instrumented
    if _instrumented:
        return
    _instrumented = True

    original_embed = Models.embed_content
    original_generate = Models.generate_content

    def embed_content(self, *args, **kwargs):
        counters["embedding_calls"] += 1
        print(f"[metrics] embedding API call #{counters['embedding_calls']}")
        return original_embed(self, *args, **kwargs)

    def generate_content(self, *args, **kwargs):
        counters["generation_calls"] += 1
        print(f"[metrics] generation API call #{counters['generation_calls']}")
        return original_generate(self, *args, **kwargs)

    Models.embed_content = embed_content
    Models.generate_content = generate_content
