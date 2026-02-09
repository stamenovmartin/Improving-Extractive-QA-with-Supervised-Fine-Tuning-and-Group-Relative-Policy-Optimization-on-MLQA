import torch
from tqdm import tqdm

import config


class MarianTranslator:
    """Превод со Helsinki-NLP/opus-mt моделите."""

    def __init__(self, source_lang="en", target_lang="mk"):
        from transformers import MarianMTModel, MarianTokenizer

        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        print(f"Loading translation model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(config.DEVICE)
        self.model.eval()

    @torch.no_grad()
    def translate_batch(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(config.DEVICE)
        outputs = self.model.generate(**inputs, max_length=512)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

    def translate_dataset(self, data, batch_size=16):
        """Преведи цел dataset (context, question, answer)."""
        contexts = [d["context"] for d in data]
        questions = [d["question"] for d in data]
        answers = [d["answer"] for d in data]

        print(f"  Translating {len(data)} contexts...")
        tr_contexts = self._batched(contexts, batch_size)
        print(f"  Translating {len(data)} questions...")
        tr_questions = self._batched(questions, batch_size)
        print(f"  Translating {len(data)} answers...")
        tr_answers = self._batched(answers, batch_size)

        return [
            {"context": c, "question": q, "answer": a}
            for c, q, a in zip(tr_contexts, tr_questions, tr_answers)
        ]

    def _batched(self, texts, batch_size):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.translate_batch(batch))
        return results


def get_translator(target_lang="mk"):
    return MarianTranslator(source_lang="en", target_lang=target_lang)
