import json
import random
import os

import config


def _load_squad_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pairs = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                ans = qa["answers"][0]["text"] if qa["answers"] else ""
                if ans:
                    pairs.append({
                        "question": qa["question"],
                        "context": ctx,
                        "answer": ans,
                    })
    return pairs


def load_mlqa_validation():
    # прво проба HuggingFace, ако не работи -> локален JSON
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/mlqa", "mlqa.en.en", split="validation")
        return [
            {"question": ex["question"], "context": ex["context"],
             "answer": ex["answers"]["text"][0]}
            for ex in ds if ex["answers"]["text"]
        ]
    except Exception:
        pass
    if os.path.exists(config.MLQA_DEV):
        return _load_squad_json(config.MLQA_DEV)
    raise FileNotFoundError(f"MLQA not found at {config.MLQA_DEV}")


def load_mlqa_test():
    return load_mlqa_test_lang("en", "en")


def load_mlqa_test_lang(context_lang="en", question_lang="en"):
    hf_config = f"mlqa.{context_lang}.{question_lang}"
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/mlqa", hf_config, split="test")
        return [
            {"question": ex["question"], "context": ex["context"],
             "answer": ex["answers"]["text"][0]}
            for ex in ds if ex["answers"]["text"]
        ]
    except Exception:
        pass
    local_filename = f"test-context-{context_lang}-question-{question_lang}.json"
    local_path = os.path.join(config.MLQA_DIR, "test", local_filename)
    if os.path.exists(local_path):
        return _load_squad_json(local_path)
    raise FileNotFoundError(
        f"MLQA test за {context_lang}-{question_lang} не е најден.\n"
        f"HuggingFace config '{hf_config}', local path '{local_path}'"
    )


def split_train_eval(data, train_ratio=config.TRAIN_RATIO, seed=config.DATA_SEED):
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    idx = int(len(shuffled) * train_ratio)
    return shuffled[:idx], shuffled[idx:]


def get_contexts(data):
    seen, out = set(), []
    for d in data:
        if d["context"] not in seen:
            out.append(d["context"])
            seen.add(d["context"])
    return out


def as_tuples(data):
    return [(d["question"], d["context"], d["answer"]) for d in data]
