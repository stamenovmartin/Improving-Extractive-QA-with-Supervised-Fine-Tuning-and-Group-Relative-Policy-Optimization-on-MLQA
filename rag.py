import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config
from embedder import Embedder
from vector_store import VectorStore


class RAGChatbot:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATOR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.GENERATOR_MODEL)
        self.model.to(config.DEVICE)

    def index_documents(self, documents):
        embeddings = self.embedder.encode(documents)
        self.vector_store.add(embeddings, documents)

    def retrieve(self, query, top_k=config.TOP_K):
        qemb = self.embedder.encode_single(query)
        return self.vector_store.search(qemb, top_k)

    def build_prompt(self, question, contexts):
        ctx_text = "\n".join(contexts)
        return f"Answer the question based on the context.\n\nContext: {ctx_text}\n\nQuestion: {question}"

    def generate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        ).to(config.DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_OUTPUT_LENGTH,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[0][1:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if outputs.scores:
            log_probs = torch.stack([torch.log_softmax(s, dim=-1) for s in outputs.scores])
            selected_log_probs = torch.gather(
                log_probs.squeeze(1), 1, gen_ids.unsqueeze(1)
            ).squeeze(1)
        else:
            selected_log_probs = torch.zeros(1)

        return answer, gen_ids, selected_log_probs

    def answer(self, question):
        retrieved = self.retrieve(question)
        contexts = [doc for doc, _ in retrieved]
        prompt = self.build_prompt(question, contexts)
        answer, _, _ = self.generate(prompt)
        return answer.strip()
