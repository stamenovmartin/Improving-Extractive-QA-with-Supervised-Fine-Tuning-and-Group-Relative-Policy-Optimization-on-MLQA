import os
from transformers import AutoModelForSeq2SeqLM

import config
from data import load_mlqa_validation, get_contexts
from rag import RAGChatbot


def main():
    data = load_mlqa_validation()
    contexts = get_contexts(data)

    chatbot = RAGChatbot()

    for model_path in [config.GRPO_BEST_DIR, config.SFT_MODEL_DIR]:
        if os.path.exists(model_path):
            chatbot.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(config.DEVICE)
            print(f"Loaded model from {model_path}/")
            break
    else:
        print("Using base flan-t5-small model")

    chatbot.index_documents(contexts)

    print("RAG Chatbot Ready. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "quit":
            break

        if not question:
            continue

        answer = chatbot.answer(question)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
