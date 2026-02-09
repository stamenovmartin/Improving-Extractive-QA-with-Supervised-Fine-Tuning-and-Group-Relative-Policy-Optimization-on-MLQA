# RAG + SFT + GRPO за Question Answering

Проект по предметот Агентски базирани системи (АБС).

Идејата е да се примени reinforcement learning (конкретно GRPO) врз мал јазичен модел (`flan-t5-small`, 80M параметри) за задачата extractive QA, и да се спореди со обичен supervised fine-tuning. Датасетот е MLQA — мултијазичен QA бенчмарк со 7 јазици.

## Како работи

Пајплајнот има две фази на тренирање и опционален RAG дел за инференца:

```
flan-t5-small
     │
     ▼
  SFT (3 епохи, cross-entropy)
     │
     ▼
  GRPO (3 епохи, RL со PPO-clip + KL penalty)
     │
     ▼
  GRPO-best модел → RAG четбот / евалуација
```

**SFT** — стандардно fine-tuning на (контекст, прашање) → одговор парови. AdamW, cosine schedule, gradient clipping.

**GRPO** — за секое прашање се генерираат K=4 кандидат-одговори, се оценуваат со reward функција (`0.7*F1 + 0.3*EM - length_penalty`), па се прави z-score нормализација и PPO-style update со KL регуларизација кон замрзнат референтен модел. Нема потреба од critic мрежа или preference податоци — тоа е главната предност на GRPO наспроти PPO или DPO.

**RAG** — при инференца, прашањето се embed-ира со Sentence-BERT (`all-MiniLM-L6-v2`), се земаат top-3 контексти од FAISS индекс, и се пуштаат низ моделот.

## Резултати

### Англиски (MLQA test, n=500)

| Модел | F1 | EM | Просечна должина |
|---|---|---|---|
| Base | 0.624 | 0.508 | 3.4 |
| SFT | 0.633 | 0.516 | 2.8 |
| **GRPO-best** | **0.668** | **0.548** | **3.2** |
| GRPO-final | 0.651 | 0.534 | 3.0 |

GRPO-best е checkpoint од step 200 (од вкупно 774). После тоа F1 паѓа — типична работа за RL fine-tuning, моделот почнува да overfit-ува на reward сигналот.

### Мултијазичен zero-shot трансфер (GRPO-best, n=100)

| Јазик | F1 | EM |
|---|---|---|
| en-en | 0.668 | 0.548 |
| es-es | 0.425 | 0.240 |
| de-de | 0.290 | 0.170 |
| mk-mk | 0.039 | 0.020 |

Моделот е тренирано само на англиски. Шпанскиот се држи некако (лексичко преклопување), германскиот помалку, а македонскиот е практично нула — flan-t5-small речиси и нема македонски податоци во pretrain корпусот, плус токенајзерот не го покрива кирилското писмо добро. За mk-mk користев MarianMT превод од англиски, што дополнително внесува грешки.

## Датасет

MLQA нема официјален train split, па го користам validation split-от (1,148 парови) поделен 90/10 за тренинг/евалуација (seed=42). Test split-от (11,590 парови) е исклучиво за финална евалуација — тренингот никогаш не гледа test примери.

## Структура

```
config.py           хиперпараметри и патеки
data.py             вчитување MLQA, train/eval split
rewards.py          F1, EM, композитна награда, z-score
sft_trainer.py      SFT тренер
grpo_trainer.py     GRPO тренер (семплирање, PPO clip, KL)
train.py            главен скрипт (SFT → GRPO)
eval.py             евалуација на сите модели + мултијазично
translate.py        MarianMT превод (en→mk)
embedder.py         SentenceTransformer wrapper
vector_store.py     FAISS vector store
rag.py              RAG логика (retrieve + generate)
chatbot.py          интерактивен CLI четбот
```

Checkpoints се зачувуваат во `models/sft/`, `models/grpo_best/`, `models/grpo_final/`. Графиконите во `plots/`, логовите во `logs/`.

## Употреба

### Зависности

```
torch, transformers, sentence-transformers, faiss-cpu
datasets, matplotlib, numpy, tqdm, sentencepiece
```

### Тренирање

```bash
python train.py              # SFT + GRPO целосно
python train.py --grpo-only  # само GRPO (треба да постои SFT checkpoint)
```

### Евалуација

```bash
python eval.py               # сите модели + мултијазично + mk
python eval.py --samples 200 # помалку примери за побрзо тестирање
```

### Четбот

```bash
python chatbot.py            # quit за излез
```

## Хиперпараметри

| Параметар | SFT | GRPO |
|---|---|---|
| Learning rate | 5e-5 | 1e-5 |
| Batch size | 8 | 4 |
| Epochs | 3 | 3 |
| Max grad norm | 1.0 | 1.0 |
| Warmup ratio | 0.1 | 0.1 |
| Group size (K) | — | 4 |
| Clip ε | — | 0.2 |
| KL β | — | 0.04 |
| Temperature | — | 0.8 |
| Top-p | — | 0.9 |

Останато: max input=512, max output=64, reward weights 0.7/0.3 (F1/EM), length penalty λ=0.001, eval на секои 200 чекори, early stopping patience=3.

AMP е исклучен затоа што flan-t5-small дава NaN загуби со mixed precision.

## Познати проблеми и ограничувања

- Моделот е мал (80M) и не може да се мери со поголеми модели — апсолутните бројки се скромни
- GRPO-final е полош од GRPO-best (не-монотонска RL динамика), затоа е битно early stopping
- Clip фракциите се високи (0.75+) заради dropout кој создава различни random маски при двата forward pass-а
- mk-mk евалуацијата е двојно ограничена: MarianMT преводот е несовршен + flan-t5 нема доволно македонски knowledge
- Token F1 нормализацијата стрипува англиски артикли (a/an/the) — не е идеално за другите јазици
- RAG компонентата не е jointly оптимизирана со GRPO

## Референци

- Lewis et al. (2020). *MLQA: Evaluating Cross-lingual Extractive QA.* ACL 2020.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning.* arXiv:2402.03300.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Chung et al. (2022). *Scaling Instruction-Finetuned Language Models.* arXiv:2210.11416.
- Tiedemann & Thottingal (2020). *OPUS-MT — Building open translation services for the World.* EAMT 2020.