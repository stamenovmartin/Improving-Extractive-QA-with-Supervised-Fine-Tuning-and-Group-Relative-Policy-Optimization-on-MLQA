import re
import string
import numpy as np

import config


def normalize_text(text):
    text = text.lower().strip()
    # тргни articles (a, an, the) како што прави SQuAD eval
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)


def compute_exact_match(prediction, ground_truth):
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def compute_reward(prediction, ground_truth, num_tokens=0):
    f1 = compute_f1(prediction, ground_truth)
    em = compute_exact_match(prediction, ground_truth)
    r = config.REWARD_F1_WEIGHT * f1 + config.REWARD_EM_WEIGHT * em
    # казна за предолги одговори
    if num_tokens > config.REWARD_MAX_LENGTH:
        r -= config.REWARD_LENGTH_PENALTY * (num_tokens - config.REWARD_MAX_LENGTH)
    return r


def normalize_rewards_zscore(rewards):
    arr = np.array(rewards, dtype=np.float64)
    std = arr.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    return ((arr - arr.mean()) / (std + 1e-8)).tolist()
