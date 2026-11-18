import bert_score
import numpy as np
import spacy
from rouge_score import rouge_scorer

from .utils import add_metrics, connect

_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
_bert_scorer = bert_score.BERTScorer(lang="en")


def punc_scores(df_captions, OUT_DB):
    """PUNC scores of the N captions of one prompt: values per image, their mean
    and spread, plus PUNC's proxies (1 - mean precision / recall)."""
    for image_id, prompt, caption in df_captions.itertuples(index=False):
        rouge = _scorer.score(prompt, caption)
        with connect(OUT_DB) as con:
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rouge1_precision",
                        "value": rouge["rouge1"].precision,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rouge1_recall",
                        "value": rouge["rouge1"].recall,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rouge1_f1",
                        "value": rouge["rouge1"].fmeasure,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rougeL_precision",
                        "value": rouge["rougeL"].precision,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rougeL_recall",
                        "value": rouge["rougeL"].recall,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "rougeL_f1",
                        "value": rouge["rougeL"].fmeasure,
                    }
                ],
            )
    df = df_captions.copy()
    df["caption"] = df["caption"].fillna("").str.strip()
    df["text"] = df["text"].fillna("").str.strip()

    empty = (df["caption"] == "") | (df["text"] == "")
    if empty.any():
        print(
            f"PUNC: skipping {empty.sum()} rows with empty caption/prompt: "
            f"{df.loc[empty, 'image_id'].tolist()[:10]}"
        )
        with connect(OUT_DB) as con:
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "skipped_empty",
                        "value": -999.0,
                    }
                    for image_id in df.loc[empty, "image_id"]
                ],
            )
    df = df[~empty]
    if df.empty:
        return

    P, R, F = _bert_scorer.score(df["caption"].tolist(), df["text"].tolist())
    for image_id, p, r, f in zip(df["image_id"], P.tolist(), R.tolist(), F.tolist()):
        with connect(OUT_DB) as con:
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "bert_precision",
                        "value": p,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "bert_recall",
                        "value": r,
                    }
                ],
            )
            add_metrics(
                con,
                [
                    {
                        "image_id": image_id,
                        "name": "punc",
                        "metric": "bert_f1",
                        "value": f,
                    }
                ],
            )


def punc_scores_simple(prompt: str, captions: list[str]) -> dict:
    """PUNC scores of the N captions of one prompt: values per image, their mean
    and spread, plus PUNC's proxies (1 - mean precision / recall)."""
    rouge = [_scorer.score(prompt, cap) for cap in captions]
    P, R, F = _bert_scorer.score(captions, [prompt] * len(captions))
    per_image = {
        "rouge1_precision": [s["rouge1"].precision for s in rouge],
        "rouge1_recall": [s["rouge1"].recall for s in rouge],
        "rouge1_f1": [s["rouge1"].fmeasure for s in rouge],
        "rougeL_precision": [s["rougeL"].precision for s in rouge],
        "rougeL_recall": [s["rougeL"].recall for s in rouge],
        "rougeL_f1": [s["rougeL"].fmeasure for s in rouge],
        "bert_precision": P.tolist(),
        "bert_recall": R.tolist(),
        "bert_f1": F.tolist(),
    }
    out = {"per_image": per_image}
    for name, values in per_image.items():
        out[f"{name}_mean"] = float(np.mean(values))
        out[f"{name}_std"] = float(np.std(values))
    for kind in ["rouge1", "rougeL", "bert"]:
        out[f"{kind}_aleatoric"] = 1 - out[f"{kind}_precision_mean"]
        out[f"{kind}_epistemic"] = 1 - out[f"{kind}_recall_mean"]
    return out


# Sara's implementation of PUNC
nlp = spacy.load("en_core_web_sm")


def extract_concepts(text):
    """
    extract noun-based concepts from a sentence using lemmatization.
    """
    doc = nlp(text.lower())
    return {token.lemma_ for token in doc if token.pos_ == "NOUN"}


def punc_scores_wrong(prompt: str, captions: list[str]) -> dict:
    # I only copied it for a singular caption, but the issue of her implemantation is present regardless
    prompt_concepts = extract_concepts(prompt)
    caption_concepts = extract_concepts(captions[0])

    intersection = prompt_concepts & caption_concepts

    if caption_concepts:
        precision = len(intersection) / len(caption_concepts)
    else:
        precision = 0.0

    if prompt_concepts:
        recall = len(intersection) / len(prompt_concepts)
    else:
        recall = 0.0

    aleatoric = 1 - precision
    epistemic = 1 - recall

    return {
        "prompt_concepts": sorted(prompt_concepts),
        "caption_concepts": sorted(caption_concepts),
        "concepts_intersection": sorted(intersection),
        "concepts_precision": precision,
        "concepts_recall": recall,
        "concepts_aleatoric": aleatoric,
        "concepts_epistemic": epistemic,
    }


if __name__ == "__main__":
    prompt = "a yellow banana next to a blue strawberry"
    captions = ["a blue banana next to a yellow strawberry"]
    print("Original Punc implementation:")
    for k, v in punc_scores_simple(prompt, captions).items():
        print(f"{k:17s} {v if isinstance(v, dict) else round(v, 3)}")
    print("Sara's implementation:")
    for k, v in punc_scores_wrong(prompt, captions).items():
        print(f"{k:17s} {v if isinstance(v, list) else round(v, 3)}")

    prompt = "thirty sheep in a field"
    captions = ["A flock of about ten sheep grazing in a green field under a blue sky."]
    print("Original Punc implementation:")
    for k, v in punc_scores_simple(prompt, captions).items():
        print(f"{k:17s} {v if isinstance(v, dict) else round(v, 3)}")
    print("Sara's implementation:")
    for k, v in punc_scores_wrong(prompt, captions).items():
        print(f"{k:17s} {v if isinstance(v, list) else round(v, 3)}")
