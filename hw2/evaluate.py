import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import argparse
import requests
import time
import json

from requests.exceptions import ConnectionError
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from typing import Tuple, List, Any, Dict

TEST_MODE = False

def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d

def read_dataset(path: str) -> List[Dict]:

    with open(path, "r") as f:
        samples = json.load(f)

    return samples

def main(test_path: str, endpoint: str, batch_size=32):

    try:
        samples = read_dataset(test_path)
    except FileNotFoundError as e:
        logging.error(f'Evaluation crashed because {test_path} does not exist')
        exit(1)
    except Exception as e:
        logging.error(f'Evaluation crashed. Most likely, the file you gave is not in the correct format')
        logging.error(f'Printing error found')
        logging.error(e, exc_info=True)
        exit(1)

    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(f'Impossible to establish a connection to the server even after 10 tries')
            logging.error('The server is not booting and, most likely, you have some error in build_model or StudentClass')
            logging.error('You can find more information inside logs/. Checkout both server.stdout and, most importantly, server.stderr')
            exit(1)

        logging.info(f'Waiting 10 second for server to go up: trial {i}/{max_try}')
        time.sleep(10)

        try:
            if TEST_MODE:
                logging.info('Test mode, skip connection test')
                break
            response = requests.post(endpoint, json={'samples': [{"text": "Lorem ipsum dolor sit amet.", "targets": [[[0, 11], "Lorem ipsum"]]}]}).json()
            response['predictions_b']
            logging.info('Connection succeded')
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)

    predictions_b = []
    predictions_ab = []
    predictions_cd = []

    progress_bar = tqdm(total=len(samples), desc='Evaluating')

    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        try:
            response = {}
            if TEST_MODE:
                response['predictions_b'] = [{"targets": [term[1:] for term in sample["targets"]]} for sample in batch]
                response['predictions_ab'] = [{"targets": [term[1:] for term in sample["targets"]]} for sample in batch]
                if "categories" in batch[0]:
                    response['predictions_cd'] = [{"categories": sample["categories"], "targets": [term[1:] for term in sample["targets"]]} for sample in batch]
            else:
                response = requests.post(endpoint, json={'samples': batch}).json()
            predictions_b.extend(response['predictions_b'])
            if 'predictions_ab' in response and response['predictions_ab']:
                predictions_ab.extend(response['predictions_ab'])
            if 'predictions_cd' in response and response['predictions_cd']:
                predictions_cd.extend(response['predictions_cd'])
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)
        progress_bar.update(len(batch))

    progress_bar.close()

    print(f'# instances: {len(predictions_b)}')

    print('MODEL: ASPECT SENTIMENT\n')
    evaluate_sentiment(samples, predictions_b)
    print('-------------------------------------------------------\n')
    
    if predictions_ab:
        print('MODEL: ASPECT SENTIMENT + ASPECT EXTRACTION\n')
        evaluate_extraction(samples, predictions_ab)
        print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
        evaluate_sentiment(samples, predictions_ab)
        print('-------------------------------------------------------\n')
    if predictions_cd:
        print('MODEL: CATEGORY SENTIMENT + CATEGORY EXTRACTION\n')
        evaluate_sentiment(samples, predictions_cd, 'Category Extraction')
        print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
        evaluate_sentiment(samples, predictions_cd, 'Category Sentiment')

def evaluate_extraction(samples, predictions_b):
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for label, pred in zip (samples, predictions_b):
        pred_terms = {term_pred[0] for term_pred in pred["targets"]}
        gt_terms = {term_gt[1] for term_gt in label["targets"]}

        scores["tp"] += len(pred_terms & gt_terms)
        scores["fp"] += len(pred_terms - gt_terms)
        scores["fn"] += len(gt_terms - pred_terms)

    precision = 100 * scores["tp"] / (scores["tp"] + scores["fp"])
    recall = 100 * scores["tp"] / (scores["tp"] + scores["fn"])
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Aspect Extraction Evaluation")

    print(
        "\tAspects\t TP: {};\tFP: {};\tFN: {}".format(
            scores["tp"],
            scores["fp"],
            scores["fn"]))
    print(
        "\t\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f}".format(
            precision,
            recall,
            f1))

def evaluate_sentiment(samples, predictions_b, mode="Aspect Sentiment"):
    scores = {}
    if mode == 'Category Extraction':
        sentiment_types = ["anecdotes/miscellaneous", "price", "food", "ambience"]
    else:
        sentiment_types = ["positive", "negative", "neutral", "conflict"]
    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in sentiment_types + ["ALL"]}
    for label, pred in zip(samples, predictions_b):
        for sentiment in sentiment_types:
            if mode == "Aspect Sentiment":
                pred_sent = {(term_pred[0], term_pred[1]) for term_pred in pred["targets"] if
                                    term_pred[1] == sentiment}
                gt_sent = {(term_pred[1], term_pred[2]) for term_pred in label["targets"] if
                                    term_pred[2] == sentiment}
            elif mode == 'Category Extraction' and "categories" in label:
                pred_sent = {(term_pred[0]) for term_pred in pred["categories"] if
                                term_pred[0] == sentiment}
                gt_sent = {(term_pred[0]) for term_pred in label["categories"] if
                                term_pred[0] == sentiment}
            elif "categories" in label:
                pred_sent = {(term_pred[0], term_pred[1]) for term_pred in pred["categories"] if
                                term_pred[1] == sentiment}
                gt_sent = {(term_pred[0], term_pred[1]) for term_pred in label["categories"] if
                                term_pred[1] == sentiment}
            else:
                continue

            scores[sentiment]["tp"] += len(pred_sent & gt_sent)
            scores[sentiment]["fp"] += len(pred_sent - gt_sent)
            scores[sentiment]["fn"] += len(gt_sent - pred_sent)

    # Compute per sentiment Precision / Recall / F1
    for sent_type in scores.keys():
        if scores[sent_type]["tp"]:
            scores[sent_type]["p"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fp"] + scores[sent_type]["tp"])
            scores[sent_type]["r"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fn"] + scores[sent_type]["tp"])
        else:
            scores[sent_type]["p"], scores[sent_type]["r"] = 0, 0

        if not scores[sent_type]["p"] + scores[sent_type]["r"] == 0:
            scores[sent_type]["f1"] = 2 * scores[sent_type]["p"] * scores[sent_type]["r"] / (
                    scores[sent_type]["p"] + scores[sent_type]["r"])
        else:
            scores[sent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[sent_type]["tp"] for sent_type in sentiment_types])
    fp = sum([scores[sent_type]["fp"] for sent_type in sentiment_types])
    fn = sum([scores[sent_type]["fn"] for sent_type in sentiment_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = sum([scores[ent_type]["f1"] for ent_type in sentiment_types])/len(sentiment_types)
    scores["ALL"]["Macro_p"] = sum([scores[ent_type]["p"] for ent_type in sentiment_types])/len(sentiment_types)
    scores["ALL"]["Macro_r"] = sum([scores[ent_type]["r"] for ent_type in sentiment_types])/len(sentiment_types)

    print(f"{mode} Evaluation\n")

    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for sent_type in sentiment_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            sent_type,
            scores[sent_type]["tp"],
            scores[sent_type]["fp"],
            scores[sent_type]["fn"],
            scores[sent_type]["p"],
            scores[sent_type]["r"],
            scores[sent_type]["f1"],
            scores[sent_type]["tp"] +
            scores[sent_type][
                "fp"]))

    return scores, precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help='File containing data you want to evaluate upon')
    args = parser.parse_args()

    main(
        test_path=args.file,
        endpoint='http://127.0.0.1:12345'
    )
