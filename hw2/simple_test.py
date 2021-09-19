from typing import List, Dict

from stud.implementation import build_model_b, build_model_ab, build_model_cd

def prepare_data(data):
    data_a = []
    data_b = []
    for sample in data:
        data_a.append({"text": sample["text"]})
        data_b.append({"text": sample["text"], "targets": [term[0:2] for term in sample["targets"]]})
    return data_a, data_b

def main(samples: List[Dict]):

    model_b = build_model_b('cpu')
    try:
        model_ab = build_model_ab('cpu')
    except:
        model_ab = None
    try:
        model_cd = build_model_cd('cpu')
    except:
        model_cd = None

    data_a, data_b = prepare_data(samples)
    predicted_b = model_b.predict(data_b)

    if model_ab:
        predicted_ab = model_ab.predict(data_a)
    else:
        predicted_ab = None

    if model_cd:
        predicted_cd = model_cd.predict(data_a)
    else:
        predicted_cd = None

    for sample, pred_b, pred_ab, pred_cd in zip(samples, predicted_b, predicted_ab, predicted_cd):
        print(f'# Text = {sample["text"]}')
        print(f'# Targets:               {sample["targets"]}')

        print(f'# Prediction (model_b):  {pred_b["targets"]}')
        if model_ab:
            print(f'# Prediction (model_ab): {pred_ab["targets"]}')
        if model_cd:
            print(f'# Targets categories:    {sample["categories"]}')
            print(f'# Prediction (model_ab): {pred_cd["categories"]}')
        print()


if __name__ == '__main__':
    main([{"categories": [
                    [
                        "food",
                        "negative"
                    ]
                ],
            "targets": [
                        [
                            [11, 29],
                            "wines by the glass",
                            "negative"
                        ]
                    ],
            "text": "Not enough wines by the glass either."
            },
            {"categories": [
                    [
                        "service",
                        "conflict"
                    ]
                ],
                "targets": [
                        [
                            [89, 94],
                            "staff",
                            "conflict"
                        ]
                    ],
            "text": "My wife and I always enjoy the young, not always well trained but nevertheless friendly, staff, all of whom have a story."
            }])
