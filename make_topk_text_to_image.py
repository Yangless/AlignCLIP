# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import numpy as np
import torch
from tqdm import tqdm
import json


def load_features(feature_path):
    ids = []
    feats = []
    with open(feature_path, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            ids.append(obj['image_id'] if 'image_id' in obj else obj['text_id'])
            feats.append(obj['feature'])
    feats_array = np.array(feats, dtype=np.float32)
    return ids, feats_array


def compute_predictions(text_ids, text_feats, image_ids, image_feats, top_k, eval_batch_size):
    predictions = []
    text_feats_tensor = torch.tensor(text_feats, dtype=torch.float).cuda()
    image_feats_tensor = torch.from_numpy(image_feats).cuda()

    for text_id, text_feat_tensor in tqdm(zip(text_ids, text_feats_tensor), total=len(text_ids)):
        score_tuples = []
        idx = 0
        while idx < len(image_ids):
            img_feats_batch = image_feats_tensor[idx: min(idx + eval_batch_size, len(image_ids))]
            batch_scores = text_feat_tensor @ img_feats_batch.t()
            for image_id, score in zip(image_ids[idx: min(idx + eval_batch_size, len(image_ids))],
                                       batch_scores.squeeze(0).tolist()):
                score_tuples.append((image_id, score))
            idx += eval_batch_size
        top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k]
        predictions.append({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})

    return predictions


def save_predictions(predictions, output_path):
    with open(output_path, "w") as fout:
        for pred in predictions:
            fout.write("{}\n".format(json.dumps(pred)))


def run_knn_search(image_feats_path, text_feats_path, top_k, eval_batch_size, output_path):
    print("Begin to load image features...")
    image_ids, image_feats = load_features(image_feats_path)
    print("Finished loading image features.")

    print("Begin to load text features...")
    text_ids, text_feats = load_features(text_feats_path)
    print("Finished loading text features.")

    print("Begin to compute top-{} predictions for texts...".format(top_k))
    predictions = compute_predictions(text_ids, text_feats, image_ids, image_feats, top_k, eval_batch_size)
    print("Finished computing top-{} predictions.".format(top_k))

    print("Saving predictions to {}...".format(output_path))
    save_predictions(predictions, output_path)
    print("Top-{} predictions are saved in {}".format(top_k, output_path))
    print("Done!")

