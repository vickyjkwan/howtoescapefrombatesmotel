from collections import defaultdict
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold
import pandas as pd
from surprise import Dataset
from surprise import Reader
import random



def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls



kf = KFold(n_splits=5)
algo = SVD(lr_all=0.0019, n_epochs=26, biased=True, reg_all=0.385)

algo = NMF(n_factors=10, reg_qi=0.05, reg_pu=0.04, n_epochs=43)

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)



    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / float(len(precisions)))
    print(sum(rec for rec in recalls.values()) / float(len(recalls)))
