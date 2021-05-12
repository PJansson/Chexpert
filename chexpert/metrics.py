import torch
from sklearn.metrics import roc_auc_score


class AUROC:
    def __init__(self, class_scores=False):
        self.class_scores = class_scores
        self.y_true = []
        self.y_score = []

    def update(self, x, y):
        self.y_true.append(y.cpu())
        self.y_score.append(x.cpu())

    def compute(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)

        # Masks out classes with 0 positive labels
        mask = ~((y_true == 0).all(0) | (y_true == 1).all(0))
        y_true = y_true[:, mask]
        y_score = y_score[:, mask]

        scores = roc_auc_score(y_true, y_score, average=None)

        if self.class_scores:
            unmasked_scores = []
            i = 0
            for m in mask:
                score = scores[i] if m else 0.5
                i = i + 1 if m else i
                unmasked_scores.append(score)

        if self.class_scores:
            return scores.mean(), unmasked_scores

        return scores.mean()

    def compute_mean_only(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)

        mask = ~((y_true == 0).all(0) | (y_true == 1).all(0))
        y_true = y_true[:, mask]
        y_score = y_score[:, mask]

        score = roc_auc_score(y_true, y_score)
        return score

    def reset(self):
        self.y_true = []
        self.y_score = []


class PerStudyAUROC:
    def __init__(self, dataset, class_scores=False, aggregation="mean"):
        self.df = dataset.df
        self.df["Patient"] = self.df["Path"].apply(lambda x: x.rsplit("/", 3)[1])
        self.df["Study"] = self.df["Path"].apply(lambda x: x.rsplit("/", 3)[2])
        self.classes = dataset.classes
        self.classes_pred = [c + " Pred" for c in self.classes]
        self.aggregation = {
            **{"Path": list, "Frontal/Lateral": list},
            **{c: "first" for c in self.classes},
            **{c: aggregation for c in self.classes_pred},
        }

        self.class_scores = class_scores
        self.y_true = []
        self.y_score = []

    def update(self, x, y):
        self.y_true.append(y.cpu())
        self.y_score.append(x.cpu())

    def compute_auroc(self, y_true, y_score):
        # Masks out classes with 0 positive labels
        mask = ~((y_true == 0).all(0) | (y_true == 1).all(0))
        y_true = y_true[:, mask]
        y_score = y_score[:, mask]

        scores = roc_auc_score(y_true, y_score, average=None)
        return scores, mask

    def compute(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)

        per_image_auroc, _ = self.compute_auroc(y_true, y_score)

        self.df[self.classes] = y_true
        self.df[self.classes_pred] = y_score

        grouped = self.df.groupby(["Patient", "Study"]).agg(self.aggregation)
        y_true = grouped[self.classes].values
        y_score = grouped[self.classes_pred].values

        scores, mask = self.compute_auroc(y_true, y_score)

        if self.class_scores:
            unmasked_scores = []
            i = 0
            for m in mask:
                score = scores[i] if m else 0.5
                i = i + 1 if m else i
                unmasked_scores.append(score)

        if self.class_scores:
            return per_image_auroc.mean(), scores.mean(), unmasked_scores

        return per_image_auroc.mean(), scores.mean()

    def compute_mean_only(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)

        self.df[self.classes] = y_true
        self.df[self.classes_pred] = y_score

        grouped = self.df.groupby(["Patient", "Study"]).agg(self.aggregation)
        y_true = grouped[self.classes].values
        y_score = grouped[self.classes_pred].values

        scores, mask = self.compute_auroc(y_true, y_score)
        return scores.mean()

    def reset(self):
        self.y_true = []
        self.y_score = []
