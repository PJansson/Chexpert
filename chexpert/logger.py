class MetricLogger:
    def __init__(self, classes, auroc_class_scores, per_study_auroc, path=None):
        self.per_study_auroc = per_study_auroc
        self.auroc_class_scores = auroc_class_scores

        if per_study_auroc:
            self.header = "| Epoch | Train loss | Valid loss |  Img AUC  | Study AUC |"
            self.format_string = "| {:5d} | {:.8f} | {:.8f} | {:.7f} | {:.7f} |"
        else:
            self.header = "| Epoch | Train loss | Valid loss |  Img AUC  |"
            self.format_string = "| {:5d} | {:.8f} | {:.8f} | {:.7f} |"

        if auroc_class_scores:
            self.columns = " | ".join([c[:5] for c in classes])
            self.column_format = " | ".join(["{:.3f}" for _ in classes])

            self.header += f" {self.columns} |"
            self.format_string += f" {self.column_format} |"

        print(self.header)

    def __call__(self, epoch, train_loss, valid_loss, valid_auroc):
        if self.per_study_auroc:
            if valid_auroc.class_scores:
                im_auroc, study_auroc, scores = valid_auroc.compute()
                write = self.format_string.format(
                    epoch, train_loss, valid_loss, im_auroc, study_auroc, *scores
                )
            else:
                im_auroc, study_auroc = valid_auroc.compute()
                write = self.format_string.format(
                    epoch, train_loss, valid_loss, im_auroc, study_auroc
                )
        else:
            if valid_auroc.class_scores:
                auroc, scores = valid_auroc.compute()
                write = self.format_string.format(
                    epoch, train_loss, valid_loss, auroc, *scores
                )
            else:
                auroc = valid_auroc.compute()
                write = self.format_string.format(epoch, train_loss, valid_loss, auroc)

        return write
