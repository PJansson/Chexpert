import cv2
import numpy as np
import pandas as pd
from torch.utils import data


class SingleViewDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        dataset,
        filename,
        classes,
        preprocessing,
        transforms,
        strategy=None,
    ):
        super().__init__()
        self.df = pd.read_csv(f"{data_dir}/{dataset}/{filename}").fillna(0)
        self.df["Path"] = f"{data_dir}/" + self.df["Path"]
        self.classes = classes

        self.files = self.df["Path"].values
        self.labels = self.df[classes].values.astype(np.float32)
        self.views = self.df["Frontal/Lateral"].values
        self.preprocessing = preprocessing
        self.transform = transforms
        self.strategy = strategy

    def __getitem__(self, index):
        x = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        y = self.labels[index]
        v = self.views[index]
        x = self.preprocessing(image=x, view=v)["image"]
        x = self.transform(image=x)["image"]
        if self.strategy:
            y = self.strategy(y)
        return x, y

    def __len__(self):
        return len(self.df)


class MultiViewDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        dataset,
        filename,
        classes,
        preprocessing,
        transforms,
        strategy=None,
    ):
        super().__init__()

        self.df = pd.read_csv(f"{data_dir}/{dataset}/{filename}").fillna(0)
        self.df["Path"] = f"{data_dir}/" + self.df["Path"]
        self.df = self.per_study_df(self.df, classes)
        self.classes = classes

        self.files = self.df["Path"].values
        self.labels = self.df[classes].values.astype(np.float32)
        self.views = self.df["Frontal/Lateral"].values
        self.preprocessing = preprocessing
        self.transform = transforms
        self.strategy = strategy
        self.image_size = preprocessing[-1].height

    def get_image(self, file, view):
        f = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        x = self.preprocessing(image=f, view=view)["image"]
        x = self.transform(image=x)["image"]
        return x

    def __getitem__(self, index):
        files = self.files[index]
        views = self.views[index]

        x = np.zeros((3, 3, self.image_size, self.image_size), dtype=np.float32)
        for i, (f, v) in enumerate(zip(files, views)):
            x[i:] = self.get_image(f, v)

        y = self.labels[index]
        if self.strategy:
            y = self.strategy(y)

        return x, y

    def __len__(self):
        return len(self.df)

    def per_study_df(self, df, columns):
        df["Patient"] = df["Path"].apply(lambda x: x.rsplit("/", 3)[1])
        df["Study"] = df["Path"].apply(lambda x: x.rsplit("/", 3)[2])
        aggregation = {
            **{"Path": list, "Frontal/Lateral": list},
            **{c: "first" for c in columns},
        }
        return df.groupby(["Patient", "Study"]).agg(aggregation)
