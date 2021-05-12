import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils import data
from tqdm import tqdm

from chexpert import models
from chexpert.config import create_configargparser
from chexpert.data import datasets
from chexpert.data.transforms import TemplateMatch, ToRGB

parser = create_configargparser()
args = parser.parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.num_classes = len(args.classes)

assert (
    args.model_path is not None
), "No model defined, use --model_path to define a model to validate."

preprocessing = albumentations.Compose(
    [
        TemplateMatch(args.data_dir, args.dataset),
        ToRGB(),
        albumentations.Resize(args.image_size, args.image_size),
    ]
)

transforms = albumentations.Compose(
    [
        albumentations.Normalize(),
        ToTensorV2(),
    ]
)

if args.multi_view_model:
    Dataset = datasets.MultiViewDataset
else:
    Dataset = datasets.SingleViewDataset

dataset = Dataset(
    data_dir=args.data_dir,
    dataset=args.dataset,
    filename="valid.csv",
    classes=args.classes,
    preprocessing=preprocessing,
    transforms=transforms,
)

dataloader = data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=args.pin_memory,
)

if args.multi_view_model:
    model = models.MultiViewModel(
        args.model_name, args.num_classes, args.max_before_pool
    ).to(args.device)
else:
    model = models.SingleViewModel(args.model_name, args.num_classes).to(args.device)

state_dict = torch.load(args.model_path, map_location=args.device)
model.load_state_dict(state_dict)
model.eval()


predictions = []
with torch.no_grad():
    for x, y in tqdm(dataloader, leave=False, ncols=80):
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            y_hat = model(x)
            predictions.append(y_hat)

predictions = torch.cat(predictions).cpu().float()
predictions = torch.sigmoid(predictions).numpy()
labels = dataset.df[args.classes].values

scores = roc_auc_score(labels, predictions, average=None)

for i, c in enumerate(args.classes):
    print(f"{c:<20} {scores[i]:.3f}")
print("--------------------------")
print(f"{'Mean':<20} {scores.mean():.3f}")
