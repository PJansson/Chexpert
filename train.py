import math

import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils import data
from tqdm import tqdm

from chexpert import losses, metrics, models
from chexpert.config import create_configargparser
from chexpert.data import datasets, strategies
from chexpert.data.transforms import TemplateMatch, ToRGB
from chexpert.ema import ModelEMA
from chexpert.logger import MetricLogger
from chexpert.utils import AverageMeter

parser = create_configargparser()
args = parser.parse_args()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.num_classes = len(args.classes)
args.model_save_path.mkdir(parents=True, exist_ok=True)


preprocessing = albumentations.Compose(
    [
        TemplateMatch(args.data_dir, args.dataset),
        ToRGB(),
        albumentations.Resize(args.image_size, args.image_size),
    ]
)

train_transforms = albumentations.Compose(
    [
        albumentations.OneOf(
            [
                albumentations.GridDistortion(),
                albumentations.ElasticTransform(),
            ]
        ),
        albumentations.Normalize(),
        albumentations.CoarseDropout(
            max_holes=args.coarse_dropout_max,
            max_height=args.coarse_dropout_h,
            max_width=args.coarse_dropout_w,
            min_holes=args.coarse_dropout_min,
        ),
        ToTensorV2(),
    ]
)

valid_transforms = albumentations.Compose(
    [
        albumentations.Normalize(),
        ToTensorV2(),
    ]
)


if args.multi_view_model:
    Dataset = datasets.MultiViewDataset
else:
    Dataset = datasets.SingleViewDataset


train_dataset = Dataset(
    data_dir=args.data_dir,
    dataset=args.dataset,
    filename="train.csv",
    classes=args.classes,
    preprocessing=preprocessing,
    transforms=train_transforms,
    strategy=strategies.UBetaStrategy(5, 2),
)

valid_dataset = Dataset(
    data_dir=args.data_dir,
    dataset=args.dataset,
    filename="valid.csv",
    classes=args.classes,
    preprocessing=preprocessing,
    transforms=valid_transforms,
)


train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=args.pin_memory,
)

valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=args.pin_memory,
)

valid_every = math.ceil(len(train_loader) / args.validations_per_epoch)


if args.multi_view_model:
    model = models.MultiViewModel(
        args.model_name, args.num_classes, args.max_before_pool
    ).to(args.device)
else:
    model = models.SingleViewModel(args.model_name, args.num_classes).to(args.device)

ema = ModelEMA(model, decay=args.ema_decay, device=args.device)
criterion = losses.LabelCorrelationAwareLoss()
train_losses = AverageMeter()
valid_losses = AverageMeter()

if args.per_study_auroc:
    valid_auroc = metrics.PerStudyAUROC(
        valid_dataset,
        class_scores=args.auroc_class_scores,
    )
else:
    valid_auroc = metrics.AUROC(class_scores=args.auroc_class_scores)

metric_logger = MetricLogger(
    args.classes, args.auroc_class_scores, args.per_study_auroc, args.model_save_path
)


optimizer = optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=args.scheduler_factor,
    patience=args.scheduler_patience,
)

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)


def validate(highest_score, previous_best):
    model.eval()
    valid_losses.reset()
    valid_auroc.reset()

    with torch.no_grad():
        for x, y in tqdm(valid_loader, leave=False, desc=f"Epoch: {epoch}", ncols=80):
            x = x.to(args.device)
            y = y.to(args.device)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                y_hat = ema.module(x)
                loss = criterion(y_hat, y)

            valid_losses.update(loss.item(), y.size(0))
            valid_auroc.update(y_hat, y)

    scheduler.step(valid_losses.avg)

    valid_auroc_score = valid_auroc.compute_mean_only()
    if valid_auroc_score > highest_score:
        highest_score = valid_auroc_score
        filename = f"{args.model_name}_{valid_auroc_score:.3f}.pth"
        if previous_best:
            if previous_best.exists():
                previous_best.unlink()
        previous_best = args.model_save_path / filename
        torch.save(ema.module.state_dict(), previous_best)

    log = metric_logger(epoch, train_losses.avg, valid_losses.avg, valid_auroc)
    tqdm.write(log)

    return highest_score, previous_best


highest_score = 0
previous_best = None

for epoch in range(args.epochs):
    model.train()
    train_losses.reset()

    for i, (x, y) in enumerate(
        tqdm(train_loader, leave=False, desc=f"Epoch: {epoch}", ncols=80)
    ):
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        scaler.scale(loss).backward()

        if (i + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        train_losses.update(loss.item(), y.size(0))

        if i != 0 and i % valid_every == 0:
            highest_score, previous_best = validate(highest_score, previous_best)
            model.train()

    highest_score, previous_best = validate(highest_score, previous_best)
