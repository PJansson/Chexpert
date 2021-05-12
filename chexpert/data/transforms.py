import cv2
import numpy as np
from albumentations import ImageOnlyTransform


class TemplateMatch(ImageOnlyTransform):
    def __init__(self, data_dir, dataset, template_size=320, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        template_frontal = f"{data_dir}/{dataset}/templates/frontal.jpg"
        template_lateral = f"{data_dir}/{dataset}/templates/lateral.jpg"

        self.template_frontal = cv2.imread(template_frontal, cv2.IMREAD_GRAYSCALE)
        self.template_lateral = cv2.imread(template_lateral, cv2.IMREAD_GRAYSCALE)
        self.template_size = template_size

    def apply(self, image, **params):
        if params["view"] == "Frontal":
            template = self.template_frontal
        elif params["view"] == "Lateral":
            template = self.template_lateral
        else:
            raise KeyError(
                "You have to pass view = 'Frontal' or 'Lateral' as named argument.\
                For example: aug(view='Frontal')"
            )

        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
        _, _, _, (x, y) = cv2.minMaxLoc(res)
        image = image[y : y + self.template_size, x : x + self.template_size]
        return image

    def update_params(self, params, **kwargs):
        params.update({"view": kwargs["view"]})
        return params


class ToRGB(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return np.repeat(image[:, :, None], 3, axis=2)
