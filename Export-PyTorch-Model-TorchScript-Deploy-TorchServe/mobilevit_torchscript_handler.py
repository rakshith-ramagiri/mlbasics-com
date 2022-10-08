import io
import os
import json
import torch
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class MobileViTHandler(BaseHandler):
    """Custom handler for MobileViT classification model."""

    # image transforms
    image_dim = (256, 256)
    custom_transforms = transforms.Compose(
            [
                transforms.Resize(image_dim),
                transforms.ToTensor(),
            ]
    )


    def initialize(self, context):
        """Load torchscript model from file and initialize model object.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing.
            NotImplementedError: Raises this error when trying to load model in eager mode.
        """

        # + setup runtime properties and manifest
        properties = context.system_properties
        self.device = 'cpu'
        self.manifest = context.manifest

        # + setup paths
        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # * load torchscript model
        logger.debug("Loading torchscript model")
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # + load model
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()
        
        # read and store idx2class mapping
        with open('birds400_index_to_class.json') as f:
            self.idx2class = json.load(f)

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # + let pytorch-serve know model is loaded
        self.initialized = True


    def preprocess(self, data):
        """Preprocessing code for transforming input image for 'doclayout_detect' model."""

        for row in data:
            image = row.get("data") or row.get("body")

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                # + load and read image data
                image = Image.open(io.BytesIO(image))
                image_tensor = self.custom_transforms(image).unsqueeze(0)

            # + other methods of receiving image data is not yet supported
            else:
                raise NotImplementedError()

        return image_tensor


    def inference(self, data, *args, **kwargs):
        """Inference Function is used to make a prediction call on the given input request.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """

        # no need to transfer input data to GPU since it's already done in '@preprocess' function
        with torch.no_grad():
            results = self.model(data.to(self.device), *args, **kwargs)
        return results


    def postprocess(self, data):
        """Postprocessing code for dealing with model output."""
        _, predicted = torch.max(data, 1)
        predicted = predicted.cpu().numpy().tolist()[0]
        return [{'status': 'ok', 'class': self.idx2class.get(str(predicted), 'unknown')}]
