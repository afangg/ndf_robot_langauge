import clip
import torch
import numpy as np
from PIL import Image


class MyCLIP:
    def __init__(self, device=None, model_type="ViT-B/32"):
        # self.is_cuda = is_cuda

        print("The following models are available:", clip.available_models())
        if not (model_type in clip.available_models()):
            print(f"model_type {model_type} not available. Defaulting to ViT-B/32")
            model_type = "ViT-B/32"
        model, preprocess = clip.load("ViT-B/32")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("CLIP model running on", self.device)

        if self.device == "cuda":
            self.model = model.cuda().eval()

        else:
            self.model = model.to(self.device).float().eval()

        # self.model = model.to(self.device).eval()
        self.image_preprocess = preprocess

    def get_image_embeddings(self, rgbs):
        images = []
        for r in rgbs:
            r = Image.fromarray(np.uint8(r))
            images.append(self.image_preprocess(r))
        image_input = torch.tensor(np.stack(images))

        if self.device == "cuda":
            image_input = image_input.cuda()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def get_text_embeddings(self, texts):
        text_tokens = clip.tokenize(texts)
        if self.device == "cuda":
            text_tokens = text_tokens.cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def get_similarity_score(self, rgbs, texts):
        text_features = self.get_text_embeddings(texts)
        image_features = self.get_image_embeddings(rgbs)
        similarity = text_features @ image_features.T
        return similarity