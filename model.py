import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch import Tensor


## TODO:
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x: Tensor):
        return self.model(x)


## TODO:
class TextDecoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, x: dict):
        tok = self.tokenizer(x["input_ids"])
        out = self.model(tok)
        return out
    

## TODO:
class ClothDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.txt_decoder = TextDecoder()

    def forward(self, x: Tensor):
        return



if __name__ == "__main__":

    model = ClothDescriptor()
