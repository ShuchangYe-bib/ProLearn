import torch.nn as nn
from transformers import AutoModel
from open_clip import create_model_from_pretrained, get_tokenizer


class BERTModel(nn.Module):

    def __init__(self, args):

        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(args.bert_type,output_hidden_states=True,trust_remote_code=True)
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)[-1]


class CLIPBERTModel(nn.Module):

    def __init__(self, args):
        super(CLIPBERTModel, self).__init__()
        self.args = args
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def forward(self, text):
        text = self.tokenizer(text)
        return self.model.text.transformer(text.to(self.args.device)).last_hidden_state