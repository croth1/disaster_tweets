from torch import nn
import torch
from transformers.models import roberta
from transformers import file_utils, AutoTokenizer

import copy


class KeywordModule(nn.Module):
    def __init__(self, word_embeddings, config):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.clf = nn.Linear(in_features=config.hidden_size, out_features=config.kw_neurons)
        self.clf_norm = nn.LayerNorm(normalized_shape=config.kw_neurons)
        self.clf_dropout = nn.Dropout(p=config.kw_dropout)

    def forward(self, keyword_ids, keyword_mask, **args):

        embeddings = self.word_embeddings(keyword_ids)
        emb_weighted = (embeddings * keyword_mask[:, :, None]).sum(axis=1)
        total_weight = keyword_mask.sum(axis=1)
        mean_embedding = emb_weighted / total_weight[:, None]

        out = self.clf(mean_embedding)
        out = self.clf_norm(out)
        out = torch.relu(out)
        out = self.clf_dropout(out)
        return out


class ClassificationModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        join_layer_in = config.hidden_size + config.kw_neurons
        self.dense = nn.Linear(in_features=join_layer_in, out_features=config.clf_neurons)
        self.dropout = nn.Dropout(config.clf_dropout)
        self.out_proj = nn.Linear(config.clf_neurons, config.num_labels)

    def forward(self, x, **args):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DisasterModel(nn.Module):
    def __init__(self, disaster_config):
        super().__init__()
        model_str = disaster_config['hf_roberta_model']
        roberta_model = roberta.RobertaModel.from_pretrained(model_str)
        self._tokenizer = AutoTokenizer.from_pretrained(model_str)
        config = roberta_model.config
        config.update(disaster_config)

        if config.freeze_roberta_params:
            for param in roberta_model.parameters():
                param.requires_grad = False

        word_embeddings = roberta_model.embeddings.word_embeddings
        self.roberta = roberta_model
        self.keyword = KeywordModule(word_embeddings, config)
        self.clf = ClassificationModule(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, keyword_mask, keyword_ids, input_ids, attention_mask, labels=None, **args):
        roberta_args = copy.copy(args)
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **args)
        roberta_out = roberta_outputs[0]
        keyword_out = self.keyword(keyword_ids, keyword_mask, **args)
        logits = self.clf(torch.cat((roberta_out[:,0,:], keyword_out), dim=1))

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        out = file_utils.ModelOutput(loss=loss, logits=logits)
        return out

    @property
    def tokenizer(self):
        return self._tokenizer
