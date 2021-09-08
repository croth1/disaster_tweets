import roberta_model
from types import SimpleNamespace
from transformers import file_utils

def get_default_model():
    cfg = SimpleNamespace()
    cfg.kw_neurons = 10
    cfg.kw_dropout = 0.1
    cfg.clf_neurons = 5
    cfg.clf_dropout = 0.1
    cfg.freeze_roberta_params = True
    cfg.hf_roberta_model = 'distilroberta-base'
    model = roberta_model.DisasterModel(vars(cfg))

    return model


def mask_special_tokens(embeddings, mask):
    mask[embeddings <= 2] = 0
    return embeddings, mask


def tokenize_fn(examples, tokenizer, return_tensors=file_utils.TensorType.NUMPY):
    data_dict = tokenizer(examples['text'], padding="max_length", truncation=True,
                          return_tensors=return_tensors)
    kw_dict = tokenizer(examples['keywords'], padding="max_length", truncation=True,
                        return_tensors=return_tensors)
    kw_ids, kw_mask = mask_special_tokens(kw_dict.input_ids, kw_dict.attention_mask)
    data_dict['keyword_ids'] = kw_ids
    data_dict['keyword_mask'] = kw_mask
    return data_dict


def preprocess_df(df):
    df = df.rename(columns={
        'target':'label',
        'keyword': 'keywords',
    })

    df.loc[:, 'keywords'].fillna('unknown', inplace=True)
    df.loc[:, 'keywords'] = df.loc[:, 'keywords'].str.replace('%20', ' ')
    return df