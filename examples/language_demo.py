import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

from IPython.display import Image, display

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    return output.start_logits, output.end_logits

def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

def squad_pos_forward_func2(input_emb, attention_mask=None, position=0):
    pred = model(inputs_embeds=input_emb, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values

def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def get_topk_attributed_tokens(attrs, k=5):
    values, indices = torch.topk(attrs, k)
    top_tokens = [all_tokens[idx] for idx in indices]
    return top_tokens, values, indices

def lig_2(model, input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, position_ids, ref_position_ids, attention_mask):
    lig2 = LayerIntegratedGradients(squad_pos_forward_func, \
                                [model.bert.embeddings.word_embeddings, \
                                 model.bert.embeddings.token_type_embeddings, \
                                 model.bert.embeddings.position_embeddings])

    attributions_start = lig2.attribute(inputs=(input_ids, token_type_ids, position_ids),
                                    baselines=(ref_input_ids, ref_token_type_ids, ref_position_ids),
                                    additional_forward_args=(attention_mask, 0))
    attributions_end = lig2.attribute(inputs=(input_ids, token_type_ids, position_ids),
                                    baselines=(ref_input_ids, ref_token_type_ids, ref_position_ids),
                                    additional_forward_args=(attention_mask, 1))

    attributions_start_word = summarize_attributions(attributions_start[0])
    attributions_end_word = summarize_attributions(attributions_end[0])

    attributions_start_token_type = summarize_attributions(attributions_start[1])
    attributions_end_token_type = summarize_attributions(attributions_end[1])

    attributions_start_position = summarize_attributions(attributions_start[2])
    attributions_end_position = summarize_attributions(attributions_end[2])

def pdf_attr(attrs, bins=100):
    return np.histogram(attrs, bins=bins, density=True)[0]

"""
@compute the attributions with respect to BertEmbedding
@Attributions for each word_embeddings, token_type_embeddings and position_embeddings w.r.t each embedding vector
"""

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './models'

    # load model
    # model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased")
    model.to(device)
    model.eval()
    model.zero_grad()

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

    question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."

    # word, token type and position embeddings
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    ground_truth = 'to include, empower and support humans of all kinds'
    ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
    ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
    ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

    start_scores, end_scores = predict(input_ids, \
                                   token_type_ids=token_type_ids, \
                                   position_ids=position_ids, \
                                   attention_mask=attention_mask)

    print('Question: ', question)
    print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

    attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                    baselines=ref_input_ids,
                                    additional_forward_args=(token_type_ids, position_ids, attention_mask, 0),
                                    return_convergence_delta=True)
    attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                    additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                    return_convergence_delta=True)
    
    attributions_start_sum = summarize_attributions(attributions_start)
    attributions_end_sum = summarize_attributions(attributions_end)

    # storing couple samples in an array for visualization purposes
    start_position_vis = viz.VisualizationDataRecord(
                            attributions_start_sum,
                            torch.max(torch.softmax(start_scores[0], dim=0)),
                            torch.argmax(start_scores),
                            torch.argmax(start_scores),
                            str(ground_truth_start_ind),
                            attributions_start_sum.sum(),       
                            all_tokens,
                            delta_start)

    end_position_vis = viz.VisualizationDataRecord(
                            attributions_end_sum,
                            torch.max(torch.softmax(end_scores[0], dim=0)),
                            torch.argmax(end_scores),
                            torch.argmax(end_scores),
                            str(ground_truth_end_ind),
                            attributions_end_sum.sum(),       
                            all_tokens,
                            delta_end)

    print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    viz.visualize_text([start_position_vis])

    print('\033[1m', 'Visualizations For End Position', '\033[0m')
    viz.visualize_text([end_position_vis])

    # lig_2(model, input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, position_ids, ref_position_ids, attention_mask)

    layer_attrs_start = []
    layer_attrs_end = []

    # The token that we would like to examine separately.
    # the index of the token that we would like to examine more thoroughly
    token_to_explain = 23
    layer_attrs_start_dist = []
    layer_attrs_end_dist = []

    input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                            token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
                                            position_ids=position_ids, ref_position_ids=ref_position_ids)

    for i in range(model.config.num_hidden_layers):
        lc = LayerConductance(squad_pos_forward_func2, model.bert.encoder.layer[i])
        layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(attention_mask, 0))
        layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(attention_mask, 1))
        layer_attrs_start.append(summarize_attributions(layer_attributions_start).cpu().detach().tolist())
        layer_attrs_end.append(summarize_attributions(layer_attributions_end).cpu().detach().tolist())

        # storing attributions of the token id that we would like to examine in more detail in token_to_explain
        layer_attrs_start_dist.append(layer_attributions_start[0,token_to_explain,:].cpu().detach().tolist())
        layer_attrs_end_dist.append(layer_attributions_end[0,token_to_explain,:].cpu().detach().tolist())

    # fig, ax = plt.subplots(figsize=(15,5))
    # xticklabels=all_tokens
    # yticklabels=list(range(1,13))
    # ax = sns.heatmap(np.array(layer_attrs_start), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    # plt.xlabel('Tokens')
    # plt.ylabel('Layers')
    # plt.savefig('start_layer_attributions.pdf', bbox_inches='tight', dpi=1200)

    fig, ax = plt.subplots(figsize=(15,5))
    ax = sns.boxplot(data=layer_attrs_start_dist)
    plt.xlabel('Layers')
    plt.ylabel('Attribution')
    plt.savefig('layer_attributions.pdf', bbox_inches='tight', dpi=1200)


    layer_attrs_end_pdf = map(lambda layer_attrs_end_dist: pdf_attr(layer_attrs_end_dist), layer_attrs_end_dist)
    layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))

    # summing attribution along embedding diemension for each layer
    # size: #layers
    attr_sum = np.array(layer_attrs_end_dist).sum(-1)

    # size: #layers
    layer_attrs_end_pdf_norm = np.linalg.norm(layer_attrs_end_pdf, axis=-1, ord=1)

    #size: #bins x #layers
    layer_attrs_end_pdf = np.transpose(layer_attrs_end_pdf)

    #size: #bins x #layers
    layer_attrs_end_pdf = np.divide(layer_attrs_end_pdf, layer_attrs_end_pdf_norm, where=layer_attrs_end_pdf_norm!=0)
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(layer_attrs_end_pdf)
    plt.xlabel('Bins')
    plt.ylabel('Density')
    plt.legend(['Layer '+ str(i) for i in range(1,13)])
    plt.savefig('pmf_attribution.pdf', bbox_inches='tight', dpi=1200)

    fig, ax = plt.subplots(figsize=(15,5))
    # replacing 0s with 1s. np.log(1) = 0 and np.log(0) = -inf
    layer_attrs_end_pdf[layer_attrs_end_pdf == 0] = 1
    layer_attrs_end_pdf_log = np.log2(layer_attrs_end_pdf)
    # size: #layers
    entropies= -(layer_attrs_end_pdf * layer_attrs_end_pdf_log).sum(0)
    plt.scatter(np.arange(12), attr_sum, s=entropies * 100)
    # plt.bar(np.arange(12), entropies)
    plt.xlabel('Layers')
    plt.ylabel('Total Attribution')
    plt.savefig('entropies_attribution.pdf', bbox_inches='tight', dpi=1200)


