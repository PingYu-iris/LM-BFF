"""Custom models for few-shot learning specific operations."""

import random
import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead, RobertaEncoder, RobertaLayer
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)



from prettytable import PrettyTable
def count_train_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        # if not parameter.requires_grad: 
        #     continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Params: {total_params}")
    return total_params


class RobertaForPromptTuning(BertPreTrainedModel):

    def __init__(self, config, soft_prompt_path: str = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        n_tokens: int = None,):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaCompactor(config)
        # self.roberta_old = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        self.soft_prompt_tokens = None
        

        for name,params in self.roberta.named_parameters():
            if "adapter" in name:
                continue
            params.requires_grad = False
        
        # for name,params in self.roberta_old.named_parameters():
        #     if name=="encoder.layer.0.attention.self.query.weight":
        #         print("here")
        
        # import numpy as np
        # model_parameters_train = filter(lambda p: p.requires_grad, self.roberta.parameters())
        # params_train = sum([np.prod(p.size()) for p in model_parameters_train])
        # for params in self.roberta.parameters():
        #     params.requires_grad = False
        # for params in self.classifier.parameters():
        #     params.requires_grad = False
        # for params in self.lm_head.parameters():
        #     params.requires_grad = False
        
        self.n_tokens = n_tokens
        self.initialize_from_vocab=initialize_from_vocab
        self.random_range=random_range
        
        # if soft_prompt_path is not None:
        #     self.roberta.set_soft_prompt_embeds(soft_prompt_path)
        # elif self.n_tokens is not None:
        #     print("Initializing soft prompt...")
        #     self.soft_prompt = self.initialize_soft_prompt(n_tokens=self.n_tokens,
        #         initialize_from_vocab=self.initialize_from_vocab,random_range=self.random_range)
        
        
        self.return_full_softmax = None
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.positive_ids = None
        self.negative_ids = None


    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.roberta.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        soft_prompt = nn.Embedding(n_tokens, self.config.hidden_size)
        # Initialize weight
        soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)
        return soft_prompt

    
    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        # inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)
        inputs_embeds = torch.cat([inputs_embeds[:,0,:].unsqueeze(1),learned_embeds,inputs_embeds[:,1:,:]],dim=1)

        return inputs_embeds
    

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1,dtype=torch.long, device=attention_mask.device), attention_mask],
            dim=1,
        )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output, pooled_output = outputs[:2]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        
    
        
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self):
        super().__init__()
        self.input_dim = 1024
        self.reduction_factor = 2
        self.down_sample_size = self.input_dim // self.reduction_factor
        self.activation = nn.ReLU()
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 
    
# add compacter layer
class RobertaLayerAdap(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter()
        
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        layer_output_adapter = self.adapter(layer_output)
        return layer_output_adapter
        
 
    
class RobertaEncoderAdapter(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([RobertaLayerAdap(config) for _ in range(config.num_hidden_layers)])
    
    
class RobertaCompactor(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = RobertaEncoderAdapter(config)
        