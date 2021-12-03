"""Custom models for few-shot learning specific operations."""

import random
import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
# from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)


class RobertaForPrefixTuningSF(BertPreTrainedModel):

    def __init__(self, config, soft_prompt_path: str = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        n_tokens: int = None,):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        self.soft_prompt_tokens = None

        for params in self.roberta.parameters():
            params.requires_grad = False
        
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
        self.n_layer = config.num_hidden_layers
        self.mid_dim = 512

        self.input_tokens = torch.arange(self.n_tokens).long()
        self.wte = nn.Embedding(self.n_tokens, self.config.hidden_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.config.hidden_size))
        
        self.match_n_layer = config.num_hidden_layers
        self.match_n_head = config.num_attention_heads
        self.match_n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_dropout = 0.0
        self.dropout = nn.Dropout(self.prefix_dropout)


    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds
    
    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens),ignore_index, dtype=torch.long, device=labels.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1,dtype=torch.long, device=attention_mask.device), attention_mask],
            dim=1,
        )
    
    def get_prompt_p5(self, input_device,control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(input_device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        
        input_device = input_ids.device
        bz = input_ids.size()[0]
        past_key_values = self.get_prompt_p5(input_device,bsz=bz)


        

        # if input_ids is not None:
        #     inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
        #         input_ids.device
        #     )

        # if labels is not None:
        #     extend_labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask_extend = self._extend_attention_mask(attention_mask)

        # Encode everything
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask=attention_mask_extend,
            past_key_values = past_key_values
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