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


class RobertaForPromptTuning(BertPreTrainedModel):

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
        
        if soft_prompt_path is not None:
            self.roberta.set_soft_prompt_embeds(soft_prompt_path)
        elif self.n_tokens is not None:
            print("Initializing soft prompt...")
            self.soft_prompt = self.initialize_soft_prompt(n_tokens=self.n_tokens,
                initialize_from_vocab=self.initialize_from_vocab,random_range=self.random_range)
        
        
        self.return_full_softmax = None
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
    
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, soft_prompt_path: str = None,
    #     initialize_from_vocab: bool = True,
    #     random_range: float = 0.5,
    #     n_tokens: int = None,*model_args, **kwargs):

    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    #     for param in model.parameters():
    #         param.requires_grad = False

    #     if soft_prompt_path is not None:
    #         model.set_soft_prompt_embeds(soft_prompt_path)
    #     elif n_tokens is not None:
    #         print("Initializing soft prompt...")
    #         model.initialize_soft_prompt(
    #             n_tokens=n_tokens,
    #             initialize_from_vocab=initialize_from_vocab,
    #             random_range=random_range,
    #         )

    #     return model


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

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds
    
    # def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
    #     if len(list(labels.shape)) == 1:
    #         labels = labels.unsqueeze(0)

    #     n_batches = labels.shape[0]
    #     return torch.cat(
    #         [
    #             torch.full((n_batches, self.n_tokens),ignore_index, dtype=torch.long, device=self.device),
    #             labels,
    #         ],
    #         dim=1,
    #     )

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
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                input_ids.device
            )

        # if labels is not None:
        #     labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # Encode everything
        outputs = self.roberta(
            attention_mask=attention_mask,
            inputs_embeds = inputs_embeds,
        )
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

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