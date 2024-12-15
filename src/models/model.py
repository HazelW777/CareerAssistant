# src/models/model.py
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoConfig

class InterviewQuestionClassifier(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config.name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 添加分类头
        self.field_classifier = nn.Linear(config.hidden_size, config.field_size if hasattr(config, 'field_size') else 10)
        self.tier_classifier = nn.Linear(config.hidden_size, config.tier_size if hasattr(config, 'tier_size') else 3)
        self.subfield_classifier = nn.Linear(config.hidden_size, config.subfield_size if hasattr(config, 'subfield_size') else 10)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_attentions_weights=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]  # 使用[CLS]标记的输出
        pooled_output = self.dropout(pooled_output)

        return {
            'field_logits': self.field_classifier(pooled_output),
            'tier_logits': self.tier_classifier(pooled_output),
            'subfield_logits': self.subfield_classifier(pooled_output)
        }