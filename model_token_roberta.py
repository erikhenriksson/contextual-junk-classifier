from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn


class CustomXLMRobertaClassifier(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        head_mask: torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        target_mask: torch.LongTensor = None,
    ) -> SequenceClassifierOutput:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Apply target mask
        masked_logits = logits * target_mask.unsqueeze(-1).float()

        # Pool the logits for target tokens
        pooled_logits = (
            masked_logits.sum(dim=1) / target_mask.sum(dim=1, keepdim=True).float()
        )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (pooled_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
