from transformers import DebertaV2PreTrainedModel, DebertaV2Model
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler


class ContextualDebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        target_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # Shape: (batch_size, seq_length, hidden_size)

        # Apply target_mask if provided
        if target_mask is not None:
            # Ensure target_mask is of shape (batch_size, seq_length)
            target_mask = target_mask.to(sequence_output.device)
            target_mask_expanded = (
                target_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            )
            # Mask the sequence output
            target_hidden_states = sequence_output * target_mask_expanded

            # Perform pooling over target tokens
            sum_hidden = torch.sum(target_hidden_states, dim=1)
            count_nonzero = target_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9).float()
            pooled_output = sum_hidden / count_nonzero
            pooled_output = self.dropout(pooled_output)
        else:
            # Default behavior: use the pooler
            pooled_output = self.pooler(sequence_output)
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fct(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                # For backward compatibility
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
