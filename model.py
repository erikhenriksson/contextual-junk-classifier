from transformers import XLMRobertaForSequenceClassification, XLMRobertaModel
import torch
import torch.nn as nn

from transformers.modeling_outputs import SequenceClassifierOutput


class CustomClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # x is of shape (batch_size, hidden_size)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContextualXLMRobertaForSequenceClassification(
    XLMRobertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = CustomClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        target_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Ensure return_dict is set
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Pass inputs through the base model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # XLM-RoBERTa doesn't use token_type_ids
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the sequence output
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
        else:
            # Default behavior: use the representation of the <s> token
            pooled_output = sequence_output[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Pass through the classifier
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Use existing loss computation logic
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return the standard output format
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
