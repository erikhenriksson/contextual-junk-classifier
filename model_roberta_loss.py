import torch
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, XLMRobertaModel
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


class ContextualLossXLMRobertaForSequenceClassification(
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

        loss = None

        if target_mask is not None:
            # Ensure target_mask is of shape (batch_size, seq_length)
            target_mask = target_mask.to(sequence_output.device).bool()

            # Get indices of target tokens
            target_indices = target_mask.nonzero(
                as_tuple=False
            )  # Shape: (num_target_tokens, 2)

            # Extract the hidden states of the target tokens
            target_hidden_states = sequence_output[
                target_indices[:, 0], target_indices[:, 1], :
            ]  # Shape: (num_target_tokens, hidden_size)

            # Pass through the classifier
            logits = self.classifier(
                target_hidden_states
            )  # Shape: (num_target_tokens, num_labels)

            # Get labels per token (repeat labels for each target token)
            if labels is not None:
                labels_per_token = labels[
                    target_indices[:, 0]
                ]  # Shape: (num_target_tokens,)

                # Compute loss
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    else:
                        self.config.problem_type = "single_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels_per_token.float())
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels_per_token.view(-1)
                    )
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels_per_token.float())

        else:
            # Default behavior: use the representation of the <s> token
            pooled_output = sequence_output[:, 0, :]  # Shape: (batch_size, hidden_size)

            # Pass through the classifier
            logits = self.classifier(pooled_output)

            # Compute loss
            if labels is not None:
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
                    loss = loss_fct(logits, labels.float())

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
