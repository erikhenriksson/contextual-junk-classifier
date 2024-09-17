import torch
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, XLMRobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ContextualXLMRobertaForSequenceClassification(
    XLMRobertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Use the default roberta and classifier modules
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # The default classification head is already defined as self.classifier

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
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the sequence output
        sequence_output = outputs[0]  # Shape: (batch_size, seq_length, hidden_size)

        if target_mask is not None:
            # Ensure target_mask is of shape (batch_size, seq_length)
            target_mask = target_mask.to(sequence_output.device).unsqueeze(
                -1
            )  # Shape: (batch_size, seq_length, 1)

            # Mask the sequence output to keep only target token representations
            target_hidden_states = (
                sequence_output * target_mask.float()
            )  # Shape: (batch_size, seq_length, hidden_size)

            # Sum over the sequence length dimension to aggregate target tokens
            target_sum = target_hidden_states.sum(
                dim=1
            )  # Shape: (batch_size, hidden_size)

            # Count the number of target tokens per example to compute mean pooling
            target_count = target_mask.sum(dim=1).clamp(
                min=1e-9
            )  # Shape: (batch_size, 1)

            # Compute the mean of target token representations
            pooled_output = (
                target_sum / target_count
            )  # Shape: (batch_size, hidden_size)
        else:
            # Default behavior: use the representation of the <s> token
            pooled_output = sequence_output[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Pass the pooled output through the classifier
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        # Initialize loss to None
        loss = None

        # Compute loss if labels are provided
        if labels is not None:
            # Determine problem type if not specified
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # Choose appropriate loss function
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
        # Return outputs
        if not return_dict:
            output = (logits,) + outputs[
                2:
            ]  # Skip hidden_states and attentions if not requested
            return ((loss,) + output) if loss is not None else output

        # Return the standard output format
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
