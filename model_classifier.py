import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits, labels.long()
            )  # Labels should be of type LongTensor

        return {"loss": loss, "logits": logits}
