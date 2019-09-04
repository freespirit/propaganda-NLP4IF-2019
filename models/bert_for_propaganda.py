import torch
import torch.nn as nn

from pytorch_transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, BCELoss


class BertForPropaganda(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForPropaganda, self).__init__(config)
        self.num_class_labels, self.num_multi_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_class_labels)

        self.multi_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multi_classifier = nn.Linear(config.hidden_size, self.num_multi_labels)

        self.apply(self.init_weights)

    # noinspection DuplicatedCode
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None,
                position_ids=None, head_mask=None):


        outputs = self.bert(input_ids, position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.multi_dropout(sequence_output)
        multi_logits = self.multi_classifier(sequence_output)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        class_logits = self.classifier(pooled_output)

        outputs = (multi_logits,) + outputs[2:]
        outputs = (class_logits,) + outputs

        if labels is not None:
            class_labels, multi_class_labels = labels

            # tokens multi class loss
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = multi_logits.view(-1, self.num_multi_labels)[
                    active_loss]
                active_labels = multi_class_labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(multi_logits.view(-1, self.num_multi_labels),
                                multi_class_labels.view(-1))
            outputs = (loss,) + outputs

            # sequences classification loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(class_logits.view(-1, self.num_class_labels),
                            class_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), (multi-class-loss), class_logits, multi_class_logits, (hidden_states), (attentions)

