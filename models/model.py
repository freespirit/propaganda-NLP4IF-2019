from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import tqdm

from keras.preprocessing.sequence import pad_sequences

from pytorch_transformers import BertTokenizer, BertModel,\
    BertForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler


COLUMN_TEXT = "text"
COLUMN_LABEL = "label"

EPOCHS = 5
BATCH_SIZE = 32


class Model:
    def __init__(self, model: torch.nn.Module = None):
        if model is not None:
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-cased', num_labels=2)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                       do_lower_case=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Device used: {}".format(self.device))

    @staticmethod
    def load(file_from) -> 'Model':
        # TODO actually load anything from the file, if recognized
        return Model()

    @staticmethod
    def save(model: 'Model', file_to):
        pass

    def train_slc(self, data: Sequence[Tuple[str, int]]):
        """ Trains a model to recognize propaganda sentences

        :type data: list of tuples - (sentence , propaganda_technique)
        """
        head = data[0]
        assert isinstance(head[0], str)
        assert isinstance(head[1], int)

        df = pd.DataFrame(data=data, columns=[COLUMN_TEXT, COLUMN_LABEL])
        print("Training with {} samples".format(len(df.index)))
        print(df.head())

        tokens_tensor = self.__tokenize_texts(df[COLUMN_TEXT].values)
        labels_tensor = torch.tensor(df[COLUMN_LABEL].values)

        dataset = TensorDataset(tokens_tensor, labels_tensor)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dev_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        test_dataloader = DataLoader(test_dataset,
                                     sampler=SequentialSampler(test_dataset),
                                     batch_size=BATCH_SIZE)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        optimizer = AdamW(self.model.parameters(), lr=1e-5, correct_bias=False)
        train_losses = []

        for epoch in range(EPOCHS):
            print("EPOCH {}/{}".format(epoch+1, EPOCHS))
            train_size = int(0.9 * len(train_dev_dataset))
            validation_size = len(train_dev_dataset) - train_size
            train_dataset, validation_dataset = torch.utils.data.random_split(
                train_dev_dataset, [train_size, validation_size])

            train_dataloader = DataLoader(train_dataset,
                                          sampler=RandomSampler(train_dataset),
                                          batch_size=BATCH_SIZE)
            validation_dataloader = DataLoader(validation_dataset,
                                               sampler=SequentialSampler(
                                                   validation_dataset),
                                               batch_size=BATCH_SIZE)

            training_loss = 0
            training_steps = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                input_tensor, labels_tensor = tuple(t.to(self.device)
                                                    for t in batch)
                outputs = self.model(input_tensor, labels=labels_tensor)
                loss = outputs[0]

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                training_loss += loss.item()
                training_steps += 1

                if step % 100 == 0:
                    print("Train loss at step {step}: {loss:.3f}".format(
                        step=step, loss=(training_loss / training_steps)))

            print("Train loss: {0:.3f}".format(training_loss / training_steps))

            self.model.eval()
            validation_accuracy, validation_steps = 0, 0
            for step, batch in enumerate(validation_dataloader):
                inputs, labels = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    logits = outputs[0]

                    _, indices = torch.max(logits, dim=1)

                validation_accuracy += sklearn.metrics.accuracy_score(
                    labels.detach().cpu().numpy(),
                    indices.detach().cpu().numpy())
                validation_steps += 1

            print("Validation accuracy: {0:.3f}".format(validation_accuracy / validation_steps))

        plt.figure(figsize=(20, 10))
        plt.title("Training loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(train_losses)
        plt.savefig("outputs/losses.png", bbox_inches='tight')
        # plt.show()
        # plt.close(fig)

        test_labels = []
        test_predictions = []
        self.model.eval()
        for batch in test_dataloader:
            inputs, labels = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs[0]

                _, indices = torch.max(logits, dim=1)

            test_predictions.extend(indices.detach().cpu().numpy())
            test_labels.extend(labels.detach().cpu().numpy())

        precision, recall, f1_score, _ = \
            sklearn.metrics.precision_recall_fscore_support(test_labels,
                                                            test_predictions,
                                                            average="macro")
        print("Test Labels vs Predictions = {} : {}".format(len(test_labels),
                                                            len(test_predictions)))
        print("Test"
              "\n\taccuracy: {:.6f}"
              "\n\tprecision: {:.6f}"
              "\n\trecall: {:.6f}"
              "\n\tf1: {:.6f}".format(
                                sklearn.metrics.accuracy_score(test_labels, test_predictions),
                                precision, recall, f1_score))

        baseline_predictions = np.ones_like(test_labels)
        print("Baseline"
              "\n\taccuracy: {:.6f}"
              "\n\tprecision: {:.6f}"
              "\n\trecall: {:.6f}"
              "\n\tf1: {:.6f}".format(
                            sklearn.metrics.accuracy_score(test_labels, baseline_predictions),
                            sklearn.metrics.precision_score(test_labels, baseline_predictions),
                            sklearn.metrics.recall_score(test_labels, baseline_predictions),
                            sklearn.metrics.f1_score(test_labels, baseline_predictions)))

    def __tokenize_texts(self, texts: np.ndarray):
        tokenizer = self.tokenizer

        # wrap with the BERT [CLS] and [SEP] tokens
        def format_fn(sentence: Sequence[str]) -> Sequence[str]:
            return "{} {} {}".format(
                tokenizer.cls_token, sentence, tokenizer.sep_token)

        texts = [format_fn(sentence) for sentence in texts]
        tokenized_text = [tokenizer.tokenize(t) for t in texts]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        indexed_tokens = pad_sequences(indexed_tokens, maxlen=150, dtype="long",
                                       padding="post", truncating="post",
                                       value=tokenizer.convert_tokens_to_ids(
                                           tokenizer.pad_token))
        indexed_tokens = [torch.tensor(x) for x in indexed_tokens]

        tokens_tensor = torch.nn.utils.rnn.pad_sequence(
            indexed_tokens,
            batch_first=True,
            padding_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))

        return tokens_tensor

    def predict_slc(self, sentences: np.ndarray) -> Sequence[int]:
        result = list()

        tokens_tensor = self.__tokenize_texts(sentences)
        predict_dataset = TensorDataset(tokens_tensor)
        predict_dataloader = DataLoader(predict_dataset,
                                        batch_size=BATCH_SIZE)
        self.model.eval()
        for i, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(batch[0])
                logits = outputs[0]
                _, indices = torch.max(logits, dim=1)
                indices = indices.detach().cpu().numpy()
            result.extend(indices)

        assert len(result) == len(sentences)

        return result

    def predict_flc(self, articles):
        return [article for article in articles]