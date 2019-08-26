from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import time
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

COLUMN_VALUE = 'value'
COLUMN_METRIC = 'metric'
METRIC_TRAINING_LOSS = 'Training Loss'
METRIC_VALIDATION_F1 = 'Validation F1'

EPOCHS = 5
BATCH_SIZE = 32


class Model(object):
    def __init__(self, model: torch.nn.Module = None):
        if model is not None:
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-cased', num_labels=2)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                       do_lower_case=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print("Found", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(model)

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
        bert_params = [parameter for name, parameter
                       in self.model.named_parameters()
                       if name.startswith("bert") and not name.startswith("bert.embeddings")]
        class_params = [parameter for name, parameter
                        in self.model.named_parameters()
                        if not name.startswith("bert")]

        adam_args = [{'params': bert_params, 'lr': 5e-6, 'weight_decay': 0.01},
                     {'params': class_params, 'lr': 1e-5}]
        optimizer = AdamW(adam_args, correct_bias=False)

        df_metrics = pd.DataFrame()
        for epoch in range(EPOCHS):
            print("EPOCH {}/{}".format(epoch+1, EPOCHS))
            time_start = time.time()
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

            running_loss = 0
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

                running_loss += loss.item()
                training_steps += 1
                if step % 100 == 0:
                    print("Train loss at step {step}: {loss:.3f}".format(
                        step=step, loss=(running_loss / training_steps)))
                batch_loss = pd.DataFrame({COLUMN_VALUE: loss.item(),
                                           COLUMN_METRIC: METRIC_TRAINING_LOSS},
                                          index=[0])
                df_metrics = df_metrics.append(batch_loss, ignore_index=True)

            print("Train loss: {0:.3f}".format(running_loss / training_steps))

            _, (_, _, f1_score) = self.__eval(validation_dataloader, 'Validation')

            validation_f1 = pd.DataFrame({COLUMN_VALUE: f1_score,
                                          COLUMN_METRIC: METRIC_VALIDATION_F1},
                                         index=[0])
            df_metrics = df_metrics.append(validation_f1, ignore_index=True)

            time_end = time.time()
            time_interval = time_end - time_start
            print("time: {min:.0f}:{sec:.0f}".format(min=time_interval / 60,
                                                     sec=time_interval % 60))

        plt.figure(figsize=(20, 10))
        plt.title("Training metrics")
        plt.xlabel("Batch")
        sns.lineplot(x=df_metrics.index, y=COLUMN_VALUE,
                     data=df_metrics, hue=COLUMN_METRIC)
        plt.savefig("outputs/metrics.png", bbox_inches='tight')

        (test_labels, _), _ = self.__eval(test_dataloader, 'Test')

    def __eval(self, dataloader, title='Evaluation results:'):
        labels = []
        predictions = []
        self.model.eval()
        for batch in dataloader:
            batch_inputs, batch_labels = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(batch_inputs)
                logits = outputs[0]

                _, indices = torch.max(logits, dim=1)

            predictions.extend(indices.detach().cpu().numpy())
            labels.extend(batch_labels.detach().cpu().numpy())

        precision, recall, f1_score, _ = \
            sklearn.metrics.precision_recall_fscore_support(labels,
                                                            predictions,
                                                            average="macro")
        print("Labels vs Predictions = {} : {}".format(len(labels),
                                                       len(predictions)))
        print("{}"
              "\n\taccuracy: {:.6f}"
              "\n\tprecision: {:.6f}"
              "\n\trecall: {:.6f}"
              "\n\tf1: {:.6f}".format(
                title,
                sklearn.metrics.accuracy_score(labels, predictions),
                precision, recall, f1_score))

        return (labels, predictions), (precision, recall, f1_score)

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
