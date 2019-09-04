from models.bert_for_propaganda import BertForPropaganda
from sklearn.utils import deprecated
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

from pytorch_transformers import\
    BertTokenizer, BertModel, BertForSequenceClassification

from pytorch_transformers import AdamW,\
    WarmupLinearSchedule, WarmupConstantSchedule, WarmupCosineSchedule

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler
from torch.utils.tensorboard import SummaryWriter


BERT_BASE_CASED = 'bert-base-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_CASED = 'bert-large-cased'
BERT_LARGE_UNCASED = 'bert-large-uncased'

BERT_VARIANT = BERT_BASE_UNCASED

COLUMN_TEXT = "text"
COLUMN_LABEL = "label"
COLUMN_TECHNIQUES = "techniques"
COLUMN_VALUE = 'value'
COLUMN_METRIC = 'metric'
METRIC_TRAINING_LOSS = 'Training Loss'
METRIC_VALIDATION_F1 = 'Validation F1'

TRAIN_TEST_DATA_RATIO = 0.95
TRAIN_DEV_DATA_RATIO = 0.95

MAX_SEQUENCE_LEN = 128
EPOCHS = 3
BATCH_SIZE = 32
LR_BASE = 4e-6
LR_LARGE = 3e-6


class Model(object):
    def __init__(self):
        do_lower_case = BERT_VARIANT is BERT_BASE_UNCASED \
                        or BERT_VARIANT is BERT_LARGE_UNCASED

        self.model = BertForPropaganda.from_pretrained(BERT_VARIANT, num_labels=(2, 19))
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VARIANT, do_lower_case=do_lower_case)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Found", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        print("Device used: {}".format(self.device))

        if BERT_VARIANT is BERT_BASE_UNCASED or BERT_VARIANT is BERT_BASE_CASED:
            self.learning_rate = LR_BASE
        else:
            self.learning_rate = LR_LARGE

        self.tb_writer = SummaryWriter()

    @staticmethod
    def load(file_from) -> 'Model':
        # TODO actually load anything from the file, if recognized
        return Model()

    @staticmethod
    def save(model: 'Model', file_to):
        pass

    # noinspection PyTypeChecker
    def train(self, data: Sequence[Tuple[str, int, Sequence[int]]]):
        """ Trains a model to recognize propaganda sentences

        :type data: list of tuples - (sentence , propaganda_technique)
        """
        head = data[0]
        assert isinstance(head[0], str)
        assert isinstance(head[1], int)

        df = pd.DataFrame(data=data, columns=[COLUMN_TEXT,
                                              COLUMN_LABEL,
                                              COLUMN_TECHNIQUES])
        print("Training with {} samples".format(len(df.index)))
        print(df.head())

        tokens_tensor = self.tokenize_texts(df[COLUMN_TEXT].values)
        labels_tensor = torch.tensor(df[COLUMN_LABEL].values)
        techniques_tensor = self.__prepare_technique_labels(df[COLUMN_TECHNIQUES].values)
        #TODO also use attention mask

        dataset = TensorDataset(tokens_tensor, labels_tensor, techniques_tensor)

        train_set_size = int(TRAIN_TEST_DATA_RATIO * len(dataset))
        test_set_size = len(dataset) - train_set_size
        train_dev_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_set_size, test_set_size])

        total_train_steps = int(EPOCHS
                                * len(train_dev_dataset) / BATCH_SIZE
                                * TRAIN_DEV_DATA_RATIO)
        warmup_train_steps = total_train_steps * 0.2
        adam_args = self.make_recommended_params()
        optimizer = AdamW(adam_args, lr=self.learning_rate, correct_bias=False)
        scheduler = WarmupConstantSchedule(optimizer, warmup_train_steps)

        df_metrics = pd.DataFrame()

        iteration = 0
        for epoch in range(EPOCHS):
            print("EPOCH {}/{}".format(epoch+1, EPOCHS))
            time_start = time.time()
            train_set_size = int(TRAIN_DEV_DATA_RATIO * len(train_dev_dataset))
            validation_size = len(train_dev_dataset) - train_set_size
            train_dataset, validation_dataset = torch.utils.data.random_split(
                train_dev_dataset, [train_set_size, validation_size])

            train_dataloader = DataLoader(train_dataset,
                                          sampler=RandomSampler(train_dataset),
                                          batch_size=BATCH_SIZE)

            running_loss = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)
                input_tensor, labels_tensor, techniques_tensor = batch
                assert len(labels_tensor) == len(techniques_tensor)
                outputs = self.model(input_tensor, labels=(labels_tensor, techniques_tensor))
                class_loss = outputs[0]
                tokens_loss = outputs[1]

                class_loss.backward(retain_graph=True)
                tokens_loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += class_loss.item()
                self.__report_training_loss(class_loss.item(), tokens_loss.item(),
                                            running_loss, step + 1, iteration)
                iteration += 1

                batch_loss = pd.DataFrame({COLUMN_VALUE: class_loss.item(),
                                           COLUMN_METRIC: METRIC_TRAINING_LOSS},
                                          index=[0])
                df_metrics = df_metrics.append(batch_loss, ignore_index=True)

            f1_score = self.__report_validation_metrics(epoch, validation_dataset)
            validation_f1 = pd.DataFrame({COLUMN_VALUE: f1_score,
                                          COLUMN_METRIC: METRIC_VALIDATION_F1},
                                         index=[0])
            df_metrics = df_metrics.append(validation_f1, ignore_index=True)

            time_end = time.time()
            time_interval = time_end - time_start
            print("time: {min:02.0f}:{sec:02.0f}".format(min=time_interval / 60,
                                                         sec=time_interval % 60))

        self.__plot_validation_metrics(df_metrics)
        self.__report_test_results(test_dataset)

    def make_recommended_params(self):
        parameters = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay = ['bias', 'gamma', 'beta'] #see mccormickml blog
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        return [
            {'params': [param for name, param in parameters
                        if not any(nd in name for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [param for name, param in parameters
                        if any(nd in name for nd in no_decay)],
             'weight_decay': 0.0}]

    @deprecated("This is not currently used but may be worth "
                "leaving it here for experiments. "
                "Use `make_recommended_params` instead")
    def make_params_dict(self):
        named_parameters = self.model.named_parameters()

        params_encoder_layer_11_10 = [parameter for name, parameter
                                      in list(named_parameters)
                                      if name.startswith("bert.encoder.layer.11")
                                      or name.startswith("bert.encoder.layer.10")]

        params_encoder_layer_9_8_7 = [parameter for name, parameter
                                      in list(named_parameters)
                                      if name.startswith("bert.encoder.layer.9")
                                      or name.startswith("bert.encoder.layer.8")
                                      or name.startswith("bert.encoder.layer.7")]

        params_middle_layers = [parameter for name, parameter
                                in list(named_parameters)
                                if name.startswith("bert.encoder.layer.6")
                                or name.startswith("bert.encoder.layer.5")
                                or name.startswith("bert.encoder.layer.4")
                                or name.startswith("bert.encoder.layer.3")
                                or name.startswith("bert.encoder.layer.2")]

        params_base_layers = [parameter for name, parameter
                              in list(named_parameters)
                              if name.startswith("bert.encoder.layer.1")
                              or name.startswith("bert.embeddings")]

        # params_embeddings = [parameter for name, parameter
        #                      in named_parameters
        #                      if name.startswith("bert.embeddings")]

        class_params = [parameter for name, parameter
                        in list(named_parameters)
                        if name.startswith("classifier")]

        return [
            {'params': params_encoder_layer_11_10, 'lr': 1e-5},
            {'params': params_encoder_layer_9_8_7, 'lr': 5e-6},
            # {'params': params_base_layers, 'lr': 1e-6, 'weight_decay': 0.001},
            # {'params': params_embeddings, 'lr': 1e-7, 'weight_decay': 0.00001},
            {'params': params_middle_layers, 'lr': 1e-6},
            {'params': params_base_layers, 'lr': 1e-7},
            {'params': class_params, 'lr': 2e-5}]

    def __eval(self, dataloader, title='Evaluation results:'):
        labels = []
        predictions = []
        self.model.eval()
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            batch_inputs, batch_labels, batch_technique_labels = batch
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

    def tokenize_texts(self, texts: np.ndarray):
        tokenizer = self.tokenizer

        def truncate(sentence: Sequence[str], max_len: int):
            if len(sentence) > max_len:
                while len(sentence) > max_len:
                    sentence.pop()
            return sentence

        # wrap with the BERT [CLS] and [SEP] tokens
        def wrap(sentence: Sequence[str]) -> Sequence[str]:
            return [tokenizer.cls_token] + sentence + [tokenizer.sep_token]

        tokenized_texts = [tokenizer.tokenize(t) for t in texts]
        tokenized_texts = [truncate(sentence, MAX_SEQUENCE_LEN - 2)
                           for sentence in tokenized_texts]
        tokenized_texts = [wrap(sentence) for sentence in tokenized_texts]

        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokens)
                          for tokens in tokenized_texts]
        indexed_tokens = self.__pad_sequences(
            indexed_tokens,
            tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        indexed_tensors = [torch.tensor(x) for x in indexed_tokens]

        tokens_tensor = self.__pad_tensors(indexed_tensors)

        return tokens_tensor

    @staticmethod
    def __pad_sequences(sequences, pad_token):
        return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN,
                             padding="post", truncating="post",
                             dtype="long", value=pad_token)

    def __pad_tensors(self, tensors):
        tokenizer = self.tokenizer
        return torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))

    def __prepare_technique_labels(self, values):
        padded_sequences = self.__pad_sequences(values, 0)
        techniques_tensor = [torch.tensor(x) for x in padded_sequences]
        techniques_tensor = self.__pad_tensors(techniques_tensor)
        return techniques_tensor

    def __report_training_loss(self, sequence_loss, tokens_loss,
                               running_loss, epoch_step, iteration):
        self.tb_writer.add_scalars("Batch loss",
                                   {'sequence': sequence_loss, "tokens": tokens_loss},
                                   iteration)

        if epoch_step % 100 == 0:
            print("Train loss at step {step}: {loss:.3f}".format(
                step=epoch_step, loss=(running_loss / epoch_step)))

    def __report_validation_metrics(self, epoch, validation_dataset):
        validation_dataloader = DataLoader(validation_dataset,
                                           sampler=SequentialSampler(
                                               validation_dataset),
                                           batch_size=BATCH_SIZE)
        _, (precision, recall, f1_score) = self.__eval(validation_dataloader,
                                                       'Validation')
        self.tb_writer.add_scalars("Validation metrics",
                                   {"precision": precision,
                                    "recall": recall,
                                    "f1_score": f1_score},
                                   epoch)
        return f1_score

    @staticmethod
    def __plot_validation_metrics(df_metrics):
        plt.figure(figsize=(20, 10))
        plt.title("Training metrics")
        plt.xlabel("Batch")
        sns.lineplot(x=df_metrics.index, y=COLUMN_VALUE,
                     data=df_metrics, hue=COLUMN_METRIC)
        plt.savefig("outputs/metrics.png", bbox_inches='tight')

    def __report_test_results(self, test_dataset):
        test_dataloader = DataLoader(test_dataset,
                                     sampler=SequentialSampler(test_dataset),
                                     batch_size=BATCH_SIZE)
        (test_labels, _), (precision, recall, f1_score) = \
            self.__eval(test_dataloader, 'Test')

        self.tb_writer.add_scalars("Test metrics",
                                   {"precision": precision,
                                    "recall": recall,
                                    "f1_score": f1_score},
                                   0)

    def predict_slc(self, sentences: np.ndarray) -> Sequence[int]:
        result = list()

        tokens_tensor = self.tokenize_texts(sentences)
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

    def predict_flc(self, sentences):
        predictions = list()

        tokens_tensor = self.tokenize_texts(sentences)
        predict_dataset = TensorDataset(tokens_tensor)
        predict_dataloader = DataLoader(predict_dataset,
                                        batch_size=BATCH_SIZE)
        self.model.eval()
        for i, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = batch[0]
                outputs = self.model(inputs)
                multi_logits = outputs[1]

                _, indices = torch.max(multi_logits, dim=2)
                indices = indices.detach().cpu().numpy()
            predictions.extend(indices)

        assert len(predictions) == len(sentences)

        return tokens_tensor.detach().cpu().numpy(), predictions
