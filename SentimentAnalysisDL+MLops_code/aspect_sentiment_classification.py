import os
import configparser
import mlflow.pytorch
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn import metrics
import numpy as np


BASE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
CONFIG_FILE = BASE_PATH + 'app.cfg'
config = configparser.ConfigParser()
config.read(CONFIG_FILE)


class Aspect_Sentiment_Classification(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.pretrained_bert_model_name = 'bert-base-uncased'
        self.pretrained_bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_model_name)
        self.pretrained_bert_model = None
        self.train_datasets, self.validation_datasets = None, None


    def load_context(self, context):
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(context.artifacts["transformer_model_path"])
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["transformer_model_path"],
                                                                num_labels=3)
        self.sentiment_labels = self.finetuned_model.config.id2label


    def prepare_asc_classification_labels(self, samples):
        inputs = self.pretrained_bert_tokenizer(samples['sentence'], samples['term'], truncation=True, padding='max_length',
                                           max_length=128, is_split_into_words=False)

        # print(inputs)

        return inputs


    def prepare_training_data(self, training_file):
        training_samples = json.load(open(training_file))
        print(f'loaded {len(training_samples)} samples')
        dict_training_samples = {'sample_id': [], 'sentence': [], 'term': [], 'label': []}

        for key in training_samples.keys():
            dict_training_samples['sample_id'].append(key)
            dict_training_samples['sentence'].append(training_samples[key]['sentence'])
            dict_training_samples['term'].append(training_samples[key]['term'])
            dict_training_samples['label'].append(0 if training_samples[key]['polarity'] == 'negative' else 1 if training_samples[key]['polarity'] == 'positive' else 2)

        df_training_samples = pd.DataFrame.from_dict(dict_training_samples)
        print(df_training_samples)

        self.pretrained_bert_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_bert_model_name, num_labels=3,
                                                        id2label={0:'negative', 1:'positive', 2:'neutral'}, label2id={'negative':0, 'positive':1, 'neutral':2})

        self.train_datasets = Dataset.from_pandas(df_training_samples[:3000]).map(self.prepare_asc_classification_labels,
            batched=True, batch_size=1000).remove_columns(['sample_id', 'sentence', 'term'])
        self.validation_datasets = Dataset.from_pandas(df_training_samples[3000:]).map(self.prepare_asc_classification_labels,
            batched=True, batch_size=100).remove_columns(['sample_id', 'sentence', 'term'])


    def train_and_log(self, local):

        def my_compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)  # axis 0 is sample dim, axis 1 is token dim, axis 2 is class dim

            # refined_predictions, refined_labels = [], []
            # for label, pred in zip(labels.flatten(), predictions.flatten()):
            #     if label != -100:
            #         refined_labels.append(label)
            #         refined_predictions.append(pred)
            # print(refined_labels)
            # print(refined_predictions)
            id2label = {0:'negative', 1:'positive', 2:'neutral'}
            accuracy = metrics.accuracy_score(labels, predictions)
            mlflow.log_metric('validation_accuracy', accuracy)
            prfs = metrics.precision_recall_fscore_support(labels, predictions)
            for i in range(0, len(prfs[0])):
                mlflow.log_metric(f'precision_{id2label[i]}', prfs[0][i])
            for i in range(0, len(prfs[1])):
                mlflow.log_metric(f'recall_{id2label[i]}', prfs[1][i])
            for i in range(0, len(prfs[2])):
                mlflow.log_metric(f'f1_{id2label[i]}', prfs[2][i])
            for i in range(0, len(prfs[3])):
                mlflow.log_metric(f'support_{id2label[i]}', prfs[3][i])
            return {'validation_accuracy': accuracy}


        args = TrainingArguments(
            output_dir=config['general']['asc_model_output_dir'],
            per_device_train_batch_size=128,
            per_device_eval_batch_size=16,
            num_train_epochs=20,
            logging_strategy='steps',
            logging_steps=25,
            evaluation_strategy='steps',
            eval_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            report_to='none'
        )

        trainer = Trainer(
            model=self.pretrained_bert_model,
            tokenizer=self.pretrained_bert_tokenizer,
            args=args,
            train_dataset=self.train_datasets,
            eval_dataset=self.validation_datasets,
            compute_metrics=my_compute_metrics
        )

        # mlflow logging of hyper-parameters, metrics and models
        mlflow.set_tracking_uri(config['mlflow']['mlflow_url'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name=config['mlflow']['asc_run_name']) as run:
            # mlflow.pytorch.autolog(log_models=True)  # work only for lighting pytorch instead of transformers?

            mlflow.log_param('per_device_train_batch_size', args.per_device_train_batch_size)
            mlflow.log_param('per_device_eval_batch_size', args.per_device_eval_batch_size)
            mlflow.log_param('num_train_epochs', args.num_train_epochs)

            trainer.train()  # train and log metrics
            trainer.save_model()

            self.pretrained_bert_tokenizer = None
            self.pretrained_bert_model = None

            if local:
                artifacts = {'transformer_model_path': config['general']['asc_model_output_dir']}
                mlflow.pyfunc.save_model(path=config['mlflow']['asc_local_mlflow_artifact_dir'],
                                         conda_env=os.path.join('conda.yaml'),
                                         python_model=self,
                                         artifacts=artifacts,
                                         code_path=[os.path.join('aspect_sentiment_classification.py')])
            else:
                artifacts = {'transformer_model_path': config['general']['asc_model_output_dir']}
                mlflow.pyfunc.log_model(artifact_path='aspect-sentiment-classification',
                                        conda_env=os.path.join('conda.yaml'),
                                        python_model=self,
                                        artifacts=artifacts,
                                        code_path=[os.path.join('aspect_sentiment_classification.py')])

            mlflow.end_run()


    def predict(self, context, model_input):
        sentences = list(model_input['sentences'])
        terms = list(model_input['terms'])

        inputs = self.finetuned_tokenizer(sentences, terms, truncation=True, padding='max_length',
                                          max_length=128, is_split_into_words=False, return_tensors='pt')
        outputs = self.finetuned_model(**{k: v for k, v in inputs.items()})
        results = [self.sentiment_labels[np.argmax(each.tolist())] for each in outputs.logits]

        return results



if __name__ == '__main__':
    asc = Aspect_Sentiment_Classification()
    rest_asc_training_file = '/Users/sean.huang/Documents/my/books and courses/ztgg/Shared Code/BERT-for-ABSA/asc/rest/train.json'
    asc.prepare_training_data(rest_asc_training_file)
    asc.train_and_log(local=False)

