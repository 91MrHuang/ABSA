import os
import configparser
import mlflow.pytorch
import json
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn import metrics
import numpy as np
from transformers import pipeline


BASE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
CONFIG_FILE = BASE_PATH + 'app.cfg'
config = configparser.ConfigParser()
config.read(CONFIG_FILE)


class Aspect_Extraction(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.pretrained_bert_model_name = 'bert-base-uncased'
        self.pretrained_bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_model_name)
        self.pretrained_bert_model = None
        self.token2id, self.id2token, self.tag2id, self.id2tag = None, None, None, None
        self.train_datasets, self.validation_datasets = None, None


    def load_context(self, context):
        finetuned_tokenizer = AutoTokenizer.from_pretrained(context.artifacts["transformer_model_path"])
        finetuned_model = AutoModelForTokenClassification.from_pretrained(context.artifacts["transformer_model_path"],
                                                                num_labels=len(self.tag2id.keys()))
        self.ae_pipeline = pipeline('ner', model=finetuned_model, tokenizer=finetuned_tokenizer)



    def get_mapping(self, data, column):
        vocab = list(set(data[column].to_list()))

        id2t = {i: t for i, t in enumerate(vocab)}
        t2id = {t: i for i, t in enumerate(vocab)}
        return t2id, id2t

    def align_ner_labels(self, samples):
        inputs = self.pretrained_bert_tokenizer(samples['sentence'], truncation=True, padding='max_length',
                                                max_length=128, is_split_into_words=True)

        label_input = []
        for i, label in enumerate(samples['label']):
            word_ids = inputs.word_ids(batch_index=i)

            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(self.tag2id[label[word_idx]])

            if i == 0:
                print(i)
                print(label)
                print(word_ids)
                print(label_ids)

            label_input.append(label_ids)

        inputs['labels'] = label_input

        # print('inputs', inputs.keys())

        return inputs

    def prepare_training_data(self, training_file):
        training_samples = json.load(open(training_file))
        print(f'loaded {len(training_samples)} samples')
        dict_training_samples = {'sentence_id': [], 'sentence': [], 'label': []}

        for key in training_samples.keys():
            dict_training_samples['sentence_id'].append(key)
            dict_training_samples['sentence'].append(training_samples[key]['sentence'])
            dict_training_samples['label'].append(training_samples[key]['label'])

        df_training_samples = pd.DataFrame.from_dict(dict_training_samples)
        print(df_training_samples)

        df_training_samples_exploded = df_training_samples.explode(['sentence', 'label'])
        print(df_training_samples_exploded)



        self.token2id, self.id2token = self.get_mapping(df_training_samples_exploded, 'sentence')
        self.tag2id, self.id2tag = self.get_mapping(df_training_samples_exploded, 'label')

        print(list(self.token2id.items())[0:5], list(self.id2token.items())[0:5])
        print(self.tag2id, self.id2tag)

        df_training_samples_exploded['word_id'] = df_training_samples_exploded['sentence'].map(
            self.token2id)
        df_training_samples_exploded['tag_id'] = df_training_samples_exploded['label'].map(self.tag2id)
        print(df_training_samples_exploded.head(20))

        df_training_samples = df_training_samples_exploded.groupby(['sentence_id'])[
            'sentence', 'label', 'word_id', 'tag_id'].agg(lambda x: list(x))
        print(df_training_samples)

        self.pretrained_bert_model = AutoModelForTokenClassification.from_pretrained(self.pretrained_bert_model_name, num_labels=len(self.tag2id.keys()),
                                                        id2label=self.id2tag, label2id=self.tag2id)

        self.train_datasets = Dataset.from_pandas(df_training_samples[:1600]).map(self.align_ner_labels, batched=True,
                                                                                      batch_size=400).remove_columns(
            ['sentence', 'label', 'word_id', 'tag_id', 'sentence_id'])
        self.validation_datasets = Dataset.from_pandas(df_training_samples[1600:]).map(self.align_ner_labels,
                                                                                           batched=True,
                                                                                           batch_size=50).remove_columns(
            ['sentence', 'label', 'word_id', 'tag_id', 'sentence_id'])

        print(self.train_datasets)
        print(self.validation_datasets)




    def train_and_log(self, local):

        def my_compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)  # axis 0 is sample dim, axis 1 is token dim, axis 2 is class dim

            refined_predictions, refined_labels = [], []
            for label, pred in zip(labels.flatten(), predictions.flatten()):
                if label != -100:
                    refined_labels.append(label)
                    refined_predictions.append(pred)
            print(refined_labels)
            print(refined_predictions)
            accuracy = metrics.accuracy_score(refined_labels, refined_predictions)
            mlflow.log_metric('validation_accuracy', accuracy)
            prfs = metrics.precision_recall_fscore_support(refined_labels, refined_predictions)
            for i in range(0, len(prfs[0])):
                mlflow.log_metric(f'precision_{self.id2tag[i]}', prfs[0][i])
            for i in range(0, len(prfs[1])):
                mlflow.log_metric(f'recall_{self.id2tag[i]}', prfs[1][i])
            for i in range(0, len(prfs[2])):
                mlflow.log_metric(f'f1_{self.id2tag[i]}', prfs[2][i])
            for i in range(0, len(prfs[3])):
                mlflow.log_metric(f'support_{self.id2tag[i]}', prfs[3][i])
            return {'validation_accuracy': accuracy}


        args = TrainingArguments(
            output_dir=config['general']['model_output_dir'],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            logging_strategy='epoch',
            evaluation_strategy='epoch',
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


        with mlflow.start_run(run_name=config['mlflow']['ae_run_name']) as run:
            # mlflow.pytorch.autolog(log_models=True)  # work only for lighting pytorch instead of transformers?

            mlflow.log_param('per_device_train_batch_size', args.per_device_train_batch_size)
            mlflow.log_param('per_device_eval_batch_size', args.per_device_eval_batch_size)
            mlflow.log_param('num_train_epochs', args.num_train_epochs)

            trainer.train()  # train and log metrics
            trainer.save_model()

            self.pretrained_bert_tokenizer = None
            self.pretrained_bert_model = None

            if local:
                artifacts = {'transformer_model_path': config['general']['model_output_dir']}
                mlflow.pyfunc.save_model(path=config['mlflow']['local_mlflow_artifact_dir'],
                                         conda_env=os.path.join('conda.yaml'),
                                         python_model=self,
                                         artifacts=artifacts,
                                         code_path=[os.path.join('aspect_extraction.py')])
            else:
                artifacts = {'transformer_model_path': config['general']['model_output_dir']}
                mlflow.pyfunc.log_model(artifact_path='aspect-extraction',
                                        conda_env=os.path.join('conda.yaml'),
                                        python_model=self,
                                        artifacts=artifacts,
                                        code_path=[os.path.join('aspect_extraction.py')])

            mlflow.end_run()


    def predict(self, context, model_input):
        sentences = list(model_input['sentences'])
        return self.ae_pipeline(sentences)


if __name__ == '__main__':
    ae = Aspect_Extraction()

    rest_ner_training_file = '/Users/sean.huang/Documents/my/books and courses/ztgg/Shared Code/BERT-for-ABSA/ae/rest/train.json'
    ae.prepare_training_data(rest_ner_training_file)
    ae.train_and_log(local=False)

