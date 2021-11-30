from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd


from Model.Common import compute_metrics
from Model.FineTurning.DataSet import ClassificationDataset
from Static.Config import MODEL, train_validate_test_split, get_device

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    eval_steps=500,
    gradient_accumulation_steps=1000,
    eval_accumulation_steps=1
)

def model_init(params):
    db_config = db_config_base
    if params is not None:
        db_config.update({'dropout': params['dropout']})
    return BertForSequenceClassification.from_pretrained(return_dict=True)

def hp_space_ray(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 16, 24, 32]),
        "dropout" : tune.uniform(0,1)
    }

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=cc_train_dataset,
    eval_dataset=cc_val_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics
)

best_trial = trainer.hyperparameter_search(
    hp_space=hp_space_ray,
    direction="maximize",
    backend="ray",
    n_trials=1,
    resources_per_trial={"gpu": 1}