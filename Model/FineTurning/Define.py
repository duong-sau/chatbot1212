from transformers import TrainingArguments

MODEL = {
    'name': 't5-small',
    'data_link': "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/POS/learn_data.csv",
    'num_decoder_layers': 6,
    'num_freeze': 1
}
strategy = 'epoch'
training_args = TrainingArguments(
    output_dir="/content/",
    overwrite_output_dir=True,
    save_strategy=strategy,
    disable_tqdm=False,
    debug="underflow_overflow",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    evaluation_strategy=strategy,
    fp16=False,
    warmup_steps=100,
    learning_rate=5e-3,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=False,
)


def freezeLayer(model):
    for param in model.encoder.parameters():
        param.requires_grad = False


def tokenConfig(tokenizer):
    tokenizer.padding_side = "left"
