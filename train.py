from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import logging
import sys
import argparse
import os
from types import SimpleNamespace
import roberta_model
import utils


def create_parser():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--learning_rate", type=str, default=1e-4)
    parser.add_argument("--train_mode", type=str, default='continue', choices=['from_scratch','continue'])

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", None))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", None))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS", None))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", None))
    return parser


def main():
    parser = create_parser()
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    model = utils.get_default_model()

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        save_steps = args.save_steps,
        save_total_limit = args.save_total_limit,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # train model
    continue_training = args.train_mode == 'continue'
    if continue_training:
        try:
            logger.info(f"attempting to initalize weights from last checkpoint")
            trainer.train(resume_from_checkpoint=True)  
        except ValueError:
            logger.warning(f"no checkpoint found. Training from scratch")
            continue_training = False

    if not continue_training:
        trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)


if __name__ == "__main__":
    main()

