import logging
import os
import sys

spliter = '\\'

current_path = os.getcwd()
current_path_list = current_path.split(spliter)

for idx, file in enumerate(current_path_list):
    if file == 'src':
        break
sys.path.insert(0, spliter.join(current_path_list[:idx+1]))


from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from retriever.dataset import TrainDataset
from retriever.collator import TrainCollator
from retriever.modeling import EncoderModel as DenseModel
from retriever.trainer import TevatronTrainer as Trainer

logger = logging.getLogger(__name__)

model_to_train = 'jmvcoelho/pythia-160m-1024-marco-docs-bow-contrastive-pretrain'
trained_model_save_path = './data/model'

if not os.path.exists(trained_model_save_path):
    os.makedirs(trained_model_save_path)

trained_model_name = 'pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft'
training_data = 'Tevatron/msmarco-passage-aug'


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # sys.argv = [
    #     "train.py",  # The script name (simulated)
    #     "--model_name_or_path", "bert-base-uncased",
    #     "--config_name", "bert-config.json",
    #     "--dataset_path", "data/dataset.csv",
    #     "--output_dir", "results/"
    # ]

    sys.argv = [
        "train.py",
          "--output_dir", f"{trained_model_save_path}/{trained_model_name}",
          "--model_name_or_path", model_to_train,
          "--dataset_name", training_data,
          "--save_steps", "0",
          "--temperature", "0.01",
          "--per_device_train_batch_size", "128",
          "--train_group_size", "10",
          "--learning_rate", "1e-4",
          "--query_max_len", "32",
          "--passage_max_len", "128",
          "--num_train_epochs", "1",
          "--logging_steps", "1",
          "--gradient_accumulation_steps", "2",
          "--run_name", trained_model_name
    ]


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file = os.path.abspath(sys.argv[1])
        )
        
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir = model_args.cache_dir,
        attn_implementation = "flash_attention_2"
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        data_collator = collator
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()