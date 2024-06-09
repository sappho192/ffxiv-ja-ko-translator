from typing import Dict, List
import csv

import datasets
import torch
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    BertJapaneseTokenizer,
    Trainer
)
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from datasets import load_dataset
import wandb


class GPT2Tokenizer(PreTrainedTokenizerFast):
    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return token_ids + [self.eos_token_id]


class PairedDataset:
    def __init__(self, 
        source_tokenizer: PreTrainedTokenizerFast, target_tokenizer: PreTrainedTokenizerFast,
        file_path: str = None,
        dataset_raw: datasets.Dataset = None
    ):
        self.src_tokenizer = source_tokenizer
        self.trg_tokenizer = target_tokenizer
        
        if file_path is not None:
            with open(file_path, 'r') as fd:
                reader = csv.reader(fd)
                next(reader)
                self.data = [row for row in reader]
        elif dataset_raw is not None:
            self.data = dataset_raw
        else:
            raise ValueError('file_path or dataset_raw must be specified')

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
#         with open('train_log.txt', 'a+') as log_file:
#             log_file.write(f'reading data[{index}] {self.data[index]}\n')
        if isinstance(self.data, datasets.Dataset):
            src, trg = self.data[index]['sourceString'], self.data[index]['targetString']
        else:
            src, trg = self.data[index]
        embeddings = self.src_tokenizer(src, return_attention_mask=False, return_token_type_ids=False)
        embeddings['labels'] = self.trg_tokenizer.build_inputs_with_special_tokens(self.trg_tokenizer(trg, return_attention_mask=False)['input_ids'])

        return embeddings

    def __len__(self):
        return len(self.data)


def main():
    encoder_model_name = "cl-tohoku/bert-base-japanese-v3"
    decoder_model_name = "skt/kogpt2-base-v2"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    device, torch.cuda.device_count()

    src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
    trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_model_name, bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

    # dataset = load_dataset("sappho192/Tatoeba-Challenge-jpn-kor")
    dataset = load_dataset("/dataset/Tatoeba-Challenge-jpn-kor")

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_first_row = train_dataset[0]
    test_first_row = test_dataset[0]

    train_dataset = PairedDataset(src_tokenizer, trg_tokenizer, dataset_raw=train_dataset)
    eval_dataset = PairedDataset(src_tokenizer, trg_tokenizer, dataset_raw=test_dataset)

    ## Training section
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model_name,
        decoder_model_name,
        pad_token_id=trg_tokenizer.bos_token_id,
    )
    model.config.decoder_start_token_id = trg_tokenizer.bos_token_id

    collate_fn = DataCollatorForSeq2Seq(src_tokenizer, model)
    wandb.init(project="fftr-poc1", name='jbert+kogpt2')

    arguments = Seq2SeqTrainingArguments(
        output_dir='dump',
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        # num_train_epochs=25,
        per_device_train_batch_size=1,
        # per_device_train_batch_size=30, # takes 40GB
        # per_device_train_batch_size=64,
        per_device_eval_batch_size=1,
        # per_device_eval_batch_size=30,
        # per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        save_total_limit=5,
        dataloader_num_workers=1,
        # fp16=True, # ENABLE if CUDA is enabled
        load_best_model_at_end=True,
        report_to='wandb'
    )

    trainer = Trainer(
        model,
        arguments,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    model.save_pretrained("dump/best_model")
    src_tokenizer.save_pretrained("dump/best_model/src_tokenizer")
    trg_tokenizer.save_pretrained("dump/best_model/trg_tokenizer")


if __name__ == "__main__":
    main()
