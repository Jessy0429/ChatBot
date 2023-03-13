import json
import os
import pickle
import logging
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from Args import Args

args = Args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    res = [[tokenizer.additional_special_tokens_ids[row["response"][1]]] + tokenizer.encode(row["response"][0]) + [tokenizer.eos_token_id]]
    conv = list(reversed([
        [tokenizer.additional_special_tokens_ids[x[1]]] + tokenizer.encode(x[0]) + [tokenizer.eos_token_id] for x in row["content"]
    ]+res))
    # conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, path, data, type, block_size):
        # block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        cached_features_file = os.path.join(path, 'DialoGPT_cached_'+type+'data')

        if os.path.exists(cached_features_file):
            print("----Loading features from cached file %s----", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.dataset = pickle.load(handle)
        else:
            print("----Creating features from dataset file at %s----", path)

            self.dataset = []
            for row in data:
                conv = construct_conv(row, tokenizer)
                if len(conv) > block_size:
                    continue
                self.dataset.append(conv)

            print("----Saving features into cached file %s----", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return torch.tensor(self.dataset[item], dtype=torch.long)


def load_and_cache_data(data_path, cache_path, tokenizer, block_size):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=args.seed)
    train_dataset = ConversationDataset(tokenizer, cache_path, train_data, 'train', block_size)
    test_dataset = ConversationDataset(tokenizer, cache_path, test_data, 'test', block_size)
    logger.info("  Num train_data = %d", len(train_dataset))
    logger.info("  Num test_data = %d", len(test_dataset))

    def collate(data):
        return pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sample = RandomSampler(train_dataset)
    test_sample = RandomSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sample, batch_size=32, collate_fn=collate, drop_last=True)
    test_dataloader = DataLoader(test_dataset, sampler=test_sample, batch_size=32, collate_fn=collate, drop_last=True)

    return train_dataloader, test_dataloader


def train(model, tokenizer, tr_dataloader, te_dataloader):
    model.resize_token_embeddings(len(tokenizer))

    t_total = len(tr_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Starting fine-tuning.")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for idx, batch in enumerate(tr_dataloader):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)

            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            logging_loss += loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info("EPOCH = [%d/%d] train_steps = %d   loss = %f", epoch, args.num_train_epochs, global_step,
                                logging_loss / args.logging_steps)
                    logging_loss = 0.0

        logger.info("Training loss epoch[%d/%d]: %f", epoch, args.num_train_epochs, tr_loss / global_step)
        result = evaluate(model, tokenizer, te_dataloader)

        tr_loss = 0.0
        global_step = 0


if __name__ == '__main__':
    args.data_path = '../data/sample.json'
    args.block_size = 256
    args.train_batch_size = 8
    args.test_batch_size = 8
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_special_tokens({'additional_special_tokens':["[C0]", "[C1]"]})
    tokenizer.pad_token = 0

    train_dataloader, test_dataloader = load_and_cache_data(args.data_path, args.cache_dir, tokenizer, args.block_size)

