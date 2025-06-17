import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from faker import Faker
from huggingface_hub import hf_hub_download
import json

# Create a Faker instance with Japanese locale
fake = Faker("ja_JP")


# Generate random Japanese text
def generate_random_japanese_text():
    return fake.text()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        # elif isinstance(maybe_tensor, Mapping):
        #     return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def create_batch_dict(tokenizer, input_texts, max_length: int = 512):
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        return_tensors="pt",
    )


def pool(last_hidden_states, attention_mask, pool_type: str):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


class KVEmbedding:
    def __init__(self, device):
        self.device = device

        # Load tokenizer and model from pretrained multilingual-e5-small
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to(self.device)

        self.model.eval()  # Set model to evaluation mode

    def average_pool(self, last_hidden_states, attention_mask):
        # Apply mask to hidden states, set masked positions to 0
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # Average the hidden states along the sequence dimension
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embedding(self, l_transcription, batch_size=32):
        # Tokenize input transcriptions
        batch_dict = self.tokenizer(
            l_transcription,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        return batch_dict

    def _do_encode(self, input_texts):
        encoded_embeds = []
        batch_size = 64
        for start_idx in range(0, len(input_texts), batch_size):
            batch_input_texts = input_texts[start_idx : start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            # batch_dict = move_to_cuda(batch_dict)
        return encoded_embeds


import random
from faker import Faker

# # Lists of Japanese characters
hiragana = [
    "あ",
    "い",
    "う",
    "え",
    "お",
    "か",
    "き",
    "く",
    "け",
    "こ",
    "さ",
    "し",
    "す",
    "せ",
    "そ",
    "た",
    "ち",
    "つ",
    "て",
    "と",
    "な",
    "に",
    "ぬ",
    "ね",
    "の",
    "は",
    "ひ",
    "ふ",
    "へ",
    "ほ",
    "ま",
    "み",
    "む",
    "め",
    "も",
    "や",
    "ゆ",
    "よ",
    "ら",
    "り",
    "る",
    "れ",
    "ろ",
    "わ",
    "を",
    "ん",
]
katakana = [
    "ア",
    "イ",
    "ウ",
    "エ",
    "オ",
    "カ",
    "キ",
    "ク",
    "ケ",
    "コ",
    "サ",
    "シ",
    "ス",
    "セ",
    "ソ",
    "タ",
    "チ",
    "ツ",
    "テ",
    "ト",
    "ナ",
    "ニ",
    "ヌ",
    "ネ",
    "ノ",
    "ハ",
    "ヒ",
    "フ",
    "ヘ",
    "ホ",
    "マ",
    "ミ",
    "ム",
    "メ",
    "モ",
    "ヤ",
    "ユ",
    "ヨ",
    "ラ",
    "リ",
    "ル",
    "レ",
    "ロ",
    "ワ",
    "ヲ",
    "ン",
]
kanji = [
    "日",
    "本",
    "語",
    "学",
    "校",
    "生",
    "時",
    "間",
    "人",
    "大",
    "小",
    "中",
    "山",
    "川",
    "口",
    "目",
    "耳",
    "手",
    "足",
    "力",
    "男",
    "女",
    "子",
    "父",
    "母",
]

# Combine all character sets
all_characters = hiragana + katakana + kanji


# Generate random Japanese text
def generate_random_japanese(length):
    return "".join(random.choices(all_characters, k=length))


def remove_invalid_characters(valid_chars, text):
    """
    Removes all invalid characters from the given text, keeping only the characters present in char_dicts.

    Args:
    char_dicts (dict): Dictionary of valid characters.
    text (str): Input text string.

    Returns:
    str: Text string with only valid characters.
    """
    # Convert dict keys to a set for faster lookup
    filtered_text = "".join(c for c in text if c in valid_chars)
    return filtered_text


if __name__ == "__main__":
    from tqdm import tqdm
    import psutil

    print("Start app ...")
    filename = hf_hub_download("intfloat/multilingual-e5-small", "tokenizer.json")
    with open(filename, "r") as file:
        character_info = json.load(file)
    character_dict = {}
    print("Vocab is loading ...")
    with tqdm(total=100, desc="cpu%", position=1) as cpubar, tqdm(total=100, desc="ram%", position=0) as rambar:
        for data in character_info["model"]["vocab"]:
            character_dict[data[0]] = data[1]
        valid_chars = set(character_dict.keys())
        print("Start loading model")
        kv_embedding = KVEmbedding("cpu")
        print("Loading model: Done!!!")
        for i in range(7500):
            print(f"============{i}==============")
            length = random.randint(600, 1000)
            # print(length)
            input_texts = []
            for s in range(length):
                text_length = random.randint(1, 10000)

                random_text = generate_random_japanese(text_length)

                # before = len(random_text)
                random_text = remove_invalid_characters(valid_chars, random_text)
                # after = len(random_text)
                # if after != before:
                #     print(before, after)
                random_text = random_text[:450]
                input_texts.append(random_text)
                rambar.n = psutil.virtual_memory().percent
                cpubar.n = psutil.cpu_percent()
                rambar.refresh()
                cpubar.refresh()

            filter_output = input_texts[:512]

            del input_texts

            # print(len(filter_output))

            output = kv_embedding.embedding(filter_output)
