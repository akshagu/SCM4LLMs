import json
import tiktoken
from tools import *
from dataclasses import dataclass
from core.meeting import Utterance
from typing import List

def get_tokenizer_func(model_name):
    if model_name not in [ENGINE_TURBO, ENGINE_DAVINCI_003]:
        raise ValueError(f'Invalid model name: {model_name} when calling get_tokenizer_func.')
    tokenizer = tiktoken.encoding_for_model(model_name)
    return tokenizer.encode

def get_token_count(func, text):
    tokens = func(text)
    return len(tokens)

@dataclass
class BookSpliter:
    model: str

    def split(self, txt_file):
        tokenize_func = get_tokenizer_func(self.model)
        encoding = detect_encode_type(txt_file)
        print(f"encoding: {encoding}")

        if 'gb' in encoding.lower():
            encoding = 'gbk'

        with open(txt_file, 'r', encoding=encoding) as f:
            book = f.read() # Read the content of the book

        lang = detect_language(book)
        if lang == LANG_ZH:
            separator = '。'
        else:
            separator = '. '

        max_tokens = 3000 # Maximum 3000 tokens per part
        split_book = []
        current_tokens = 0
        current_block = ""

        for sentence in book.split(separator): # Split the book text based on sentences
            token_count = get_token_count(tokenize_func, sentence) # Calculate the number of tokens in the sentence

            if current_tokens + token_count > max_tokens:
                split_book.append(current_block)
                current_block = sentence + separator
                current_tokens = token_count
            else:
                current_tokens += token_count
                current_block += sentence + separator

        if current_block: # Add the last block
            split_book.append(current_block)

        dst_dir = './logs/book_split/'
        dst_json_file = os.path.join(dst_dir, os.path.basename(txt_file) + f".{self.model}.json")
        makedirs(dst_json_file)
        save_json_file(dst_json_file, split_book)

        line_separator = '#' * 30
        line_separator = f"\n\n{line_separator}\n\n"
        final_str = line_separator.join(split_book)

        dst_txt_file = os.path.join(dst_dir, os.path.basename(txt_file) + f".{self.model}.txt")
        makedirs(dst_txt_file)
        save_file(dst_txt_file, [final_str])

        return split_book

@dataclass
class MeetingSpliter:
    model: str

    def split(self, dialogues: List[Utterance], meeting_id):
        tokenize_func = get_tokenizer_func(self.model)
        max_tokens = 2800 # Maximum 2800 tokens per part
        meetings = [u.to_text() for u in dialogues]
        meetings_str = '\n\n'.join(meetings)

        separator = '。'

        split_parts = []
        current_tokens = 0
        current_block = ""

        for sentence in meetings_str.split(separator): # Split the meeting text based on sentences
            token_count = get_token_count(tokenize_func, sentence) # Calculate the number of tokens in the sentence

            if current_tokens + token_count > max_tokens:
                split_parts.append(current_block)
                current_block = sentence + separator
                current_tokens = token_count
            else:
                current_tokens += token_count
                current_block += sentence + separator

        if current_block: # Add the last block
            split_parts.append(current_block)

        dst_dir = './logs/meeting_split/'
        dst_json_file = os.path.join(dst_dir, meeting_id + f".{self.model}.json")
        makedirs(dst_json_file)
        save_json_file(dst_json_file, split_parts)

        line_separator = '#' * 30
        line_separator = f"\n\n{line_separator}\n\n"
        final_str = line_separator.join(split_parts)

        dst_txt_file = os.path.join(dst_dir, meeting_id + f".{self.model}.txt")
        makedirs(dst_txt_file)
        save_file(dst_txt_file, [final_str])

        return split_parts
