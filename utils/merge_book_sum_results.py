import os
import re
import sys
import tiktoken

sys.path.append('.')
sys.path.append('..')

from tools import *

turbo_tokenizer = tiktoken.encoding_for_model(ENGINE_TURBO)
gpt4_tokenizer = tiktoken.encoding_for_model(ENGINE_GPT4)
davinci_tokenizer = tiktoken.encoding_for_model(ENGINE_DAVINCI_003)


def extract_book_name(file_name):
    front_pos = file_name.find("summary-") + len("summary-")
    end_pos = file_name.find(".txt")
    book_name = file_name[front_pos:end_pos]
    return book_name


def extract_model_name(file_name):
    front_pos = file_name.find(".txt-") + len(".txt-")
    end_pos = file_name.find(".json")
    name = file_name[front_pos:end_pos]
    return name


# Extract a list of JSON file names
root = 'history/SCM_BOOK_SUMMARY'
files = get_files(root, '.json')

# Extract a list of book names (without duplicates)
book_names = []
for f in files:
    bn = extract_book_name(f)
    if bn not in book_names:
        book_names.append(bn)

# Construct JSON
book_model_to_summary = {}
for f in files:
    bn = extract_book_name(f)
    mn = extract_model_name(f)
    key = f"{bn}_{mn}"

    summary_list = load_json_file(f)
    final_sum = summary_list[-1]['final summary']

    tokens = 100
    tokenizer = None
    if mn == ENGINE_TURBO:
        tokenizer = turbo_tokenizer
    elif mn == ENGINE_DAVINCI_003:
        tokenizer = davinci_tokenizer
    elif mn == ENGINE_GPT4:
        tokenizer = gpt4_tokenizer

    total_tokens = 0
    for js in summary_list:
        if 'paragraph' in js:
            p = js['paragraph']
            tks = tokenizer.encode(p)
            total_tokens += len(tks)

    book_model_to_summary[key] = [final_sum, total_tokens]


data = []
for i, book in enumerate(book_names):
    obj = {}
    obj['id'] = f"book_{i + 1}"
    obj['book_name'] = book
    lang = detect_language(book.replace('_', ' '))
    obj['language'] = lang
    obj[f"{ENGINE_DAVINCI_003}_summary"] = book_model_to_summary[f"{book}_{ENGINE_DAVINCI_003}"][0]
    obj[f"{ENGINE_GPT4}_summary"] = book_model_to_summary[f"{book}_{ENGINE_GPT4}"][0]
    obj[f"{ENGINE_TURBO}_summary"] = book_model_to_summary[f"{book}_{ENGINE_TURBO}"][0]
    obj[f"{ENGINE_DAVINCI_003}_content_tokens"] = book_model_to_summary[f"{book}_{ENGINE_DAVINCI_003}"][1]
    obj[f"{ENGINE_TURBO}_content_tokens"] = book_model_to_summary[f"{book}_{ENGINE_TURBO}"][1]
    obj[f"{ENGINE_GPT4}_content_tokens"] = book_model_to_summary[f"{book}_{ENGINE_GPT4}"][1]
    obj[f"{ENGINE_DAVINCI_003}_win"] = False
    obj[f"{ENGINE_TURBO}_win"] = False
    obj[f"{ENGINE_GPT4}_win"] = False
    obj[f"comment"] = ""

    data.append(obj)

# Save JSON list
dst_file = 'annotation_data/book_summary.json'
save_json_file(dst_file, data)
