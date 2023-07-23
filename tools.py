# coding=utf-8
import re
import os
import time
import json
import chardet
import pickle
import tiktoken

ENGINE_TURBO = 'gpt-3.5-turbo'
ENGINE_DAVINCI_003 = 'text-davinci-003'
ENGINE_EMBEDDING_ADA_002 = 'text-embedding-ada-002'

LANG_EN = 'English'
LANG_ZH = 'Chinese'
LANG_UN = 'Unknown'


def detect_language(text):
    # Count total characters and English characters
    total_count = len(text)
    en_count = len(re.findall(r'[a-zA-Z]', text))
    # Check if the percentage of English letters exceeds 60%
    if en_count / total_count > 0.6:
        return LANG_EN
    else:
        return LANG_ZH


def detect_encode_type(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        result = chardet.detect(data)
        return result['encoding']


def get_token_count_davinci(text):
    tokenizer = tiktoken.encoding_for_model(ENGINE_DAVINCI_003)
    tokens = tokenizer.encode(text)
    return len(tokens)


def print_doc(json_list):
    with open('tmp.txt', 'w') as f:
        for item in json_list:
            doc = item.get('doc')
            if doc:
                f.write(doc + '\n')
                f.write('-' * 30)
                f.write('\n\n')


# Recursively create folders
def makedirs(filename):
    dir_path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('makedirs %s' % dir_path)


# Save data as a pickle file
def save_pickle_file(data, filename):
    makedirs(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print('saved pkl file ', filename)


# Load a pickle file
def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


# Save a list of content as a text file
def save_file(filename, content):
    """
    :param filename: Output filename
    :param content: List of sentences, each element with its own line
    :return:
    """
    makedirs(filename)
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(content)
    print('save file %s successful!' % filename)


def save_jsonl_file(filename, data, indent=None):
    """
    :param filename: Output filename
    :param data: Data object, List[json]
    :param indent: Indentation
    :return: json line format file
    """
    makedirs(filename)
    with open(filename, 'w', encoding='utf-8') as fp:
        for js in data:
            if indent:
                js_str = json.dumps(js, indent=indent, ensure_ascii=False)
            else:
                js_str = json.dumps(js, ensure_ascii=False)
            fp.write(js_str + '\n')
    print('save file %s successful!' % filename)


def save_json_file(filename, data):
    """
    :param filename: Output filename
    :param data: Data object, json/list
    :return:
    """
    makedirs(filename)
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print('save file %s successful!' % filename)


def load_json_file(filename):
    """
    :param filename: Filename
    :return: Data object, json/list
    """
    with open(filename, encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_file(path):
    lines = get_lines(path)
    data = [json.loads(x) for x in lines]
    return data


# Overwrite the given file with the specified pickled object
def overwrite_pkl_file(filename, data):
    tmp_filename = filename + '.swp'
    save_pickle_file(data, tmp_filename)
    if os.path.exists(filename):
        os.rename(filename, filename + '.old.' + datetime2str())
    os.rename(tmp_filename, filename)
    print('overwrite %s successful!' % filename)


# Overwrite the given file with the specified string list
def overwrite_txt_file(filename, data):
    tmp_filename = filename + '.swp'
    
    save_file(tmp_filename, data)
    if os.path.exists(filename):
        os.rename(filename, filename + '.old.' + datetime2str())
    os.rename(tmp_filename, filename)
    print('overwrite %s successful!' % filename)


# Read each line of a file and return a list
def get_lines(filename):
    with open(filename, encoding='utf-8') as f:
        data = [i.strip() for i in f.readlines() if i.strip() != '']
        return data


# Read the entire content of a file and return it as a string
def get_txt_content(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.read()
        return text


def get_files(root, suffix):
    """
    Get all files with the specified suffix in the specified directory.
    :param root: Specified directory str type, e.g., '.'
    :param suffix: Specified suffix str type, e.g., '.txt'
    :return: List of files
    """
    import os
    import glob
    if not os.path.exists(root):
        raise FileNotFoundError(f'path {root} not found.')
    res = glob.glob(f'{root}/**/*{suffix}', recursive=True)
    res = [os.path.abspath(p) for p in res]
    return res


# Check if a word contains only Chinese characters, i.e., no other symbols outside Chinese characters
def is_chinese_word(word):
    for c in word:
        if not ('\u4e00' <= c <= '\u9fa5'):
            # print(word)
            return False
    return True


# Check if a character is a Chinese character
def is_chinese_char(c):
    if len(c.strip()) == 1 and '\u4e00' <= c <= '\u9fa5':
        return True
    return False


def datetime2str():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d-%H%M%S')


# Calculate the time cost from 'start' to the current time
def time_cost(start):
    cost = int(time.time() - start)
    h = cost // 3600
    m = (cost % 3600) // 60
    print('')
    print('Total time cost from %s: %s hours %s mins' % (datetime2str(), h, m))


# Append 'content_list' to 'filename'
def append_file(filename, content_list, new_line=False):
    if not content_list:
        return
    if new_line:
        content_list = [text if text.endswith('\n') else text+'\n' for text in content_list]
    with open(filename, 'a+', encoding='utf-8') as f:
        f.writelines(content_list)
    # timestamp = datetime2str()
    # print('[%s]: append_file %s successful!' % (timestamp, filename))


def keep_only_alnum_chinese(s):
    # Regular expression to keep only Chinese characters, English characters, and numbers
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')

    # Remove non-Chinese characters, English characters, and numbers, get the new string
    s2 = re.sub(pattern, '', s)
    return s2


def choose_language_template(lang2template: dict, text: str):
    lang = detect_language(text)
    if lang not in lang2template:
        lang = LANG_EN
    return lang2template[lang]


def replace_newline(text):
    # print(f"\n\nraw:\n{text}\n--------------\n")
    result = ""
    code_start = False
    for i in range(len(text)):
        if i >= 3 and text[i-3: i] == "```":
            code_start = not code_start
            if code_start == False:
                result += f"\n{text[i]}"
            else:
                result += text[i]
        elif not code_start and text[i] == "\n":
            result += "<br>\n"
        else:
            result += text[i]
    # print(f"\n\nresult:\n{result}\n--------------\n")
    return result
