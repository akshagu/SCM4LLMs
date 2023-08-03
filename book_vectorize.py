import os
import sys
import argparse
from os.path import join
from tools import *
import logging
from core.api import set_api_logger, KEY_MANAGER
from core.book import SummaryBot, SummaryTurn, set_chat_logger
from utils.spliter import BookSpliter
from prompts.book import *

args: argparse.Namespace = None
bot: SummaryBot = None


def get_user_input(user_str, pre_sre, hist_str=None):
    lang2template = {
        LANG_EN: en_agent_scm_prompt,
        LANG_ZH: zh_agent_scm_prompt
    }

    template: str = choose_language_template(lang2template, user_str)

    current_text = user_str
    previous_content = pre_sre
    if hist_str:
        previous_content = f"{hist_str}\n\n{pre_sre}"
    
    input_text = template.format(previous_content=previous_content, current_text=current_text)

    return input_text

def summarize_embed_one_turn(bot: SummaryBot, dialogue_text, dialogue_text_with_index):
    lang2template = {
        LANG_EN: en_turn_summarization_prompt,
        LANG_ZH: zh_turn_summarization_prompt
    }

    tmp = choose_language_template(lang2template, dialogue_text)
    input_text = tmp.format(input=dialogue_text)
    logger.info(f'turn summarization input_text: \n\n{input_text}')
    # 如果原文很短，保留原文即可
    summarization = dialogue_text_with_index
    if get_token_count_davinci(input_text) > 300:
        logger.info(f'current turn text token count > 300, summarize !\n\n')
        summarization = bot.ask(input_text)
        logger.info(f'Summarization is:\n\n{summarization}\n\n')
    else:
        logger.info(f'Raw content is short, keep raw content as summarization:\n\n{summarization}\n\n')
    embedding = bot.vectorize(dialogue_text_with_index)
    return summarization, embedding


def check_key_file(key_file):
    if not os.path.exists(key_file):
        print(f'[{key_file}] not found! Please put your apikey in the txt file.')
        sys.exit(-1)


def get_first_prompt(user_text):
    lang2template = {
        LANG_EN: en_start_prompt,
        LANG_ZH: zh_start_prompt
    }
    tmp = choose_language_template(lang2template, user_text)
    user_input = tmp.format(text=user_text)
    return user_input


def get_paragragh_prompt(user_text):
    lang2template = {
        LANG_EN: en_agent_no_scm_prompt,
        LANG_ZH: zh_agent_no_scm_prompt
    }

    tmp = choose_language_template(lang2template, user_text)
    user_input = tmp.format(text=user_text)
    return user_input


def summarize_book(book_file, model_name, scm=True):
    global args
    global bot

    bot.clear_history()
    hist_lst = []
    spliter = BookSpliter(model_name)

    paragraphs = spliter.split(book_file)

    total = len(paragraphs)
    i = 0
    for text in paragraphs:
        if (len(text)>0):
          user_input = ''
          if scm:
              if i == 0:
                  user_input = get_first_prompt(text)
              else:
                  pre_info = bot.get_turn_for_previous()
                  user_input = get_user_input(text, pre_info)
          else:
              user_input = get_paragragh_prompt(text)
          
          logger.info(f'\n--------------\n[第{i+1}/{total}轮] book_file: {book_file}  model_name:{model_name};  USE SCM: {scm} \n\nuser_input:\n\n{user_input}\n--------------\n')
          print(f'\n--------------\n[第{i+1}/{total}轮] book_file: {book_file} model_name:{model_name};  USE SCM: {scm}\n--------------\n')

          system_response: str = bot.ask(user_input).strip()
          logger.info(f"model_name:{model_name}; USE SCM: {scm}; Summary:\n\n{system_response}\n\n")
          cur_text_without_index = 'User: {}\n\nAssistant: {}'.format(user_input, system_response)
          cur_text_with_index = '[Turn {}]\n\nUser: {}\n\nAssistant: {}'.format(i, user_input, system_response)
          summ, embedding = summarize_embed_one_turn(bot, cur_text_without_index, cur_text_with_index)
          hist_lst.append({'user_input': user_input, 'summ': summ, 'user_sys_text': cur_text_with_index, 'system_response': system_response, 'embedding': embedding})
          # just book summarization do not need embedding
          # embedding = None
          cur_turn = SummaryTurn(paragraph=text, summary=system_response, embedding=bot.vectorize(system_response))
          bot.add_turn_history(cur_turn)
          logger.info(f"model_name:{model_name}; USE SCM: {scm};  Processing: {i+1}/{total}; add_turn_history is done!")
          i=i+1
    
    suffix = ''
    if scm is False:
        suffix = 'no_scm'
    basename = os.path.basename(book_file)
    json_filename = f'history/book-sum/book-summary-{basename}-{model_name}{suffix}.json'
    makedirs(json_filename)
    save_json_file(json_filename, hist_lst)
    bot.clear_history()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_choices = [ENGINE_GPT4, ENGINE_TURBO, ENGINE_DAVINCI_003]
    parser.add_argument("--apikey_file", type=str, default="./config/apikey.txt")
    parser.add_argument("--model_name", type=str, default=ENGINE_DAVINCI_003, choices=model_choices)
    parser.add_argument("--book_files", nargs='+', type=str, required=True)
    parser.add_argument("--logfile", type=str, default="./logs/book.summary.log.txt")
    parser.add_argument('--no_scm', action='store_true', help='do not use historical memory, default is False')
    args = parser.parse_args()

    check_key_file(args.apikey_file)

    log_path = args.logfile
    makedirs(log_path)
    # 配置日志记录

    logger = logging.getLogger('summary_logger')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('【%(asctime)s - %(levelname)s】 - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    set_chat_logger(logger)
    set_api_logger(logger)

    logger.info('\n\n\n')
    logger.info('#################################')
    logger.info('#################################')
    logger.info('#################################')
    logger.info('\n\n\n')
    logger.info(f"args: \n\n{args}\n")

    book_list = args.book_files
    
    # whether use scm for history memory
    USE_SCM = False if args.no_scm else True
    model_name = args.model_name

    
    bot = SummaryBot(model_name=model_name)
    for book_file in book_list:
        book_name = os.path.basename(book_file)
        logger.info(f'\n\n※※※ Begin Summarize Book : {book_name} ※※※\n\n')
        summarize_book(book_file, model_name, scm=USE_SCM)
    KEY_MANAGER.remove_deprecated_keys()