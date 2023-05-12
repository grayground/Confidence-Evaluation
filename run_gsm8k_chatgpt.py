import openai
import os
import argparse
import numpy as np
import random
import time
import tqdm
import backoff
import json
import re
from datasets import load_dataset


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--one_turn_reasoning", action='store_true')
    parser.add_argument("--framework_type", type=str, default="progressive", choices=["progressive", "direct"])

    parser.add_argument("--seed", type=int, default = 42, help="Random seed")

    # model related
    parser.add_argument("--model_name", type=str, default = "gpt-3.5-turbo", help = "The name of LLM")
    parser.add_argument("--api_key", type=str, default="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", help="The api-key of LLM, default ChatGPT")
    parser.add_argument("--temperature", type=float, default=0, choices=[0, 0.5, 0.7], help="Sampling temperature, between 0 and 2. Higher values will make the output more random, while lower values will make it more focused and deterministic")
    parser.add_argument("--top_p", type=float, help="Nucleus sampling, where the model considers the results of the tokens with top_p probability mass")
    parser.add_argument("--max_tokens", type=int, help="The total length of input tokens and generated tokens is limited by the model's context length")
    
    args = parser.parse_args()

    return args


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_data_gsm8k():
    gsm8k = load_dataset("gsm8k", "main")
    gsm8k_test = gsm8k["test"]

    questions, answers = [], []
    for data in gsm8k_test:
        questions.append(data["question"])
        answers.append(data["answer"].strip().split("#### ")[1])
        if data["answer"].strip().split("#### ")[1].find(".") != -1:
            print(data["question"])
    return questions, answers


def get_additional_query():
    pass


def get_response(args, query):
    if args.one_turn_reasoning:
        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        response = openai.ChatCompletion.create(
            model=args.model_name,
            messages=user_messages,
            temperature=args.temperature
        )
        
        return response["choices"][0]["message"]["content"]
    else:
        pass


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_response_with_backoff(args, query):
    return get_response(args, query)


def format_response(response):
    if bool(re.search(r"\d", response)):
        if response.find("Answer:") == -1:
            return "NULL"
        else:
            system_answer = response.split("Answer:")[1].strip()

            # decomposition equation, like 1 + 2 = 3 -> 3
            if system_answer.find("=") != -1:
                system_answer = system_answer.split("=")[1].strip()

            # remove period, like 3. -> 3
            if system_answer[-1:] == ".":
                system_answer = system_answer[:-1]

            # convert float to integer, like 3.00 -> 3
            if system_answer[-3:] == ".00":
                system_answer = system_answer[:-3]

            # remove comma, like 30,300 -> 30300
            while True:
                flag = re.search("\d,\d", system_answer)
                if flag:
                    flag = flag.group()
                    system_answer = system_answer.replace(flag, flag.replace(",", ""))
                else:
                    break

            # keep only numbers, like $33 -> 33
            number_ls = re.findall(r"\d+\.?\d*", system_answer)
            
            if len(number_ls) == 0:
                return "NULL"
            else:
                return number_ls[0]
    else:
        return "NULL"


def compute_accuracy(answers, system_answers):
    accuracy = 0
    for idx in range(len(system_answers)):
        if answers[idx] == system_answers[idx]:
            accuracy += 1
    acc = accuracy / len(system_answers) * 100
    print("*" * 30)
    print("Accuracy: ", acc)
    return acc


def evaluate(args, prompt_format, log_file_path, result_file_path):
    log_file = open(log_file_path, 'w+', encoding='utf-8')
    res_file = open(result_file_path, 'w+', encoding='utf-8')

    questions, answers = load_data_gsm8k()
    system_answers = []
    for idx in range(len(questions)):
        if idx == 50:
            break
        # evaluation only
        if args.one_turn_reasoning:
            query = f"{questions[idx]} {prompt_format}"

            while True:
                try:
                    response = get_response_with_backoff(args, query) 
                except Exception:
                    time.sleep(20)
                    continue
                else:
                    break

            print("*" * 30 + str(idx) + "*" * 30)
            print(query)
            print(response)
            system_answer = format_response(response)
            system_answers.append(system_answer)
            print(system_answer)
            print(f"GT: {answers[idx]}")
            log_file.write(f"{'*' * 30 + str(idx) + '*' * 30}\n{query}\n{response}\n{system_answer}\nGT: {answers[idx]}\n")
            res_file.write(system_answer + "\n")

            time.sleep(10)
        else:
            if args.framework_type == "direct":
                pass
            elif args.framework_type == "progressive":
                pass
    
    acc = compute_accuracy(answers, system_answers) 

    res_file.write("*" * 30 + " Accuracy " + "*" * 30 + "\n")
    res_file.write(f"{acc}")

    log_file.close()
    res_file.close()
    print("Done!")


if __name__ == '__main__':
    args = init_args()
    setup_seed(args.seed)
    openai.api_key = args.api_key

    prompt_format = "If the answer is numbers, you just have to give the number. Give the number separately on the last line of your response, such as: ’Answer: ...‘"
    
    log_file_path = f"output/{args.dataset}_{args.temperature}_log.txt"
    result_file_path = f"output/{args.dataset}_{args.temperature}_results.txt"

    evaluate(args, prompt_format, log_file_path, result_file_path)
