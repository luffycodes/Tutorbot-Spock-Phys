"""Inference for FastChat models."""
import abc
import gc
import math
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import re
import json
import psutil
import torch
import ast
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# create a dict file.. map "decitiond to string in that dict"
def run_python(python_code, result_variables):
    """
    Execute the inputted string as Python code and return the output.

    Args:
        python_code (str): Python code as a string
        result_variables (str): the name(s) of the variable(s) that stores the final result(s)

    Returns:
        str: "Tutorbot calculated "result_variable" to be "result". || "Unable to execute Python code."
    """
    
    try:
        print("python to execute:" + python_code + "\n")
        python_code = python_code.replace('\\n', '\n').replace("python", "").replace("`", "")
        results_dict = {}
        exec(python_code, globals(), results_dict)
        
        res_vars_split = result_variables.split(",")
        res_vars_split_strip = [v.strip() for v in res_vars_split]
        results = []
        
        for v in res_vars_split_strip:
            result = results_dict[v]
            results.append(result)
                        
        results_dict = dict(zip(res_vars_split_strip, results))
        final_output = ""
        for var, value in results_dict.items():
            final_output += f"Tutorbot calculated \"{var}\" to be {value}.\n"

        return final_output
    
    except Exception as err:
        print(err)
        return "Tutorbot calculated student calculation to be correct.\n"


@torch.inference_mode()
def generate_local(prompt, model, tokenizer, device, temperature, repetition_penalty, top_p, top_k, max_new_tokens):
    print("**ENTERING GEN**")
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p = top_p,
        top_k = top_k,
        max_new_tokens=max_new_tokens,
        #do_sample=True,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print("****")
    print(outputs)
    print("**EXITING GEN**")
    del output_ids
    gc.collect()
    torch.cuda.empty_cache()
    return outputs

def extract_and_replace_substring_between_backticks(input_string):
    # Define the regular expression pattern to match the substring between backticks
    pattern = r"```([\s\S]*?)```"

    # Use re.sub to replace all occurrences of the pattern with the word "ignore"
    new_string = re.sub(pattern, "ignore", input_string)

    # Use re.findall to find all occurrences of the pattern in the input string
    matches = re.findall(pattern, input_string)

    # Return the first match (if found) and the new string
    extracted_substring = f"```{matches[0]}```" if matches else None
    return extracted_substring, new_string

def get_string_dict_code(output):
    #output_dict = ast.literal_eval(str(output))
    #output_dict2 = ast.literal_eval(str(output_dict['Python']))
    if not output.endswith("}}"):
        output = output + '}'
    code, output_dict = extract_and_replace_substring_between_backticks(str(output))
    output_dict = ast.literal_eval(str(output_dict))
    output_dict2 = ast.literal_eval(str(output_dict['Python']))
    output_dict2['Python Code'] = code
    return output_dict2

def get_string_dict(output):
    output_dict = ast.literal_eval(str(output))
    return output_dict

def parse_python_json(output):
    output = str(output)
    output = output.replace("'", '"')
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        print("error decoding json")
    return output['Use Python']

def call_python(prompt, model, tokenizer, device, temperature, repetition_penalty,  top_p, top_k, max_new_tokens):
    #deciding state
    try:
        print('deciding_state reached')
        deciding_state = 'Based on the last input from student, decide if you need to use python or not: ' \
                     'a) Yes, use python. If student provided a mathematical answer, verify it by using Python ' \
                     'b) Yes, use python. If tutorbot think his response relies on math calculations, ' \
                     'get the output from Python ' \
                     'c) No, do not use python. ' \
                     'The function of the "Description" field is to provide a natural language description ' \
                     'of the desired calculation. Include the numerical values of the inputs and be as detailed ' \
                     'as possible.If you choose to use Python ("a" or "b"), output the following JSON object, ' \
                     'replacing any instances of ".." and following each field with a comma except for the last one:'\
                     '{{'\
                     '"Use Python": "a/b",'\
                     '"Description": ".."'\
                     '}}'\
                     'Again, utilize Python code as often as possible. ' \
                     'If the student provides a mathematical calculation in their most recent response, ' \
                     'always verify it.If you choose not to use python ("c"), output the following JSON structure, ' \
                     'replacing any instances of ".." and following each field with a comma except for the last one:'\
                     '{{'\
                     '"Use Python": "c"'\
                     '}}'
        prompt = prompt + " INSTRUCTION: " + deciding_state + " ASSISTANT:"
        output = generate_local(prompt, model, tokenizer, device, temperature, repetition_penalty, top_p, top_k, max_new_tokens)
        prompt = prompt + str(output)

        output_dict = get_string_dict(output)
        abc = output_dict['Use Python']

        if abc in ['a', 'b']:
            #get python code here
            print('use_python_state reached')
            use_python = 'Generate the Python code with surrounding backticks and "python" keyword. ' \
                     'Include comments. State the variable that stores the result on the last line. ' \
                     'Output everything in the following JSON object, following each field with a ' \
                     'comma except for the last one:' \
                     '{{' \
                     '"Python":' \
                     '{{' \
                     '"Python Code": "``` python ..```",' \
                     '"Result Variable": "Variable that the final answer is stored in"' \
                     '}}' \
                     '}}'
            prompt = prompt + ' INSTRUCTION: Based on this description: ' + output_dict['Description'] + use_python + " ASSISTANT:"
            output = generate_local(prompt, model, tokenizer, device, temperature, repetition_penalty, top_p, top_k, max_new_tokens)
            prompt = prompt + str(output)
            output_dict = get_string_dict_code(output)
            code_output = run_python(output_dict['Python Code'], output_dict['Result Variable'])
            #receive_python
            print('receive_python_state reached')
            prompt = prompt + " PYTHON: " + str(code_output) + " INSTRUCUTION: Use Tutorbot's Python output " \
                                                          "to evaluate the student's answer and provide " \
                                                          "feedback to the student.Remember to output the " \
                                                          "response in the json format specified earlier."
        elif abc == 'c':
            print('no_python_state reached')
            prompt = prompt +" INSTRUCUTION: Generate a response that provide futher guidance to the student." \
                          "Remember to output the response in the json format specified earlier."
        else:
            print('generate failure')
            raise Exception('generation failure')
    except Exception:
        print('The code ran into problem')
    return prompt + " ASSISTANT:"


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: GptqConfig,
    revision: str,
    judge_sent_end: bool,
    debug: bool,
    history: bool = True,
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_config,
        revision,
        debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        return conv

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break

        if inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue

        if inp.startswith("!!save"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!save <filename>")
                continue
            else:
                filename = args[1]

            # Add .json if extension not present
            if not "." in filename:
                filename += ".json"

            print("saving...", filename)
            with open(filename, "w") as outfile:
                json.dump(conv.dict(), outfile)
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # CODE TO CHECK IF PYTHON NEEDED
        if_python_prompt_str = prompt + if_python_prompt()

        gen_params = {
            "model": model_path,
            "prompt": if_python_prompt_str,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output("### MONOLOGUE STARTS ###\n**INTERNAL** DECISION IF TO USE PYTHON")
        output_stream = generate_stream_func(
            model,
            tokenizer,
            gen_params,
            device,
            context_len=context_len,
            judge_sent_end=judge_sent_end,
        )
        first_output = chatio.stream_output(output_stream)

        output_dict = get_string_dict(first_output.strip())
        abc = output_dict.get('Use Python', 'd')

        if abc in ['y']:
            gen_code_prompt = pre_run_python_prompt(output_dict['Description'])
            prompt10 = prompt + if_python_prompt_str + first_output.strip() + gen_code_prompt
            gen_params['prompt'] = prompt10
            chatio.prompt_for_output("**INTERNAL** PYTHON CODE")
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            second_output = chatio.stream_output(output_stream)
            output_dict2 = get_string_dict_code(second_output.strip())
            code_output = run_python(output_dict2['Python Code'], output_dict2['Result Variable'])
            print("**INTERNAL** CODE OUTPUT: " + code_output.strip())
            post_code_prompt = post_run_python_prompt(second_output.strip(), code_output)
            prompt11 = prompt10 + post_code_prompt
            gen_params['prompt'] = prompt11
            chatio.prompt_for_output("### MONOLOGUE ENDS ###\nASSISTANT")
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            third_output = chatio.stream_output(output_stream)
            conv.update_last_message(prompt11 + "\n" + third_output.strip())
        elif abc in ['n']:
            no_code_prompt = " INSTRUCTION: Generate a response to the student. ASSISTANT:"
            prompt12 = prompt + if_python_prompt_str + first_output.strip() + no_code_prompt
            gen_params['prompt'] = prompt12
            chatio.prompt_for_output("### MONOLOGUE ENDS ###\nASSISTANT")
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            third_output = chatio.stream_output(output_stream)
            conv.update_last_message(prompt12 + "\n" + third_output.strip())
        else:
            chatio.prompt_for_output("### MONOLOGUE ENDS ###\nASSISTANT")
            third_output = first_output
            conv.update_last_message(third_output.strip())



def if_python_prompt():
    prompt = '''Decide if you want to use python to check student's calculations or frame your reply. Provide a description if you decide to use code. Please avoid making any conclusions about correctness of the solution in the description. Wait for python code output to make conclusions about correctness. Also, remember to output decision in json {"Use Python": "..", "Description": ".."}'''
    prompt = '''Decide if you want to use python to check student's calculations before arriving at any conclusions about its correctness or frame your reply. Provide a description if you decide to use code. Also, remember to output decision in json {"Use Python": "..", "Description": ".."}'''
    prompt = '''Decide if you want to use python to check student's calculations. If you decide to use code, generate the description of the desired python program. Please do not reach any conclusions about correctness of calculations in description, since description is meant to check correctness of most recent student input. Also, remember to output decision in the json {"Use Python": "..", "Description": ".."}'''
    prompt = " INSTRUCTION: " + prompt + " ASSISTANT: "
    return prompt


def post_run_python_prompt(output_dict, code_output):
    prompt = output_dict
    prompt = prompt + " PYTHON: " + str(code_output) + ''' INSTRUCTION: Now based on the code output, please generate the response to the student. '''
    return prompt + " ASSISTANT: "


def pre_run_python_prompt(description):
    use_python = ' Generate the Python code with surrounding backticks and "python" keyword and result variable to store the final answer. '
    use_python = ' Generate the Python code with surrounding backticks and "python" keyword and result variable to store the final answer in json format below. ' \
                     '{"Python": {"Python Code": "..", "Result Variable": ".."}}'
    prompt = ' INSTRUCTION: Based on this description: ' + description + use_python
    return prompt + " ASSISTANT: "

