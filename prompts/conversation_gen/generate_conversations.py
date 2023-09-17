import random
import string
import time
import openai
import json
import pandas as pd
import os
from dotenv import load_dotenv
from math import *

load_dotenv()

# Get the API key for the OpenAI API.
openai.api_key = os.getenv('API_KEY')

def run_python(python_code, result_variables):
    """
    Execute the inputted string as Python code and return the output.

    Args:
        python_code (str): Python code as a string
        result_variables (str): the name(s) of the variable(s) that stores the final result(s)

    Returns:
        str: "Tutorbot calculated "result_variable" to be "result". || "Unable to execute Python code."
    """
    python_code = python_code.replace('\\n', '\n').replace("python", "").replace("`", "")
    
    try:
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
        return "Unable to execute Python code."

def get_tutorbot_sys_prompt(question, solution, history, state="deciding", response_id=None, description=None, python_output=None):
    """
    Generates system prompt for tutorbot gpt based on the desired state.

    Args:
        question (str): the initial question that the student asked
        solution (str): the provided step-by-step textbook solution
        history (str): current conversation history
        state (str, optional): the state of the tutorbot i.e. "deciding" | "initial" | "use_python" | "no_python" | "received_python". Defaults to "deciding".
        response_id (str, optional): current step number. Defaults to None.
        description (str, optional): the description of the code to be generated; state must be "use_python". Defaults to None.
        python_output (str, optional): the python output calculated from the most recently generated python code; state must be "received_python". Defaults to None.

    Returns:
        str: the tutorbot system prompt
    """
    if state == "initial":
        # Initial state
        with open("data_gen/conversation_generation/v4/states/initial_state.txt") as file:
            initial_state = file.read()
            return initial_state.format(question=question, solution=solution)
    elif state == "no_python":
        # Chose not to use python
        with open("data_gen/conversation_generation/v4/states/no_python_state.txt") as file:
            no_python_state = file.read()
            return no_python_state.format(question=question, solution=solution, history=history, response_id=response_id)
    elif state == "use_python":
        # Chose not to use python
        with open("data_gen/conversation_generation/v4/states/use_python_state.txt") as file:
            use_python_state = file.read()
            return use_python_state.format(description=description)
    elif state == "received_python":
        # Received a python output and need to evaluate student's response state
        with open("data_gen/conversation_generation/v4/states/received_python_state.txt") as file:
            received_python_state = file.read()
            return received_python_state.format(question=question, solution=solution, history=history, response_id=response_id, description=description, python_output=python_output)
    else:
        # (deciding) Deciding between using python or not state
        with open("data_gen/conversation_generation/v4/states/deciding_state.txt") as file:
            deciding_state = file.read()
            return deciding_state.format(question=question, solution=solution, history=history, response_id=response_id)

def get_student_sys_prompt(question, history):
    """
    Generates the system prompt for student gpt.

    Args:
        question (str): the initial question that the student asks
        history (str): the current conversation

    Returns:
        str: the student's system prompt
    """
    with open("data_gen/conversation_generation/v4/states/student_state.txt") as file:
        student_state = file.read()
        return student_state.format(question=question, history=history)

def run_chat_completion(history):
    """
    Run openai's chat completion on the current conversation history.

    Args:
        history (str): current conversation history

    Returns:
        str: chat completion response
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=history,
        )
        response = completion['choices'][0]['message']['content']
    except Exception as err:
        print(err)
        time.sleep(60)
        response = run_chat_completion(history)
    
    return response

def run_tutorbot_gpt(question, solution, history, state="deciding", response_id=None, description=None, python_output=None):
    """
    Run tutorbot gpt.

    Args:
        question (str): initial question that the student asked
        solution (str): provided step-by-step textbook solution
        history (str): current conversation history
        state (str, optional): the state of the tutorbot i.e. "deciding" | "initial" | "use_python" | "no_python" | "received_python". Defaults to "deciding".
        response_id (str, optional): current step number. Defaults to None.
        description (str, optional): the description of the code to be generated; state must be "use_python". Defaults to None.
        python_output (str, optional): the python output calculated from the most recently generated python code; state must be "received_python". Defaults to None.

    Returns:
        str: tutorbot gpt's response
    """
    tutor_history = [
            {"role": "system", "content": get_tutorbot_sys_prompt(question, solution, history, state=state, description=description, python_output=python_output)},
            {"role": "user", "content": ""},
        ]
    
    return run_chat_completion(tutor_history)

def run_student_gpt(question, history):
    """
    Run student gpt.

    Args:
        question (str): initial question that the student asked
        history (str): current conversation history

    Returns:
        str: student gpt's response
    """
    student_history = [
            {"role": "system", "content": get_student_sys_prompt(question, history)},
            {"role": "user", "content": ""}
        ]
    
    return run_chat_completion(student_history)
    
def validate_json_string(response: str):
    """
    Corrects the format of JSON string.

    Args:
        response (str): JSON string

    Returns:
        str: correctly formatted JSON string
    """
    # Add any missing commas
    response = response.replace("\"\n\"", "\",\n\"")
        
    # Strip any extra text before first bracket and after last bracket
    response = response[response.find('{'):]
    response = response[:response.rfind('}')+1]
        
    return response

def create_tutor_response(question, solution, history, conversation_file, conversation_json, state="deciding", response_id=None, description=None, python_output=None):
    """
    Generates a JSON object that represents the tutorbot's response.

    Args:
        question (str): initial question that the student asked
        solution (str): provided step-by-step textbook solution
        history (str): current conversation history
        conversation_file (str): name of the destination file to output the raw conversation
        state (str, optional): the state of the tutorbot i.e. "deciding" | "initial" | "use_python" | "no_python" | "received_python". Defaults to "deciding".
        response_id (str, optional): current step number. Defaults to None.
        description (str, optional): the description of the code to be generated; state must be "use_python". Defaults to None.
        python_output (str, optional): the python output calculated from the most recently generated python code; state must be "received_python". Defaults to None.

    Returns:
        object: JSON object that represents the tutorbot's response
    """
    response = run_tutorbot_gpt(question, solution, history, state, response_id, description, python_output)
    
    # Fix formatting for JSON object
    response = validate_json_string(response)
    
    # print("\n" + response)
    # print("--------------------------------------")
    conversation_file.write("\n\n" + response)
    
    try:
        response_json = json.loads(response, strict=False)
        conversation_json["conversations"].append({
            "from": f"{state}_state",
            "value": response_json
        })
    except Exception as err:
        # print(err)
        # exit()
        response_json = create_tutor_response(question, solution, history, conversation_file, conversation_json, state, response_id, description, python_output)
    
    return response_json

def create_python_response(question, solution, history, conversation_file, conversation_json, response_id=None, description=None, python_output=None):
    state = "use_python"
    response = run_tutorbot_gpt(question, solution, history, state, response_id, description, python_output)
    
    # Fix formatting for JSON object
    response = validate_json_string(response)
    
    # print("\n" + response)
    # print("--------------------------------------")
    conversation_file.write("\n\n" + response)
    
    try:
        response_json = json.loads(response, strict=False)
    except Exception as err:
        # print(err)
        # exit()
        response_json = create_python_response(question, solution, history, conversation_file, conversation_json, response_id, description, python_output)
    
    return response_json

def create_student_response(question, history, conversation_file, conversation_json):
    """
    Generates student response message.

    Args:
        question (str): initial question that the student asked
        history (str): current conversation history
        conversation_file (str): name of the destination file to output the raw conversation

    Returns:
        str: student's response message
    """
    response_msg = run_student_gpt(question, history)
    
    # print("\n" + response_msg)
    # print("--------------------------------------")
    conversation_file.write("\n\n" + response_msg)
    conversation_json["conversations"].append({
        "from": "student",
        "value": response_msg
    })
    
    return response_msg
    

def simulate_question(question, solution, filename, conversation_json):
    """
    Generate a conversation between tutorbot gpt and student gpt for a single question.

    Args:
        question (str): question asked by the student
        solution (str): provided step-by-step textbook solution
        filename (str): name of the destination file to output the raw conversation
    """
    # Open conversation file
    conversation_file = open(filename, "w", encoding="utf-8")
    actual_conversation_file = open("data_gen/conversation_generation/v4/actual_conversation.txt", "w", encoding="utf-8")
    
    # Initialize conversation history + response id counter
    conversation_history = ""
    response_id = 1
    
    # Read in initial student question
    student_response_msg = f"Please help me with this question: \"{question}\" Break it down and guide me through step by step."
    conversation_file.write(student_response_msg)
    
    conversation_json["conversations"].append({
        "from": "student",
        "value": student_response_msg
    })
    
    student_convo = f"Student: {student_response_msg}"
    print(student_convo)
    print("--------------------------------------")
    conversation_history += student_convo
    actual_conversation_file.write(student_convo)
    
    # Get tutor response
    # Enter initial_state
    tutor_response_json = create_tutor_response(question, solution, conversation_history, conversation_file, conversation_json, "initial", response_id=response_id)
    
    tutor_response_msg = tutor_response_json["Tutorbot Response"]
    tutor_convo = f"\n\nTutorbot: {tutor_response_msg}"
    conversation_history += tutor_convo
    actual_conversation_file.write(tutor_convo)
    print(tutor_convo)
    print("--------------------------------------")
    
    # While there are still more steps in the solution
    while tutor_response_json["Step State"] != "t":
        response_id += 1
        
        # Get student response
        student_response_msg = create_student_response(question, conversation_history, conversation_file, conversation_json)
        
        if student_response_msg.startswith("Student:"):
            student_convo = f"\n\n{student_response_msg}"
        else:
            student_convo = f"\n\nStudent: {student_response_msg}"
        conversation_history += student_convo
        actual_conversation_file.write(student_convo)
        print(student_convo)
        print("--------------------------------------")
        
        # Enter deciding_state (default)
        tutor_response_json = create_tutor_response(question, solution, conversation_history, conversation_file, conversation_json, state="deciding", response_id=response_id)
        
        # If bot decides to use Python
        if tutor_response_json["Use Python"] == "y" and "Description" in tutor_response_json:
            
            description = tutor_response_json["Description"]
            print("Description:", description)
            
            # Enter use_python_state
            tutor_response_json = create_python_response(question, solution, conversation_history, conversation_file, conversation_json, response_id=response_id, description=description)
            
            # Run Python code and get output
            python_code = tutor_response_json["Python"]["Python Code"]
            print("\n\n" + python_code)
            result_variables = tutor_response_json["Python"]["Result Variable"]
            python_output = run_python(python_code, result_variables)
            
            # Generate new code if old code could not be run
            while python_output == "Unable to execute Python code.":
                print("regenerating python code\n")
                
                # Enter use_python_state
                tutor_response_json = create_python_response(question, solution, conversation_history, conversation_file, conversation_json, response_id=response_id, description=description)
                
                # Run Python code and get output
                python_code = tutor_response_json["Python"]["Python Code"]
                print("\n\n" + python_code)
                result_variables = tutor_response_json["Python"]["Result Variable"]
                python_output = run_python(python_code, result_variables)
            
            # Add python code
            conversation_json["conversations"].append({
                "from": "use_python_state",
                "value": tutor_response_json
            })
            
            # Add python output
            conversation_json["conversations"].append({
                "from": "python_output",
                "value": python_output
            })
            
            print("\nPython output:", python_output)
            conversation_file.write("\n\nPython Output: " + str(python_output))
            
            # Feed back student question and python output to tutorbot
            # Enter received_python_state
            tutor_response_json = create_tutor_response(question, solution, conversation_history, conversation_file, conversation_json, state="received_python", response_id=response_id, description=description, python_output=python_output)
            
            # Add tutorbot response to conversation
            tutor_response_msg = tutor_response_json["Tutorbot Response"]
            
            tutor_convo = f"\n\nTutorbot: {tutor_response_msg}"
            conversation_history += tutor_convo
            actual_conversation_file.write(tutor_convo)
            print(tutor_convo)
            print("--------------------------------------")
        else:
            tutor_response_json = create_tutor_response(question, solution, conversation_history, conversation_file, conversation_json, state="no_python", response_id=response_id)
            tutor_response_msg = tutor_response_json["Tutorbot Response"]
            tutor_convo = f"\n\nTutorbot: {tutor_response_msg}"
            conversation_history += tutor_convo
            actual_conversation_file.write(tutor_convo)
            print(tutor_convo)
            print("--------------------------------------")
        
    conversation_file.close()
    actual_conversation_file.close()
    
def get_conversation(filename):
    """
    Retrieve the raw conversation from the destination file.

    Args:
        filename (str): name of the file that stores the conversation

    Returns:
        str: full conversation
    """
    with open(filename, "r", encoding="utf-8") as conversation_file:
        conversation = conversation_file.read()
    return conversation

def run_on_problem_set(problem_set_file: str, solutions_json_filename: str, iteration, starting_row_idx=-1, user_ending_row_idx=-1):
    """
    Generate conversations sequentially for a problem set. File must be a .csv with a "Question" and "Solution" column.

    Args:
        problem_set_file (str): name of the problem set file
    """
    try:
        df = pd.read_csv(problem_set_file)
    except FileNotFoundError:
        print("Problem set file not found.")
        exit()
        
    if starting_row_idx >= len(df.index):
        raise IndexError("Starting row index is out of bounds.")
        
    # Assign ending row index if included
    if user_ending_row_idx != -1:
        ending_row_idx = user_ending_row_idx
    else:
        ending_row_idx = len(df.index)
        
    if ending_row_idx > len(df.index):
        raise IndexError("Ending row index is out of bounds.")
    
    # Get column for 'conversation' or create if it doesn't exist
    conversation_col_name = f'Conversation {iteration}'
    df[conversation_col_name] = df.get(conversation_col_name, None)
    
    # Find row with the first empty 'conversation' cell if starting_row_idx is not specified
    is_full = True
    if starting_row_idx == -1:
        for index, row in df.iterrows():
            if pd.isnull(df.loc[index, conversation_col_name]):
                starting_row_idx = index
                is_full = False
                break
            
    if ending_row_idx < starting_row_idx:
        raise IndexError("Ending row index must be greater than or equal to starting row index.")
                
    if is_full and starting_row_idx == -1:
        # If dataframe is full, starting_row_idx should never have updated
        # Only care about full dataframe when starting_row_idx is unspecified
        print("Dataframe is full")
    else:
        # Else if we specified a starting_row_idx, always start at that idx
        print(f"Starting at row {starting_row_idx}")
        
        # get solutions json to grab detailed solutions
        with open(solutions_json_filename, 'r', encoding='utf-8') as solution_json_file:
            solutions_contents = solution_json_file.read()
        solutions_json = json.loads(solutions_contents, strict=False)
        
        for index, row in df.iloc[starting_row_idx : ending_row_idx].iterrows():
            question = row['Question']
            solution = solutions_json[index]["Detailed Solution"]
            print(solution + '\n')
            
            # Put conversation into conversation file + conversation json
            conversation_json = {
                "id": f"conversation_{index}",
                "conversations": []
            }
            
            simulate_question(question, solution, 'row_conversation.txt', conversation_json)
            
            # Set column to be contents of conversation file 
            df.at[index, conversation_col_name] = conversation_json
            
            # Generate filename with random string of 6 characters
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            conversation_json_filename = f"{conversation_json['id']}_v4_{random_string}"
            
            conversation_json_file = f"datasets/v4/conversation_jsons/{conversation_json_filename}.json"
            
            json.dump(conversation_json, open(conversation_json_file, "w", encoding="utf-8"), indent=4)
                
            # Overwrite csv file to save dataframe
            df.to_csv(problem_set_file, index=False)
            
            time.sleep(1)

if __name__ == '__main__':
    # iteration = which round the conversation generation is on -> make sure to start at 1
    for iteration in range(1, 2):
        run_on_problem_set('problem set.csv', 'datasets/v4/solutons_array.json', iteration)