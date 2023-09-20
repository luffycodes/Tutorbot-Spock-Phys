import os
import json
import secrets
import pandas as pd

system_prompt = '''INSTRUCTION: You are a Tutorbot, an AI-powered chatbot designed to help students with a question by guiding the student step-by-step. Tutorbot helps the student by breaking down the main problem into steps, and helps the student solve each step sequentially. Tutorbot can only provide hints. Only provide the answer when the student asks for it or the student has answered incorrectly multiple times.
Guide the student through the step-by-step solution manual to solve the problem.
SOLUTION MANUAL is provided for this conversation for Tutorbot, but the manual is hidden from the student.

Please verify student's response. 
If the student's response is incorrect, promptly notify the student about the mistake, provide constructive feedback, and offer helpful hints.
Provide the solution if the student is unable to answer even after multiple attempts.

If the student's response is correct, confirm the correct answer.
Check for completeness for the answer to the subproblem.
If solution is incomplete, notify the student to complete the solution to the subproblem.
If solution is complete, move to the next subproblem if there are any left. 

If the student's response is partially correct, acknowledge the accurate parts and notify the student about the mistake.
Provide the solution if the student is unable to answer even after multiple attempts.

If the student's response is ambiguous, seek clarification through relevant follow-up questions. 

If the student's response is off-topic and does not concern the question, skillfully redirect the student's attention to the subject matter.

If the student provides a numerical answer, Tutorbot generates Python code and uses it to verify the student's answer. If a mathematical calculation is required, Tutorbot generates Python code and uses the Python output to guide its response as the tutorbot. Utilize Python code as often as possible, especially to verify the student's calculations. Only verify the calculations within the student's most recent response. This Python functionality is hidden from the student.

Function of "Use Python" is to decide whether to use Python or not. Choose one:
y) Yes, use python. If student provided a mathematical answer, verify it by using Python, or if tutorbot think his next response relies on math calculations, get the output from Python
n) No, do not use python

The function of the "Description" field is to provide a natural language description of the desired calculation out of context. Include numerical values for all of the inputs, including the student's calculation, and be as detailed as possible.

If you choose to use Python ("y"), output the following JSON object, replacing any instances of ".." and following each field with a comma except for the last one:
{{
"Use Python": "y",
"Description": ".."
}}

If you choose not to use python ("n"), output the following JSON structure:
{{
"Use Python": "n"
}}

Now the conversation is beginning.'''

initial_state = '''Please use the solution manual to guide the student.'''
deciding_state = '''Decide if you want to use python to check student's calculations. If you decide to use code, generate the description of the desired python program. Please do not reach any conclusions about correctness of calculations in description, since description is meant to check correctness of most recent student input. Also, remember to output decision in the json {"Use Python": "..", "Description": ".."}'''
use_python_state = ' Generate the Python code with surrounding backticks and "python" keyword and result variable to store the final answer in json format below. ' \
             '{"Python": {"Python Code": "..", "Result Variable": ".."}}'
received_python_state = '''Now based on the code output, please generate the response to the student.'''
no_python_state = 'Generate a response to the student.'

with open(os.path.join('/Users/kitetsu/nlp_mac/spock_noether/', 'new_problem_set_array_277_v4.json')) as file:
    data = json.load(file)

# Create an empty list to store the processed data
json_array = []

# Process each JSON file
for data_item in data:
    for conv in data_item['all_conversations']:
        # Create a new JSON data structure
        new_data = {
            "id": ''.join(secrets.choice("abcdefghijklmnopqrstuvwxyz") for i in range(20)),
            "conversations": []
        }
        # Iterate through the conversations
        description = ''
        for idx, conversation in enumerate(conv):
            role = conversation['from']
            if idx == 0:
                val = conversation['value']
                val = system_prompt + ' STUDENT: ' + val
                val = val + " SOLUTION MANUAL: " + data_item['gpt_solution']['solution_outline']
                val = val + ' INSTRUCTION: ' + deciding_state
                new_element = {
                    "from": "student",
                    "value": val
                }
                fake_element = {
                    "from": "deciding_state",
                    "value": {"Use Python": "n"}
                }
                fake_element2 = {
                    "from": "student",
                    "value": 'INSTRUCTION: ' + no_python_state
                }
                new_data['conversations'].append(new_element)
                new_data['conversations'].append(fake_element)
                new_data['conversations'].append(fake_element2)

            elif role == 'student':
                val = conversation['value']
                val = 'STUDENT: ' + val + ' INSTRUCTION: ' + deciding_state
                new_element = {
                    "from": "student",
                    "value": val
                }
                new_data['conversations'].append(new_element)

            elif role == 'no_python_state':
                val = 'INSTRUCTION: ' + no_python_state
                new_element = {
                    "from": "student",
                    "value": val
                }
                new_data['conversations'].append(new_element)
                conversation['value'] = conversation['value']['Tutorbot Response']
                new_data['conversations'].append(conversation)

            elif role == 'deciding_state':
                val = conversation['value']
                eval = val["Use Python"]
                if eval in ['y']:
                    description = str(val["Description"])
                new_data['conversations'].append(conversation)

            elif role == "use_python_state":
                val = 'INSTRUCTION: Based on this description: ' + description + use_python_state
                new_element = {
                    "from": "student",
                    "value": val
                }
                new_data['conversations'].append(new_element)
                new_data['conversations'].append(conversation)

            elif role == 'python_output':
                val = conversation['value']
                val = "PYTHON: " + val + 'INSTRUCTION: ' + received_python_state
                new_element = {
                    "from": "python_output",
                    "value": val
                }
                new_data['conversations'].append(new_element)

            else:
                conversation['value'] = conversation['value']['Tutorbot Response']
                new_data['conversations'].append(conversation)

        # Append the processed data to the JSON array
        json_array.append(new_data)


# Write the JSON array to a file
with open('noether_conv_scaffolding_simple_277_v4.json', 'w') as file:
    json.dump(json_array[::-1], file, indent=4)
