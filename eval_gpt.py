import sys
sys.path.append(('../'))
sys.path.append(('../../'))
import os
import openai
from openai import OpenAI
import json
import base64
import requests
# openai.api_key = os.getenv("OPENAI_API_KEY")
from resource_api import openai_key
client = OpenAI()
import json
import requests
import openai
from openai import OpenAI


def extract_factuality_score_and_justification(evaluation_result):
    # Extract score and justification from the evaluation result
    try:
        # Find the line that contains "Factuality Score"
        score_line = [line for line in evaluation_result.split('\n') if "Factuality Score" in line][0]
        # Extract the score and remove commas
        score = score_line.split(':')[-1].strip().replace(',', '')

        # Find the line that contains "Justification"
        justification_line = [line for line in evaluation_result.split('\n') if "Justification" in line][0]
        justification = justification_line.split(':', 1)[-1].strip()

        return int(score), justification
    except Exception as e:
        print(f"Error extracting factuality score and justification: {e}")
        return None, None
    
    

def evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="generation"):
    # Custom prompt for factuality evaluation
    prompt = f"""
        You will be provided with two types of questions: generation questions and description questions.
        For each, you will evaluate the **factuality** of the "generated_answer" or "generated_description" 
        against the "ground_truth" or "ground_truth_description" respectively. Your task is to assess how well 
        the generated response aligns with the factual content of the ground truth and assign a **factuality score** 
        from 1 to 10 based on the following criteria:

        1. **Factuality (core importance)**:
        - **10-9:** The generated response is fully factually correct and has the same meaning as the ground truth, even if phrased differently.
        - **8-7:** The response is mostly correct but may be missing minor details or contain slightly less important deviations.
        - **6-5:** The response is partially correct but has a noticeable factual error or significant missing information.
        - **4-3:** The response has major factual errors or lacks crucial elements of the ground truth.
        - **2-1:** The response is nonsensical, completely incorrect, or irrelevant.

        2. **Relevance and Detail**:
        - More detail does not always improve the score; added details should be factually relevant.
        - If the generated response contains excessive or irrelevant details (e.g., adding personal information when only appearance is requested), lower the score accordingly.

        ### Task Type: {task_type.capitalize()}
        - **Image ID**: {image_id}
        - **Question**: {question}
        - **Generated Answer**: {generated_answer}
        - **Ground Truth**: {ground_truth}

        Please evaluate the factuality of the generated response based on the rubric above, and return a score (1-10) along with a short justification.
        Example Output:
        {{
            "Factuality Score": [Insert score from 1-10],
            "Justification": "[Optional] Provide a brief justification explaining why the factuality score was assigned."
        }}
    """

    # Call the OpenAI API to evaluate factuality
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at evaluating the factuality of responses."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 700,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    evaluation_result = response.json()['choices'][0]['message']['content']

    # print(evaluation_result)
    return evaluation_result


output_folder = 'result/llava_1.5_7b_vanilla_model_forget_10'

### fill_blank_task
# fill_blank_file = 'result/llava_1.5_7b_vanilla_model_forget_10/llava_1.5_7b_vanilla_model_forget_10_fill_blank_results_official.json'
# fill_blank_scores = []
# with open(fill_blank_file, "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line.strip())
#         image_id = data['id']
#         question_type = data['question type']
#         question = data['question']
#         model_answer = data['model_answer']
#         ground_truth = data['ground_truth']
#         prompt = data['prompt']
#         generated_answer = data['generated_answer']
#         image_textual_correct = data['image_textual_correct']
#         image_textual_questions = data['image_textual_questions']
#         pure_text_correct = data['pure_text_correct']
#         pure_text_questions = data['pure_text_questions']
#         evaluation = evaluate_factuality_questions(image_id, question, model_answer, ground_truth, task_type="generation")
#         factuality_score, justification = extract_factuality_score_and_justification(evaluation)
        
#         if factuality_score is not None:
#             fill_blank_scores.append(factuality_score)
        
#         result = {
#                 "id": image_id,
#                 "question_type": question_type,
#                 "question": question,
#                 "model_answer": model_answer,
#                 "ground_truth": ground_truth,
#                 "prompt": prompt,
#                 "generated_answer": generated_answer,
#                 "image_textual_correct": image_textual_correct,
#                 "image_textual_questions": image_textual_questions,
#                 "pure_text_correct": pure_text_correct,
#                 "pure_text_questions": pure_text_questions,
#                 "factuality_score": factuality_score,
#                 "justification": justification
#             }
            
#         with open(f'{output_folder}/llava_1.5_7b_vanilla_model_forget_10_fill_blank_results_official_add_fact.json', 'a') as f:
#             f.write(json.dumps(result) + "\n")
        
        
### generation_task
generation_file = 'result/llava_1.5_7b_vanilla_model_forget_10/llava_1.5_7b_vanilla_model_forget_10_generation_results_official.json'
generation_scores = []

with open(generation_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        id = data['id']
        question_type = data['question type']
        question = data['question']
        model_answer = data['model_answer']
        ground_truth = data['ground_truth']
        prompt = data['prompt']
        generated_answer = data['generated_answer']
        bleu_img = data['bleu_img']
        rouge1_img = data['rouge1_img']
        rouge2_img = data['rouge2_img']
        rougeL_img = data['rougeL_img']
        image_textual_questions = data['image_textual_questions']
        bleu_text = data['bleu_text']
        rouge1_text = data['rouge1_text']
        rouge2_text = data['rouge2_text']
        rougeL_text = data['rougeL_text']
        pure_text_questions = data['pure_text_questions']
        evaluation = evaluate_factuality_questions(id, question, model_answer, ground_truth, task_type="generation")
        factuality_score, justification = extract_factuality_score_and_justification(evaluation)
        
        if factuality_score is not None:
            generation_scores.append(factuality_score)
            
        result = {
            "id": id,
            "question_type": question_type,
            "question": question,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "generated_answer": generated_answer,
            "bleu_img": bleu_img,
            "rouge1_img": rouge1_img,
            "rouge2_img": rouge2_img,
            "rougeL_img": rougeL_img,
            "image_textual_questions": image_textual_questions,
            "bleu_text": bleu_text,
            "rouge1_text": rouge1_text,
            "rouge2_text": rouge2_text,
            "rougeL_text": rougeL_text,
            "pure_text_questions": pure_text_questions,
            "factuality_score": factuality_score,
            "justification": justification
            }
        
        with open(f'{output_folder}/llava_1.5_7b_vanilla_model_forget_10_generation_results_official_add_fact.json', 'a') as f:
            f.write(json.dumps(result) + "\n")
        


