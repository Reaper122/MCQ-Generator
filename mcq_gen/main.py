import random
import csv
from transformers import pipeline
import spacy

# Step 1: Summarize the text
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Step 2: Extract key information
def extract_key_information(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Step 3: Generate questions based on entities
def generate_questions(summary, entities):
    questions = []
    for entity in entities:
        if entity[1] == 'PERSON':
            questions.append(f"Who is {entity[0]}?")
        elif entity[1] == 'ORG':
            questions.append(f"What is {entity[0]}?")
        elif entity[1] == 'GPE':
            questions.append(f"Where is {entity[0]} located?")
        # Add more entity types as needed
    return questions

# Step 4: Generate MCQ options
def generate_options(correct_answer, entities):
    options = [correct_answer]
    distractors = [entity[0] for entity in entities if entity[0] != correct_answer]
    if len(distractors) >= 3:
        options.extend(random.sample(distractors, 3))
    else:
        options.extend(distractors)
    random.shuffle(options)
    return options

# Step 5: Combine everything into an MCQ generator
def mcq_generator(text):
    summary = summarize_text(text)
    entities = extract_key_information(summary)
    questions = generate_questions(summary, entities)
    
    mcqs = []
    for question in questions:
        correct_answer = next((entity[0] for entity in entities if entity[0] in question), None)
        if correct_answer:
            options = generate_options(correct_answer, entities)
            mcqs.append({
                'question': question,
                'options': options,
                'answer': correct_answer
            })
    return mcqs

# Step 6: Save MCQs to CSV
def save_mcqs_to_csv(mcqs, filename="mcqs.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Option 1', 'Option 2', 'Option 3', 'Option 4', 'Answer'])
        for mcq in mcqs:
            row = [mcq['question']] + mcq['options'] + [mcq['answer']]
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving".
    """
    
    mcqs = mcq_generator(text)
    save_mcqs_to_csv(mcqs)

    print(f"MCQs saved to 'mcqs.csv'")
