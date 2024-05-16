import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import gradio as gr

import csv
import random
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from together_ai import gemmaResponse


# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token='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')
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", token='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')

# Read the CSV file
csv_path = "indeed_jobs.csv" #indeed job scrape
jobs = []
with open(csv_path, "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header

    for row in reader:
      job = {}
      for i in range(len(header)):
            job[header[i]] = row[i]
      jobs.append(job)
        # for row in reader:
    #     jobs.append(dict(zip(header, row)))

# Read the linkedinjobs.csv file
linkedin_csv_path = "linkedinjobs.csv" #first linkedin job scrape
linkedin_jobs = []
with open(linkedin_csv_path, "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    for row in reader:
        job = {}
        for i in range(len(header)):
            job[header[i]] = row[i]
        linkedin_jobs.append(job)

# Read the linkedin_job_details.csv file
linkedin_job_details_csv_path = "linkedin_job_details.csv" #second linkedin job scrape with descriptions
linkedin_job_details = []
with open(linkedin_job_details_csv_path, "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    for row in reader:
        job = {}
        for i in range(len(header)):
            job[header[i]] = row[i]
        linkedin_jobs.append(job)

# Combine the job information from both sources
all_jobs = jobs + linkedin_jobs + linkedin_job_details

from google.colab import drive
drive.mount('/content/drive')

# Initialize ChromaDB
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-albert-small-v2")

documents = []

for job in all_jobs:
  document = ""
  for key, value in job.items():
    document += f"{value} "
  documents.append(document.strip())
  # documents.append(f"{job['Title']} {job['Duration']} {job['Company']} {job['Location']}")

ids = list(range(len(documents)))

ids = [str(id) for id in ids]

client = chromadb.PersistentClient(path="./docs_cache/")

collection = client.get_or_create_collection(name="job_collection", embedding_function=sentence_transformer_ef)

collection.add(documents=documents, ids=ids)

collection.count()

# Function to recommend a job based on user input
def recommend_job(user_input):
    # query_embedding = sentence_transformer_ef.encode(user_input)
    result = collection.query(query_texts = [user_input], n_results=3)
    # recommended_job = jobs[int(result[0].id)]
    return result['documents'][0]

# Define the Gradio interface
def chatbot_interface(query):
    context = "\n".join(user_query)

    input_text = f"Answer the question to the best of your ability strictly from the context.\n\nContext:{context}\n\nQuestion:{user_query}"
    # idnput_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    # outputs = model.generate(**input_ids, max_new_tokens=200)
    # recommendation_text = tokenizer.decode(outputs[0])
    recommendation_text = gemmaResponse(st.secrets['TOGETHERAI_API_KEY'], input_text)
    print("Job Recommendation:")
    print(recommendation_text)
    return recommendation_text

# import os
# os.environ["GRADIO_SERVER_NAME"] = "tensorflow"  # Set a dummy server name to disable the Gradio queue

# Main code
print("Welcome to the Job Recommendation Chatbot!")

while True:
    user_input = input("Enter your query or type 'recommend' to get a job recommendation: ")

    if user_input.lower() == "recommend":
        inputs = gr.Textbox(label="Enter your job preferences")
        outputs = gr.Textbox(label="Job Recommendation")
        gr.Interface(fn=chatbot_interface, inputs=inputs, outputs=outputs, title="Job Recommendation Chatbot", theme='freddyaboulton/dracula_revamped').launch()

    else:
        query = user_input
