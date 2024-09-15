# Loading Libraries 
import os
from tqdm import tqdm
import numpy as np
import pandas as pd                   
from pathlib import Path

from sklearn.metrics import accuracy_score

from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings

# Parameters Setting 
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here"  

ChunksFileName = "./intermediate_data/EU_REG_QA_Chunks.csv"
ResultsFileName = "./results/QA_Matching_Results.csv"
AccuracyResultsFileName = "./results/QA_Matching_Accuracy_Results.csv"

max_chunks_to_read = 3

# Initialize the tqdm library to add progress bars to loops and iterable objects
tqdm.pandas()

def Save_Results(df, FileName):
    try:
        df.to_csv(FileName, index=False)
        print(f"Data successfully saved to {FileName}")
    except IOError as e:
        print(f"Failed to write to {FileName}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def Answer_Question(row, FAISS_retriever):
    """ Main processes for answering user's question """

    # retieve documents from vectorized data, having 'subject' as input
    retrieved_docs = FAISS_retriever.similarity_search_with_score(row["Question"], k=max_chunks_to_read)   
 
    answers_ranks = []
        
    if len(retrieved_docs) > 0:                               # Checking when the Question is not relevant with corpus
        for doc in retrieved_docs: 
            lines = doc[0].page_content.split('\n')
            firstline  = lines[0].strip()                     # the first sentence of each Chunk is of "AnsewerNo: x" form
            number = int(firstline.split()[1].strip())        # Assumes format "AnsewerNo: x" to get the number of the answer
            answers_ranks.append(number)
    return answers_ranks  

def batch_process():
    """"Running the batch process by reading the questions from the EU_REG_QA_Chunks.csv file """
    
    results = {
        "Regulation": [], 
        "Language": [], 
        "Model": [], 
        "Question_No": [],
        "Question_Text": [],
        "FAISS 1st Ranked": [], 
        "FAISS 2nd Ranked": [], 
        "FAISS 3rd Ranked": [], 
        "FAISS Best Choice": [], 
        }
    
    df = pd.read_csv(ChunksFileName)
    
    for category, group in df.groupby(["Regulation", "Language", "Model"]):
           
        """ Initialization of models for each distinct combination of Regulation, Language and Model """    
        df_File_questions = df[(df["Regulation"] == category[0]) & (df["Language"] == category[1]) & (df["Model"] == category[2])]
        df_File_questions = df_File_questions[["Regulation","Language", "Model", "FAISS_FileName", "Question_No", "Question"]] 
        df_File_questions = df_File_questions.drop_duplicates()
     
        """ Initialization of Embedding Model """
        embedding_model = OpenAIEmbeddings(model=df_File_questions["Model"].iloc[0])
        
        """ Initialization of Vector Databases """
        FAISS_storage_file = df_File_questions["FAISS_FileName"].iloc[0]
    
        # Load the FAISS index from disk
        if not Path(FAISS_storage_file).exists():
            raise ValueError(f"{FAISS_storage_file}  directory does not exist, please run ingest python file first")
    
        vector_store = LocalFileStore(FAISS_storage_file)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model, vector_store, namespace=embedding_model.model)
        FAISS_retriever = FAISS.load_local(FAISS_storage_file, cached_embedder, allow_dangerous_deserialization=True)
        
        """ Finding relevant chunks for particular Questions and update of the results dictionary """
        for index, row in df_File_questions.iterrows():
            
            print(f"Regulation: {row["Regulation"]} Language: {row["Language"]} Model: {row["Model"]} QNo:{row["Question_No"]}")
    
            results['Regulation'].append(row["Regulation"]) 
            results['Language'].append(row["Language"]) 
            results['Model'].append(row["Model"]) 
            results['Question_No'].append(row["Question_No"]) 
            results['Question_Text'].append(row["Question"])
            
            answers_ranks = Answer_Question(row,FAISS_retriever)
          
            if len(answers_ranks)>0:
                results["FAISS 1st Ranked"].append(answers_ranks[0])
                results["FAISS 2nd Ranked"].append(answers_ranks[1])
                results["FAISS 3rd Ranked"].append(answers_ranks[2])  
                if row["Question_No"] in answers_ranks: 
                    # overriding the best choice if the correct answer is within the group of top 3 alternative chunks
                    results["FAISS Best Choice"].append(row["Question_No"])  
                else:
                    results["FAISS Best Choice"].append(answers_ranks[0])
            else:
                results["FAISS 1st Ranked"].append(0)
                results["FAISS 2nd Ranked"].append(0)
                results["FAISS 3rd Ranked"].append(0) 
                results["FAISS Best Choice"].append(0)
    
    results_df = pd.DataFrame(results)
    results_df  = results_df.drop_duplicates(subset=["Regulation", "Language", "Model", "Question_No"], keep="first")
    Save_Results(results_df, ResultsFileName)
    return results_df

def performance_assessment(results_df):
    """ Assessment of Accuracy results """
    
    accuracy_results = {
        "Regulation": [], 
        "Language": [], 
        "Model": [],
        "Accuracy": [],
        "Accuracy_alt": [],    # to count for the cases the correct answer is within the  top 3 alternative chunks givren by the model
        }
    
    grouped_results_df = results_df.groupby(["Regulation", "Language", "Model"]).agg({"Question_No": list, "FAISS 1st Ranked": list, "FAISS Best Choice": list}).reset_index()
    
    for index, row in grouped_results_df.iterrows():
        actual_values = np.array(row["Question_No"])
        predicted_values = np.array(row["FAISS 1st Ranked"]) 
        predicted_values_alt = np.array(row["FAISS Best Choice"]) 
        
        accuracy = accuracy_score(actual_values, predicted_values)
        accuracy_alt = accuracy_score(actual_values, predicted_values_alt)
    
        print(f"{row["Regulation"]} | {row["Language"]} | {row["Model"]} | {accuracy * 100:.2f}% | {accuracy1 * 100:.2f}%")
        
        accuracy_results["Regulation"].append(row["Regulation"])
        accuracy_results["Language"].append(row["Language"])
        accuracy_results["Model"].append(row["Model"])
    
        accuracy_results["Accuracy"].append(accuracy)
        # accuracy evaluation by considering the correct answer is within the  top 3 alternative chunks givren by the model
        accuracy_results["Accuracy_alt"].append(accuracy_alt) 
    
    accuracy_results_df = pd.DataFrame(accuracy_results)
    accuracy_results_df.head()
    Save_Results(accuracy_results_df, AccuracyResultsFileName)

if __name__ == "__main__":

    """ Main Process """                        
    results_df  = batch_process()                              # Running the batch process by reading the questions from the EU_REG_QA_Chunks.csv file
    performance_assessment(results_df)                         # Assessment of Accuracy results