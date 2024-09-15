## Embedding models vs. Languages - Evaluation on EU Regulations (OpenAI, LangChain)
This application leverages **OpenAI** and **LangChain** to assess the accuracy of embedding models across multiple languages. As benchmarks, it utilizes a set of Q&A documents related with EU Regulations, written in different European languages. The application compares the answers selected by embedding models with the authoritative answers provided in the EU's official Q&A documents.

Initially, the Q&A files are embedded and stored using **OpenAI** services combined with the **FAISS** library, enabling efficient retrieval of relevant content. A secondary process then leverages these embeddings to accurately match the questions with their corresponding answers according to the EU Q&A documents.

This application is not intended as an academic research tool aimed at producing definitive conclusions on model performance. Rather, it serves as a practical demonstration of how various evaluation tasks can be performed using **GenAI** models. By focusing on real-world use cases, it highlights how embedding models and retrieval mechanisms can support multilingual content processing, making it particularly relevant for professionals in AI and IT.

### Purpose 
European banking, payments, and financial regulations are both crucial and highly intricate. Retrieving relevant information from extensive regulatory documents is labor-intensive and time-consuming. **GenAI** models offer significant potential in automating this process. However, their performance may vary across different European languages, posing a challenge in achieving consistent performance. This application provides a paradigm for building processes that evaluate the effectiveness of information retrieval models across various languages, as a preliminary step to ensure that GenAI solutions are robust and efficient in multilingual regulatory environments. 

### Process Overview 
The ingestion and querying of the EU Regulations Q&As follow the below steps: 
![Process Overview](https://github.com/antdragiotis/Embedding-models-vs-Languages-performance/blob/main/assets/Process_Overview.PNG)

### Features
- **Source Data**: The application uses as source data file the following EU regulatory documents:
  - **AI Act** (EU regulation 2024/1689) to promote the uptake of human-centric and trustworthy Artificial Intelligence and ensure better conditions for the development and use of this innovative technology. **26** Questions & Answers written in English, German, Greek, Hungarian, Italian and Romanian.
  - **Late Payment Directive** (EU Directive 2011/7/EU) on combating late payment in commercial transactions, to protect European businesses, particularly SMEs, against late payment. **13** Questions & Answers written in English, German, Greek, Hungarian, Italian and Romanian.
  - **Single Currency Package** as a proposal of the European Commission to ensure that citizens and businesses can continue to access and pay with euro banknotes and coins across the euro area, and to set out a framework for a possible new digital form of the euro that the European Central Bank may issue in the future, as a complement to cash. **39** Questions & Answers written in English, French and German.
- **Embedding Models**: The **OpenAI** Embedding Models they are used for the embedding of the source files are: 
  - **text-embedding-3-small**
  - **text-embedding-ada-002**
  - **text-embedding-3-large**  
- **Ingestion**: The 'Ingestion' process is performed by the *EVAL_EMBEDDINS_EU_REG_INGEST.py* Python file. This process reads the Source Data and splits the Source documents text into chunks, assigning also to each chunk information about the relevant question number. It saves this information into the *EU_REG_QA_Chunks.csv* file in the *intermediate_data* directory. Next to this, the process uses **FAISS** library to store the vectorized data to the *FAISS_storage* directory. 
- The *EVAL_EMBEDDINGS_EU_REG_BATCH.py* Python file executes three key processes to retrieve the questions in the Source Files and assess the accuracy of the embedding models responses:
  - **Question Auto-Submission**: Reads the *EU_REG_QA_Chunks.csv* file in the *intermediate_data* directory to get the Questions to be submmitted for retrieval.  
  - **Retrieval of Embedded Data** each Question is compared with the embedded data to find the most relevant chunks (top 3) based on their proximity with the Question.
  - **Accuracy Evaluation** The selected chunks and their corresponding question numbers are compared with the submitted Question number to find if the selection was successful anf assess the accuracy of the embedding models. The accuracy is evaluated in two ways:
     - **Exact Match** when a submitted question is the same with the top selected question (the question corresponding to the highest proximity chunk).  
     - **Match with the top 3 selections** when a  submitted question is within the the top 3 questions (questions corresponding to the top 3 chunks according to their proximity with the submitted question).

### Results
The accuracy evaluations are saved in the in the *QA_Matching_Accuracy_Results.csv* file in the *results/* directory and are summarized in the following tables:

![Accuracy - Exact Match](https://github.com/antdragiotis/Embedding-models-vs-Languages-performance/blob/main/assets/Results_Accuracy.PNG)
Accuracy - Exact Match

![Accuracy - Top 3 selections](https://github.com/antdragiotis/Embedding-models-vs-Languages-performance/blob/main/assets/Results_Accuracy_alt.PNG)
Accuracy - Top 3 selections

The above evaluations confirm suggestions of current research that GenAI models show higher performance when processing English or languages they are direct descendants of Latin (like Italian, French and Romanian). **text-embedding-3-large** and **text-embedding-ada-002** demonstrate higher performance when comparing with **text-embedding-3-small**.    


### How to run the app:
- clone the repository: https://github.com/antdragiotis/Embedding-models-vs-Languages-performance
- change current directory to cloned repository
- pip install -r requirements.txt
- it is assumed that your system has been configured with the OPENAI_API_KEY and LANGCHAIN_API_KEY, otherwise you need to add the following statements to the python code files:
  - import os
  - os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
  - os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here"     
- run the two Python files as the processes described above: 
  - EVAL_EMBEDDINS_EU_REG_INGEST.py
  - EVAL_EMBEDDINGS_EU_REG_BATCH.py
 
You get pre-generated sample results the results in the *results/QA_Matching_Accuracy_Results.csv* file

### Project Structure
- *.py: main application code
- source: directory with the source data file
- intermediate_data: directory with intermediate data file that facilitate review of the ingestion process
- results: directory with a file containing the results of the process
- README.md: project documentation
