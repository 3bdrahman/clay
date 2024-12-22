import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
import os, getpass
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
import json 
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph
from IPython.display import Image, display
from PIL import Image
import io
import hashlib
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import re 
import sentence_transformers
import time 
import pandas as pd
import matplotlib.pyplot as plt
from dataset_manager import DatasetManager
import numpy as np
from io import StringIO
import sys
@st.cache_resource
def initialize_llm():
    st.write("Initializing LLM ...")
    local_llm = "llama3.2-vision"
    llm = ChatOllama(model = local_llm, temperature =0)
    llm_json = ChatOllama(model = local_llm, temperature =0, format = 'json')
    return llm, llm_json


@st.cache_resource
def initialize_vectorstore():
    st.write("Initialize VectorDB...")
    # load existing file hashes 
    file_hashes={}
    if os.path.exists('file_hashes.pkl'):
        with open('file_hashes.pkl', 'rb') as f:
            file_hashes = pickle.load(f)

    #load PDF docs
    loader = DirectoryLoader(
        "./Company",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    print(f"Loading documents from: {loader.path}")
    docs = loader.load()
    enhanced_docs = []
    for doc in docs:
        # Generate a unique parent document ID
        parent_doc_id = hashlib.md5(f"{doc.metadata['source']}_{doc.metadata.get('page', 0)}".encode()).hexdigest()
        
        # Text splitter with sentence-based splitting
        text_splitters = SpacyTextSplitter(chunk_size=1700, chunk_overlap=130, pipeline='en_core_web_sm')
        doc_splits = text_splitters.split_documents([doc])
        
        for split in doc_splits:
            # Attach parent document metadata to each chunk
            split.metadata.update({
                'parent_document_id': parent_doc_id,
                'source_file': doc.metadata['source'],
                'page_number': doc.metadata.get('page', 'N/A')
            })
            enhanced_docs.append(split)

    # add to Vector DB
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    if os.path.exists('faiss_index'):
        vectorstore = FAISS.load_local('faiss_index', embeddings,  allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(enhanced_docs, embeddings)
        # Optionally save the vector store for future use
        vectorstore.save_local('faiss_index')
    return vectorstore

llm, llm_json= initialize_llm()
vectorstore=initialize_vectorstore()
# set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Search using Tavily
web_search_tool = TavilySearchResults(k=2)


# Router 
router_instructions = """ You are highly capable at routing a user question to either a vectorestore, Python, or web search.

The vectorstore contains documents related to information about Aurora Consulting Inc. 
The documents introduce the company, clients and employees.

- Use the vectorestore for questions about:
    - Aurora Consulting as a company
    - Fact-based information about clients, employees.
    - General information about the services, policies, revenue. 

- Use the Python API for questions about:
    - Math 
    - Analysis of finances
    - Projects and database questions 


- Use web search only for FACTS like stock prices. 

Return JSON with a single key, "datasource", that is "vectorstore", "python", or "websearch" depending on where the information can be found."""

# document grader
doc_grader_instructions = """
You are a grader assessing the relavence and usefulness of a retrieved document to answer a user questions. 

if the document contains keywords(s) or semantic meaning related to the question, grade it as relavent.
"""

# Grader prompt 
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

Think carefully and objectively assess whether the document contains information that is relevant to answering the question. It's sufficient if the document has keywords or terms that are present in the question.

Return JSON with single key, binary_score, that is 'yes' if you think the document could be used to answer the questions or 'no' if not."""

## Augmented Generation 
rag_prompt = """You are an internal assistant at a consulting agency. You help employees by providing accurate information based on multiple data sources.

Here is the relevant context from different sources:

{formatted_context}

Question: {question}

Instructions:
1. Base your answer ONLY on the provided context
2. Answer directly and concisely
3. When interpreting data analysis results:
    - Explain the key findings clearly
    - Reference specific numbers and trends
    - Connect the analysis to the question
4. If you see Python code and results:
    - Explain what the code did
    - Interpret the numerical results
    - Highlight important patterns or findings
5. If you can't answer fully from the context, say so
6. Name 'references' using filname (source) when they are provided in the relavent context.

Answer:"""

# Hallucination grader to check whether the answer is based on the doc
hallucination_grader_instructions = """
You are a teacher grading a quiz to check if the student’s answer is based solely on the provided FACTS.

Here is the grading criteria:
(1) Ensure that the STUDENT ANSWER is strictly grounded in the information given in the FACTS.
(2) Ensure that the STUDENT ANSWER does not contain information that is outside the scope of the FACTS (i.e., hallucinated information).

Score:
- A score of "yes" means that the STUDENT ANSWER fully matches and is based on the FACTS.
- A score of "no" means that the STUDENT ANSWER includes information that is not present in the FACTS or omits critical information from the FACTS.

Provide a clear, step-by-step explanation of your reasoning to justify the score.
Avoid adding any details or assumptions not present in the FACTS.
"""

# Grader prompt 
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.

Evaluate whether the STUDENT ANSWER is fully based on the FACTS. A "yes" score indicates the answer is accurate and grounded in the provided FACTS; a "no" score indicates it includes details not found in the FACTS.

Return JSON with two keys:
- "binary_score": 'yes' or 'no' to indicate grounding in the FACTS.
- "explanation": a step-by-step explanation of the score, focusing only on matching the FACTS without introducing unrelated details.
"""

# Answer grader instructions 
answer_grader_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER.
Here is the grade criteria to follow: 
(1) The STUDENT ANSWER helps to answer the QUESTION
score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.
The student of no means that the studetn's answer doesn't meet all or the criteria. This is the lowest possible score.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""

# Analysis code generation instructions
code_generation_prompt = """You must analyze data about employee performance using Python.

Question: {question}

Available datasets (loaded and ready to use): {dataset_names}
Dataset columns: {available_datasets}

Return a JSON object with exactly this structure:
{{
    "code": "<your full python code here>",
    "explanation": "<your explanation here>"
}}

Your code MUST:
1. Start with imports
2. Use the existing dataframes (they're already loaded)
3. Store the final answer in a variable named 'result'
4. Strive to make the result look clean. For example, round large floats and improve the plot appearance. 

Example valid response:
{{
    "code": "import pandas as pd\\n\\n# Calculate average scores\\nresult = df['score'].mean()",
    "explanation": "Calculated the mean score from the dataset"
}}"""

error_correction_prompt = """Your previous code generation had an error: {error}

Please fix the code and ensure it:
1. Properly handles the error case
2. Returns complete, working code
3. Stores final result in 'result' variable

Previous code that failed:
{failed_code}

Return fixed code in this JSON structure:
{{
    "code": "<your corrected python code here>",
    "explanation": "<explanation of fixes made>"
}}"""
# Grader prompt 
answer_grader_prompt = """QUESTION: \n\n {question} \n\n SUDENT ANSWER: {generation}.

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""
#=======================================================================================================================================================
# Util
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
## Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "progress_placeholder" not in st.session_state:
    st.session_state.progress_placeholder = st.empty()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def update_progress(message):
    """Helper function to update progress messages with animation"""
    progress_container = st.session_state.progress_placeholder
    show_thinking_animation(progress_container, message)

def clear_progress():
    st.session_state.progress_placeholder.empty()

def format_chat_history(history, max_turns=3):
    """Format recent chat history for context"""
    recent_history = history[-max_turns*2:]  # Get last n turns
    formatted = []
    for msg in recent_history:
        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
        formatted.append(f"{role_prefix}{msg['content']}")
    return "\n".join(formatted)

def format_recent_context(history, max_turns=2):
    """Extract relevant context from recent interactions"""
    recent = history[-max_turns*2:]
    return " ".join(msg["content"] for msg in recent)

class EnhancedDocument:
    def __init__(self, content, parent_document_id, source_file, page_number=None):
        self.content = content
        self.parent_document_id = parent_document_id
        self.source_file = source_file
        self.page_number = page_number
# ===================================================================================================================================================================================
## Graph Nodes
def retrieve(state):
    """
    Retrieve Doc from VectorDB
    args: state(dict): current graph state
    return: state(dict): New Key added to state, documents, that contains retrieved documents
    """
    update_progress("Accessing Vector DB")
    question = state["question"]
    # write the retrieved docs to "documents" key in state
    documents = retriever.invoke(question)
    parent_docs = {}
    for doc in documents:
        # Use metadata to get parent document information
        parent_id = doc.metadata.get('parent_document_id', 'unknown')
        source_file = doc.metadata.get('source_file', 'Unknown Source')
        page_number = doc.metadata.get('page_number', 'N/A')

        if parent_id not in parent_docs:
            parent_docs[parent_id] = {
                'chunks': [],
                'source_file': source_file,
                'page_number': page_number
            }
        parent_docs[parent_id]['chunks'].append(doc.page_content)
    state.update({
        "documents": documents,
        "parent_documents": parent_docs
    })
    return state

def generate(state):
    """
    Generate answer using RAG on retrieved docs
    args: state(dict): current graph state
    returns: state(dict): New key added to state, generation , that contains model generation"""
    update_progress("Generation")
    question = state["question"]
    documents = state.get("documents", [])
    loop_step = state.get("loop_step", 0)
    chat_history = state["chat_history"]
    # Format context by source type
    formatted_sections = []
    has_context = False

    if documents:
        doc_texts = [doc.page_content for doc in documents]
        if doc_texts:
            has_context = True
            formatted_sections.append("COMPANY DOCUMENTATION:\n" + "\n---\n".join(doc_texts))

    if state.get("python_analysis"):
        analysis = state["python_analysis"]
        if analysis.get("type") == "python_analysis":
            has_context = True
            analysis_text = [
                "DATA ANALYSIS RESULTS:",
                f"Question: {state.get('question', 'N/A')}",
                f"Code:\n```python\n{analysis.get('code', 'No code')}\n```",
                f"Results:\n{analysis.get('results', 'No results')}",
                f"Explanation:\n{analysis.get('explanation', 'No explanation')}"
            ]
            
            formatted_sections.append("\n\n".join(analysis_text))

    if "web_search_results" in state:
        web_results = state["web_search_results"]
        if web_results:
            has_context = True
            web_texts = []
            for result in web_results:
                if isinstance(result, dict):
                    content = result.get("content", "")
                    title = result.get("title", "Web Result")
                    web_texts.append(f"{title}:\n{content}")
            if web_texts:
                formatted_sections.append("WEB SEARCH RESULTS:\n" + "\n---\n".join(web_texts))
    if not has_context:
        state.update({
            "generation": json.dumps("I'm unable to process your request as I don't have access to the necessary data or context. Please ensure the required data sources are available."),
            "loop_step": loop_step + 1,
            "chat_history": chat_history,
            "formatted_context": []
        })
        return state
    
    formatted_context = "\n\n============\n\n".join(formatted_sections)
    parent_references = []
    for parent_id, parent_info in state.get("parent_documents", {}).items():
        ref = f"Source: {parent_info['source_file']}"
        if parent_info.get('page_number'):
            ref += f" (Page {parent_info['page_number']})"
        ref += f"\nRelevant Chunks: {' | '.join(parent_info['chunks'])}"
        parent_references.append(ref)
    
    formatted_context += "\n\n==== DOCUMENT REFERENCES ====\n" + "\n\n".join(parent_references)
    rag_prompt_formatted = rag_prompt.format(
        formatted_context=formatted_context,
        question=question
    )
    generation = llm.invoke([
        SystemMessage(content="""You are a helpful assistant that answers questions based solely on the provided context.
        If you see Python analysis results, make sure to interpret them clearly and explain their significance.
Always reference specific data points and findings from the context."""),
        HumanMessage(content=rag_prompt_formatted)
    ])
    new_exchange = {"role":"assistant", "content": generation.content}
    updated_history = chat_history + [
        {"role": "user", "content": question},
        new_exchange
    ]
    
    state.update({
        "generation": generation,
        "loop_step": loop_step + 1, 
        "chat_history": updated_history,
        "formatted_context": formatted_sections
    }) 
    print("Python analysis @ end of generate:", state.get("python_analysis"))
    return state

def python_data_analysis(state):
    """Generate and execute python code"""
    update_progress("Analysis with Python")
    question = state["question"]
    dm = DatasetManager()
    max_retries = 2
    retry_count = 0
    last_error = None
    previous_analysis = state.get("python_analysis", {})
    while retry_count <= max_retries:
        try:
            # Format the appropriate prompt based on retry status
            if last_error:
                formatted_prompt = error_correction_prompt.format(
                    error=last_error,
                    failed_code=state.get("python_analysis", {}).get("code", "No previous code")
                )
                update_progress(f"Attempting code correction (Attempt {retry_count + 1})")
            else:
                formatted_prompt = code_generation_prompt.format(
                    question=question,
                    dataset_names=", ".join(dm.datasets.keys()),
                    available_datasets=json.dumps({name: list(df.columns) for name, df in dm.datasets.items()})
                )
                update_progress("Generating analysis code")

            # Generate code with structured message
            messages = [
                SystemMessage(content="You are a Python code generator. Always return complete code in a JSON structure with 'code' and 'explanation' keys."),
                HumanMessage(content=formatted_prompt)
            ]
            
            code_generation = llm_json.invoke(messages)
            generated_response = json.loads(code_generation.content)
            code = generated_response.get('code', '')
            
            if not code or code == "import pandas as pd":
                raise ValueError("Incomplete code generation")
                
            with st.expander(f"View Analysis Code (Attempt {retry_count + 1})", expanded=False):
                st.code(code, language='python')
            
            # Set up execution environment
            execution_namespace = {
                **dm.datasets,
                'pd': pd,
                'plt': plt,
                'np': np,
                'result': None
            }
            
            # Execute the code
            exec(code, execution_namespace)
            analysis_results = execution_namespace.get('result')
            
            if analysis_results is None:
                raise ValueError("Code executed but no results were generated")
            
            if isinstance(analysis_results, pd.DataFrame):
                results_str = analysis_results.to_string()
            elif isinstance(analysis_results, (plt.Figure, plt.Axes)):
                # Save plot to string buffer
                buf = io.StringIO()
                analysis_results.figure.savefig(buf, format='svg')
                results_str = buf.getvalue()
            else:
                results_str = str(analysis_results)

            # Display results
            with st.expander(f"View Analytics Results (Attempt {retry_count + 1})", expanded=False):
                st.subheader("Analysis Results:")
                if isinstance(analysis_results, pd.DataFrame):
                    st.dataframe(analysis_results)
                elif isinstance(analysis_results, (plt.Figure, plt.Axes)):
                    st.pyplot(analysis_results)
                else:
                    st.write(analysis_results)
            
            # Success! Update state and break the retry loop
            
            updated_state= {
                **state,
                "python_analysis": {
                    "type": "python_analysis",
                    "code": code,
                    "explanation": generated_response.get('explanation', ''),
                    "results": results_str,
                    "attempts": retry_count + 1
                }
            }
            print("Python analysis @ end of python_analysis:", updated_state.get("python_analysis"))
            return updated_state
                
            
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            
            if retry_count > max_retries:
                # If we have a previous successful analysis, keep it
                if previous_analysis.get("type") == "python_analysis":
                    error_context = f"Failed after {max_retries} attempts. Last error: {last_error}"
                    previous_analysis.update({
                        "error_context": error_context,
                        "last_error": last_error
                    })
                    return {
                        **state,
                        "python_analysis": previous_analysis
                    }
                
                # If no previous analysis, set error state
                error_analysis = {
                    "type": "python_analysis_error",
                    "error": last_error,
                    "attempts": retry_count,
                    "context": f"Failed after {max_retries} attempts. Last error: {last_error}"
                }
                return {
                    **state,
                    "python_analysis": {
                        "type": "python_analysis_error",
                        "error": last_error,
                        "attempts": retry_count,
                        "context": f"Failed after {max_retries} attempts. Last error: {last_error}"
                    }
                }
            else:
                st.warning(f"Attempt {retry_count} failed: {last_error}. Trying again...")
    
    return state

def grade_documents(state):
    """
    Determines whether the retrieved documents are relavent to the question
    if any document is not relavent, we will set a flag to run web search
    args: state(dict): current graph state
    returns: state(dict): filtered out irrelavent documents and updated web_search state
    """
    update_progress("Grading Documents")
    question = state["question"]
    documents = state["documents"]
    # score each document
    filtered_docs = []
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question = question)
        result = llm_json.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        # document relavent
        if grade.lower() == "yes":
            filtered_docs.append(d)
        # document not relevant
    web_search = "Yes" if not filtered_docs else "No"
    print("Python analysis @ end of grade_documents:", state.get("python_analysis"))
    return {
        **state,
        "documents": filtered_docs,
        "web_search": web_search
    }

def web_search(state):
    """
    Web search based on the question
    args: state(dict) : current graph state
    returns: state(dict) : appended web results to documents"""
    update_progress("Accessing Web Search")
    question = state["question"]
    documents = state.get("documents", [])
    # Web search 
    docs = web_search_tool.invoke({"query":question})
    web_results = [
        {
            "type": "web_search",
            "content": d["content"],
            "title": d.get("title", "Untitled Web Result")
        } for d in docs
    ]
    state.update({
        "web_search_results": web_results,
    })
    return state

## Edges 
def route_question(state):
    """
    Route question to web search or RAG
    args: state(dict) : currect graph state
    returns: state(dict) : next node to call
    """
    update_progress("Routing Question")
    route_question = llm_json.invoke(
        [SystemMessage(content=router_instructions)] 
        + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)['datasource']
    return source
    

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search
    args: state(dict): current graph state
    returns: state(dict): Binary decision for next call
    """
    web_search = state["web_search"]
    return "websearch" if web_search == "Yes" else "generate"

def grade_generation_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers questions
    args: state(dict): current graph state
    returns: str: decision for next node to call"""
    update_progress("Evaluating Relevance")
    question = state["question"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    context_sources = []
    print("Python analysis @ beginning of grade_generation_documents_and_question:", state.get("python_analysis"))
    if state.get("documents"):
        context_sources.append("Vector Store Documents:\n" + format_docs(state["documents"]))
    
    if state.get("python_analysis"):
        print("Adding Python analysis to context sources")
        analysis_context = state["python_analysis"]
        context_sources.append("Python Analysis:\n" + analysis_context.get("results", ""))

    if state.get("web_search_results"):
        web_results = [f"{r['title']}:\n{r['content']}" for r in state["web_search_results"]]
        context_sources.append("Web Search Results:\n" + "\n---\n".join(web_results))

    full_context = "\n\n---\n\n".join(context_sources)
    
    # If we have no context at all
    if not full_context.strip():
        if state["loop_step"] <= max_retries:
            return "not supported"
        return "max_retries"
    
    # Grade hallucination
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=full_context,
        generation=generation.content
    )
    result = llm_json.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    hallucination_grade = json.loads(result.content)["binary_score"].lower()

    # If not hallucinating, check if it answers the question
    if hallucination_grade == "yes":
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question,
            generation=generation.content
        )
        result = llm_json.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        answer_grade = json.loads(result.content)["binary_score"].lower()
        
        if answer_grade == "yes":
            return "useful"
        elif state["loop_step"] <= max_retries:
            return "not useful"
        else:
            return "max_retries"
    
    # If hallucinating and haven't hit max retries, try again
    elif state["loop_step"] <= max_retries:
        return "not supported"
    else:
        return "max_retries"

def show_thinking_animation(container, message):
    for _ in range(3):
        for i in range(3):
            container.markdown(message + ' ' + '.' * (i + 1))
            time.sleep(0.3)
#========================================================================================================================================================================
## Chat Application
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "progress_placeholder" not in st.session_state:
    st.session_state.progress_placeholder = st.empty()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class ChatState(TypedDict):
    question: str
    documents: List[str]
    generation: str | None
    web_search: str | None
    max_retries: int
    loop_step: Annotated[int, operator.add]
    chat_history: List[dict]
    current_context: str
def run_workflow(state):
    while True:
        step = state.get("current_step", "route_question")
        if step == "route_question":
            source = route_question(state)
            if source == "websearch":
                state["current_step"] = "websearch"
            elif source == "vectorstore":
                state["current_step"] = "retrieve"
            elif source == "python":
                state["current_step"] = "python_analysis"
            else:
                state["current_step"] = "end"
        
        elif step == "websearch":
            state = web_search(state)
            state["current_step"] = "generate"

        elif step == "retrieve":
            state = retrieve(state)
            state["current_step"] = "grade_documents"

        elif step == "grade_documents":
            state = grade_documents(state)
            next_step = decide_to_generate(state)
            state["current_step"] = "websearch" if next_step == "websearch" else "generate"

        elif step == "generate":
            state = generate(state)
            decision = grade_generation_documents_and_question(state)
            if decision == "useful":
                state["current_step"] = "end"
            elif decision == "not useful":
                state["current_step"] = "websearch"
            elif decision == "not supported":
                state["current_step"] = "generate"  # Retry generation
            elif decision == "max_retries":
                state["current_step"] = "end"

        elif step == "python_analysis":
            state = python_data_analysis(state)
            state["current_step"] = "generate"

        elif step == "end":
            break

        else:
            raise ValueError(f"Unknown step: {step}")
    
    return state


def process_user_message(user_input, chat_history):
    
    # Reset progress message
    state = {
    "question": user_input,
    "documents": [],
    "generation": None,
    "web_search": None,
    "python_analysis": None,
    "max_retries": 3,
    "loop_step": 0,
    "chat_history": chat_history,
    "current_step": "route_question",  # Entry point
}
    
    
    # print(dir(graph))
    # print(f"State before: {state}")
    final_state = run_workflow(state)
    generation = final_state.get("generation")
    if isinstance(generation, str):
        # Handle JSON-encoded string case
        try:
            response = json.loads(generation)
        except (json.JSONDecodeError, TypeError):
            response = generation
    else:
        # Handle Message object case
        response = generation.content
    # print(f"State after: {final_state}")
    return response, final_state.get("chat_history")

def initialize_interface():
    st.title("RAG Agent")
    chat_container = st.container()
    # Display history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    user_input = st.chat_input("Type your message here...")
    return user_input, chat_container

def update_chat_history(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def main():
    user_input, chat_container = initialize_interface()

    if user_input:
        # add the message to chat
        update_chat_history("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # process the user message with graph workflow
        with st.chat_message("assistant"):
            st.session_state.progress_placeholder = st.empty()
            assistant_response, updated_chat_history = process_user_message(user_input, st.session_state.messages)
            st.session_state.chat_history=update_chat_history
            st.session_state.progress_placeholder.empty()
            st.markdown(assistant_response)
            update_chat_history("assistant", assistant_response)



if __name__ == "__main__":
    main()

# Add custom CSS for better chat UI
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    max-width: 80%;
}
.stChatMessage[data-role="user"] {
    background-color: #e6f3ff;
    margin-left: auto;
}
.stChatMessage[data-role="assistant"] {
    background-color: #f0f2f6;
    margin-right: auto;
}
.stExpander {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    margin: 10px 0;
    background-color: white;
}

.stExpander > div:first-child {
    background-color: #f8f9fa;
    padding: 10px;
    font-weight: 500;
}

/* Code block styling */
.stCode {
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}

/* Results styling */
.element-container {
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)