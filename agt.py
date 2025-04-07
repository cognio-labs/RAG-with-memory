import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END
# Import SqliteSaver with error handling
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    # Fallback for backwards compatibility or if the package is not properly installed
    SqliteSaver = None
    import logging
    logging.warning(
        "SqliteSaver could not be imported. Will use in-memory checkpointer.")
from typing import List, Optional, Dict, Any, Callable, TypedDict, Union, cast
from langgraph.graph import MessagesState
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from exa_py import Exa
import logging
from typing import TypeVar, Literal, cast
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
import asyncio
import replicate
import datetime
import requests
# Import Tavily search tools for improved web search
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_available = True
except ImportError:
    tavily_available = False
    logging.warning("TavilySearchResults could not be imported. Web search will use Exa only.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('CLAUDE_API_KEY')
groq_key = os.getenv('GROQ_API_KEY')
exa_key = os.getenv('EXA_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
replicate_api_token = os.getenv('REPLICATE_API_TOKEN')
musicfy_api_key = os.getenv('MUSICFY_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# Debug log for API keys
logger.info(f"MUSICFY_API_KEY loaded: {'Yes' if musicfy_api_key else 'No'}")
logger.info(f"TAVILY_API_KEY loaded: {'Yes' if tavily_api_key else 'No'}")

# Force set the MUSICFY_API_KEY if it's not loaded from .env
if not musicfy_api_key:
    musicfy_api_key = "cm8009x9i0029lb0cp5dqjczu"
    logger.info("MUSICFY_API_KEY manually set from hardcoded value")

required_keys = {
    'OPENAI_API_KEY': openai_key,
    'GROQ_API_KEY': groq_key,
    'TAVILY_API_KEY': tavily_api_key,
    'QDRANT_URL': qdrant_url,
    'QDRANT_API_KEY': qdrant_api_key,
    'MUSICFY_API_KEY': musicfy_api_key
}

# Soft requirements - only log warning if missing
if not replicate_api_token:
    logger.warning("REPLICATE_API_TOKEN is not set. Image generation will not work.")
    
if not tavily_api_key:
    logger.warning("TAVILY_API_KEY is not set. Advanced web search will fall back to Exa only.")

missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    error_msg = f"Missing required environment variables: {', '.join(missing_keys)}"
    logger.error(error_msg)
    raise ValueError(error_msg)


# Define state with type annotations
class VaaniState(MessagesState):
    """State for the Vaani chatbot."""
    messages: List[BaseMessage]
    summary: Optional[str] = None
    file_url: Optional[str] = None
    indexed: bool = False
    deep_research_requested: bool = False
    agent_name: Optional[str] = None
    model_name: str = "gpt4o"  # Default model
    collection_name: Optional[str] = None
    reflect_iterations: int = 0
    reflection_data: Optional[Dict[str, Any]] = None
    max_search_results: int = 5  # Default number of search results for web search


# Return types for conditional routing
AgentNames = Literal["deep_research", "orchestrator", "rag_agent",
                     "web_search_agent", "image_generator", "music_generator", "default"]
T = TypeVar('T')


# Reflection Pydantic models for the enhanced web search
class Reflection(BaseModel):
    missing: str = Field(
        description="Critique of what is missing from the current answer.")
    superfluous: str = Field(
        description=
        "Critique of what is superfluous or incorrect in the current answer.")


class InitialResearch(BaseModel):
    """Research and answer the question based on web search results."""
    answer: str = Field(
        description=
        "~250 word detailed answer to the question that incorporates the search results."
    )
    reflection: Reflection = Field(
        description="Critical self-reflection on the current answer.")
    search_queries: List[str] = Field(
        description=
        "1-3 search queries for researching improvements to address the critique of your current answer."
    )


class RevisedResearch(InitialResearch):
    """Revise your original answer based on additional search results."""
    references: List[str] = Field(
        description="Citations supporting your updated answer.")


# Helper functions
def build_conversation_context(state: VaaniState) -> str:
    """Builds the conversation context including summary and recent messages."""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Start with the summary if available
    context = f"Conversation summary: {summary}\n\n" if summary else ""
    
    # Add recent messages
    if messages:
        context += "Recent messages:\n"
        for msg in messages:
            prefix = "User: " if isinstance(msg, HumanMessage) else "Assistant: "
            context += f"{prefix}{msg.content}\n"
    
    return context.strip()


def get_model(model_name: str):
    """Returns the selected language model based on user choice."""
    try:
        logger.info(f"Getting language model for: {model_name}")
        if model_name == "gpt4o":
            logger.info("Using GPT-4o model with OpenAI")
            if not openai_key:
                logger.error("OpenAI API key is missing!")
                raise ValueError("OpenAI API key is required but not provided")
            return ChatOpenAI(model="gpt-4o",
                              api_key=openai_key,
                              temperature=0.3)
        elif model_name == "claude":
            if not anthropic_key:
                logger.warning(
                    "Claude API key not found, falling back to gpt4o")
                if not openai_key:
                    logger.error(
                        "OpenAI API key is also missing for fallback!")
                    raise ValueError(
                        "API keys are missing for both Claude and OpenAI")
                return ChatOpenAI(model="gpt-4o",
                                  api_key=openai_key,
                                  temperature=0.3)
            logger.info("Using Claude model with Anthropic")
            return ChatAnthropic(model_name="claude-3-5-sonnet-20240620",
                                 api_key=anthropic_key,
                                 temperature=0.3)
        elif model_name == "llama":
            logger.info("Using Llama model with Groq")
            if not groq_key:
                logger.error("Groq API key is missing!")
                raise ValueError("Groq API key is required but not provided")
            return ChatGroq(model="llama-3.1-8b-chat",
                            api_key=groq_key,
                            temperature=0.3)
        else:
            logger.warning(
                f"Unknown model: {model_name}, falling back to gpt4o")
            if not openai_key:
                logger.error("OpenAI API key is missing for fallback!")
                raise ValueError(
                    "OpenAI API key is required for fallback but not provided")
            return ChatOpenAI(model="gpt-4o",
                              api_key=openai_key,
                              temperature=0.3)
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}",
                     exc_info=True)
        raise


def is_image_file(file_url: str) -> bool:
    """Checks if the file is an image."""
    if not file_url:
        return False
    return file_url.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))


def is_document_file(file_url: str) -> bool:
    """Checks if the file is a document."""
    if not file_url:
        return False
    return file_url.lower().endswith(('.pdf', '.docx', '.txt'))


# Reflection utility class for web search
class ResponderWithRetries:
    """Handles response generation with validation retries."""

    def __init__(self, runnable, validator, max_retries=3):
        self.runnable = runnable
        self.validator = validator
        self.max_retries = max_retries

    def respond(self, state):
        original_messages = state.get("messages", [])
        working_messages = original_messages.copy()
        
        for attempt in range(self.max_retries):
            try:
                # Try to generate a valid response
                response = self.runnable.invoke({"messages": working_messages})
                # Validate it
                self.validator.invoke(response)
                # If validation successful, return response
                return response
            except ValidationError as e:
                logger.warning(
                    f"Validation error on attempt {attempt+1}: {str(e)}")
                
                # Don't modify the working messages on the last attempt
                if attempt < self.max_retries - 1:
                    # Start fresh with original messages each time
                    working_messages = original_messages.copy()
                    # Add guidance as a system message
                    working_messages.append(
                        SystemMessage(
                            content=
                            f"The previous response was invalid: {str(e)}. "
                            f"You **MUST** follow the schema exactly. "
                            f"**Ensure you include the 'search_queries' field as a list of strings.** "
                            f"Example: \"search_queries\": [\"query 1\", \"query 2\"]"
                        ))
        
        # If we exhausted all retries, try one more time with a very explicit prompt
        try:
            final_messages = original_messages.copy()
            final_messages.append(
                SystemMessage(
                    content=
                    "The previous attempts failed validation. Provide a valid response adhering STRICTLY to this structure:\n"
                    "{\n"
                    "  \"answer\": \"[detailed answer]\",\n"
                    "  \"reflection\": {\n"
                    "    \"missing\": \"[critique missing]\",\n"
                    "    \"superfluous\": \"[critique superfluous]\"\n"
                    "  },\n"
                    "  \"search_queries\": [\"query 1\", \"query 2\", \"query 3\"]  <-- **THIS FIELD IS MANDATORY**\n"
                    "}\n"
                    "The 'search_queries' field MUST be a list of strings and cannot be empty or omitted."
                ))
            response = self.runnable.invoke({"messages": final_messages})
            # Attempt validation one last time, but return even if invalid
            try:
                self.validator.invoke(response)
            except ValidationError as final_val_error:
                 logger.error(f"Final validation attempt failed: {final_val_error}")
            return response # Return the response even if final validation fails
        except Exception as final_error:
            logger.error(f"Final attempt failed: {final_error}")
            # Return a fallback message
            return AIMessage(
                content="I couldn't generate a properly formatted response. Let me try to answer directly: I'll need to search for more information on this topic.")


# Node implementations
def entry_node(state: VaaniState) -> VaaniState:
    """Entry node that initializes routing but returns state."""
    logger.info(
        f"Entry node processing: deep_research={state['deep_research_requested']}"
    )
    return state


def summarizer_node(state: VaaniState) -> VaaniState:
    """Summarizes the conversation if it exceeds 6 messages."""
    try:
        messages = state["messages"]
        if len(messages) <= 6:
            logger.info("Skipping summarization as message count is <= 6")
            return state
        summarizer = ChatGroq(model="llama-3.3-70b-versatile",
                              api_key=groq_key,
                              temperature=0.3)
        conversation = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in messages])
        prompt = ChatPromptTemplate.from_template("""
        Provide a concise summary of the following conversation:
        {conversation}
        Summary:
        """)
        response = summarizer.invoke(prompt.format(conversation=conversation))
        state["summary"] = response.content
        
        # Keep only the most recent message (the user's latest query)
        latest_message = messages[-1]
        state["messages"] = [latest_message]
        
        logger.info("Conversation summarized successfully and messages list cleared except for the latest message")
        return state
    except Exception as e:
        logger.error(f"Error in summarizer_node: {e}")
        return state


orchestrator_prompt = """
You are an orchestrator agent. Based on the conversation history, current query, and state, decide which agent to route to.
Possible agents:
- rag_agent: if a document is attached (file_url is set and it's a document) or the query is clearly about an already indexed document
- web_search_agent: if the query might benefit from web search (e.g. current events, facts, etc.)
- image_generator: if the query explicitly asks to generate an image (e.g., "create an image", "draw me a picture")
- music_generator: if the query explicitly asks to generate or create music, a song, melody, or sound
- default: for general questions, chat, opinions, or queries best answered directly from the model's knowledge
State:
- file_url: {file_url}
- indexed: {indexed}
Conversation history:
{conversation_context}
Current query: {current_query}
Output only one of the exact agent names: "rag_agent", "web_search_agent", "image_generator", "music_generator", or "default". 
No explanation, just the agent name.
"""


def orchestrator_node(state: VaaniState) -> VaaniState:
    """Routes the query to the appropriate agent based on state and query."""
    try:
        # First, explicitly check for image generation requests
        current_query = state["messages"][-1].content if state[
            "messages"] else ""

        # Check for explicit image generation terms
        image_keywords = [
            "create an image", "generate an image", "make an image", "draw",
            "picture of", "image of"
        ]
        if any(keyword in current_query.lower() for keyword in image_keywords):
            logger.info(
                "Image generation keywords detected, routing to image_generator"
            )
            state["agent_name"] = "image_generator"
            return state
            
        # Check for music generation requests
        music_keywords = [
            "create music", "generate music", "make music", "compose music",
            "create a song", "generate a song", "make a song", "compose a song",
            "create a melody", "generate a melody", "create a tune", "create a beat",
            "make a soundtrack", "generate a soundtrack", "create some music",
            "generate audio", "generate some music", "create audio"
        ]
        if any(keyword in current_query.lower() for keyword in music_keywords):
            logger.info(
                "Music generation keywords detected, routing to music_generator"
            )
            state["agent_name"] = "music_generator"
            return state

        # Check for queries that likely need web search
        web_search_indicators = [
            "latest", "recent", "news", "current", "today", "ceo of", "who is",
            "what is", "when did", "where is", "how to", "update on",
            "where can i find", "website", "stock price", "price of",
            "weather in", "events in"
        ]
        if any(indicator in current_query.lower()
               for indicator in web_search_indicators):
            logger.info(
                "Web search indicators detected, routing to web_search_agent")
            state["agent_name"] = "web_search_agent"
            return state

        # If no special case, proceed with LLM-based routing
        orchestrator = ChatGroq(model="llama-3.3-70b-versatile",
                                api_key=groq_key,
                                temperature=0.2)
        conversation_context = build_conversation_context(state)

        prompt = orchestrator_prompt.format(
            file_url=state["file_url"],
            indexed=state["indexed"],
            conversation_context=conversation_context,
            current_query=current_query)
        response = orchestrator.invoke(prompt)
        agent_name = response.content.strip().lower()
        valid_agents = [
            "rag_agent", "web_search_agent", "image_generator", "music_generator", "default"
        ]

        # Log the raw response for debugging
        logger.info(f"Orchestrator raw response: '{response.content}'")

        if agent_name not in valid_agents:
            logger.warning(
                f"Invalid agent name '{agent_name}', defaulting to 'default'")
            agent_name = "default"

        logger.info(f"Orchestrator selected agent: {agent_name}")
        state["agent_name"] = agent_name

        # Reset reflection counters when starting a new query
        state["reflect_iterations"] = 0
        state["reflection_data"] = None
        return state
    except Exception as e:
        logger.error(f"Error in orchestrator_node: {e}")
        state["agent_name"] = "default"
        return state


def indexor_node(state: VaaniState) -> VaaniState:
    """Indexes a document using Qdrant and updates the state."""
    try:
        file_url = state["file_url"]
        if not file_url or not is_document_file(file_url):
            logger.warning(f"Invalid file for indexing: {file_url}")
            return state
        logger.info(f"Indexing document: {file_url}")
        config = state.get("configurable", {})
        thread_id = config.get("thread_id", "default")
        collection_name = f"vaani_{hashlib.md5(thread_id.encode()).hexdigest()[:16]}"
        if file_url.endswith('.pdf'):
            loader = PyPDFLoader(file_url)
        elif file_url.endswith('.txt'):
            loader = TextLoader(file_url)
        elif file_url.endswith('.docx'):
            loader = Docx2txtLoader(file_url)
        else:
            raise ValueError(f"Unsupported file type: {file_url}")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {file_url}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        try:
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if collection_name in collection_names:
                logger.info(
                    f"Collection {collection_name} already exists, deleting it"
                )
                client.delete_collection(collection_name)
        except Exception as collection_err:
            logger.warning(f"Error checking collections: {collection_err}")
        vector_store = QdrantVectorStore(client=client,
                                         collection_name=collection_name,
                                         embeddings=embeddings)
        vector_store.add_documents(splits)
        state["indexed"] = True
        state["collection_name"] = collection_name
        logger.info(
            f"Document indexed successfully into collection {collection_name}")
        return state
    except Exception as e:
        logger.error(f"Error in indexor_node: {e}", exc_info=True)
        return state


def rag_agent_node(state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Handles document-based queries, indexing if necessary, using the selected model."""
    try:
        if not state["indexed"] and state["file_url"] and is_document_file(
                state["file_url"]):
            logger.info("Document needs indexing, calling indexor_node")
            state = indexor_node(state)
            if not state["indexed"]:
                return {
                    "messages": [
                        AIMessage(
                            content=
                            "I couldn't index your document. Please try uploading it again."
                        )
                    ]
                }
        
        # Get the current query and conversation context
        current_query = state["messages"][-1].content
        conversation_context = build_conversation_context(state)
        
        # Retrieve context from the indexed document
        context = ""
        if state["indexed"] and state["collection_name"]:
            try:
                logger.info(
                    f"Retrieving context from indexed document in collection {state['collection_name']}"
                )
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                embeddings = OpenAIEmbeddings(api_key=openai_key)
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=state["collection_name"],
                    embeddings=embeddings)
                retrieved_docs = vector_store.similarity_search(current_query,
                                                                k=3)
                context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved {len(retrieved_docs)} document chunks")
            except Exception as retrieval_error:
                logger.error(
                    f"Error retrieving from vector store: {retrieval_error}")
                context = "Error retrieving document context. Proceeding with best available information."
        
        # Create the prompt with conversation context and document context
        prompt = ChatPromptTemplate.from_template("""
        Conversation history:
        {conversation_context}
        
        Based on the conversation history and document context, answer the question.
        Document context:
        {context}
        
        Question: {question}
        Answer:
        """)
        
        # Get the appropriate model and generate a response
        llm = get_model(state["model_name"])
        response = llm.invoke(
            prompt.format(conversation_context=conversation_context,
                          context=context,
                          question=current_query))
        
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"Error in rag_agent_node: {e}", exc_info=True)
        error_response = "I encountered an error while processing your document-based query. Please try again with a simpler query."
        return {"messages": [AIMessage(content=error_response)]}


def web_search_agent_node(
        state: VaaniState) -> Dict[str, Union[List[BaseMessage], VaaniState]]:
    """Enhanced web search agent using Reflexion for iterative improvement."""
    try:
        file_url = state["file_url"]
        current_query = state["messages"][-1].content
        conversation_context = build_conversation_context(state)
        llm = get_model(state["model_name"])

        # Handle image-based queries first
        if file_url and is_image_file(file_url):
            logger.info(f"Processing image-based query with file: {file_url}")
            vision_model = ChatOpenAI(model="gpt-4o",
                                      api_key=openai_key,
                                      temperature=0.3)
            messages = [
                SystemMessage(
                    content=
                    f"Previous conversation context:\n{conversation_context}"),
                HumanMessage(content=[{
                    "type":
                    "text",
                    "text":
                    f"Based on this image and the following question: {current_query}"
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{file_url}"
                    }
                }])
            ]
            response = vision_model.invoke(messages)
            return {"messages": [AIMessage(content=response.content)]}

        # Check if this is a continuing reflection or new query
        if state.get("reflect_iterations", 0) == 0:
            # Initial search phase
            logger.info("Starting new web search reflection process")

            # Create initial search prompt
            initial_search_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 """You are an expert researcher and web search specialist.
Current time: {time}

Consider the conversation history to provide context for your answer.

Your task is to:
1. Provide a detailed ~250 word answer to the user's question.
2. Reflect and critique your answer critically.
3. **Crucially, you MUST recommend 1-3 specific search queries** to find information that addresses the critiques and improves your answer.

Respond using the InitialResearch function, ensuring **ALL fields**, especially `search_queries` (as a list of strings), are included."""
                 ),
                MessagesPlaceholder(variable_name="messages"),
                ("user",
                 "\n\n<s>Reflect on the user's original question, formulate an initial answer, and provide search queries. Respond using the InitialResearch function. Ensure the 'search_queries' field is populated.</s>"
                 )
            ]).partial(time=lambda: datetime.datetime.now().isoformat())

            # Set up LLM with tool binding
            initial_answer_chain = initial_search_prompt | llm.bind_tools(
                tools=[InitialResearch])
            validator = PydanticToolsParser(tools=[InitialResearch])

            # Create responder with validation
            first_responder = ResponderWithRetries(
                runnable=initial_answer_chain, validator=validator)

            # Generate initial response
            try:
                initial_response = first_responder.respond({
                    "messages": [
                        SystemMessage(
                            content=
                            f"Previous conversation context:\n{conversation_context}"
                        ),
                        HumanMessage(content=current_query)
                    ]
                })

                logger.info("Successfully generated initial response")

                # Verify the response has expected attributes
                if not hasattr(
                        initial_response,
                        'tool_calls') or not initial_response.tool_calls:
                    logger.error(
                        "Initial response is missing tool_calls attribute")
                    return {
                        "messages": [
                            AIMessage(
                                content=
                                "I couldn't understand your query properly. Could you please rephrase it?"
                            )
                        ]
                    }

                # Extract search queries and perform search
                try:
                    tool_args = initial_response.tool_calls[0]["args"]
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    search_queries = tool_args.get("search_queries", [])

                    if not search_queries:
                        logger.warning("No search queries generated")
                        fallback_answer = tool_args.get(
                            "answer",
                            "I don't have enough information to answer that question comprehensively."
                        )
                        return {
                            "messages": [AIMessage(content=fallback_answer)]
                        }

                    logger.info(
                        f"Generated {len(search_queries)} search queries: {search_queries}"
                    )

                    # Execute web search
                    exa_client = Exa(api_key=exa_key)
                    search_results = []

                    # Add debug logging for search process
                    logger.info(f"Performing web search with Exa API")

                    for query in search_queries:
                        try:
                            logger.info(f"Searching for: {query}")
                            results = exa_client.search(query,
                                                        use_autoprompt=True,
                                                        num_results=3)

                            if not results or not results.results:
                                logger.warning(
                                    f"No results found for query: {query}")
                                continue

                            logger.info(
                                f"Got {len(results.results)} results for query: {query}"
                            )

                            for r in results.results:
                                search_results.append({
                                    "query": query,
                                    "url": r.url,
                                    "title": r.title,
                                    "content": r.text[:800]
                                })
                        except Exception as query_error:
                            logger.error(
                                f"Error searching for query '{query}': {query_error}"
                            )
                            continue

                    # If we couldn't get any search results after trying all queries
                    if not search_results:
                        logger.warning(
                            "No search results found for any queries")
                        fallback_answer = tool_args.get(
                            "answer",
                            "I don't have enough external information to fully answer your question. Here's what I know based on my training:"
                        )
                        return {
                            "messages": [AIMessage(content=fallback_answer)]
                        }

                    # Store reflection data in state
                    state["reflection_data"] = {
                        "original_query": current_query,
                        "initial_response":
                        initial_response.tool_calls[0]["args"],
                        "search_results": search_results,
                        "final_answer": None
                    }

                    # Increment reflection counter
                    state["reflect_iterations"] = 1

                    # Format search results for next step
                    search_context = "\n\n".join([
                        f"Source: {r['url']}\nTitle: {r['title']}\nQuery: '{r['query']}'\nContent: {r['content']}"
                        for r in search_results
                    ])

                    # Prepare revision prompt
                    revision_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         """You are an expert researcher and web search specialist.
Current time: {time}

Consider the conversation history to provide context-appropriate responses.

1. Revise your previous answer using the new information from web search.
   - You MUST include numerical citations in your revised answer to ensure it can be verified.
   - Add a "References" section to the bottom of your answer (which does not count towards the word limit).
   - Format references as:
     [1] https://example.com
     [2] https://example.com
2. Reflect and critique your revised answer. Be severe to maximize improvement.
3. Recommend additional search queries if needed to improve your answer further."""
                         ),
                        ("user",
                         "Conversation context: {conversation_context}\n\nOriginal question: {question}"
                         ), ("assistant", "{initial_response}"),
                        ("user",
                         "Here are search results based on your queries:\n\n{search_results}\n\nPlease revise your answer based on this information using the RevisedResearch function."
                         )
                    ]).partial(
                        time=lambda: datetime.datetime.now().isoformat(),
                        conversation_context=conversation_context,
                        question=current_query,
                        initial_response=json.dumps(tool_args),
                        search_results=search_context)

                    # Set up revision chain
                    revision_chain = revision_prompt | llm.bind_tools(
                        tools=[RevisedResearch])
                    revision_validator = PydanticToolsParser(
                        tools=[RevisedResearch])
                    revisor = ResponderWithRetries(
                        runnable=revision_chain, validator=revision_validator)

                    # Generate revised response
                    revised_response = revisor.invoke({})

                    # Extract final answer and store
                    if hasattr(revised_response,
                               'tool_calls') and revised_response.tool_calls:
                        tool_args = revised_response.tool_calls[0]["args"]
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)
                        final_answer = tool_args.get("answer",
                                                     "") + "\n\nReferences:\n"
                        for i, ref in enumerate(tool_args.get(
                                "references", [])):
                            final_answer += f"[{i+1}] {ref}\n"

                        # Format the final answer for display
                        state["reflection_data"]["final_answer"] = final_answer

                        # For MAX_ITERATIONS control, we'll keep a counter
                        MAX_ITERATIONS = 2  # Limit to 2 rounds for performance

                        if state["reflect_iterations"] < MAX_ITERATIONS:
                            # If more iterations needed, extract new search queries
                            new_queries = tool_args.get("search_queries", [])
                            if new_queries:
                                # Could continue the process, but for simplicity we'll just return the current answer
                                logger.info(
                                    f"Completed web search reflection after {state['reflect_iterations']} iterations"
                                )
                                return {
                                    "messages":
                                    [AIMessage(content=final_answer)]
                                }

                        # Return final answer
                        return {"messages": [AIMessage(content=final_answer)]}
                    else:
                        # Fallback if structured response fails
                        logger.warning(
                            "Failed to get structured response from revisor")
                        return {
                            "messages": [
                                AIMessage(
                                    content=revised_response.content if
                                    hasattr(revised_response, 'content') else
                                    "I couldn't find a satisfactory answer to your question after searching the web."
                                )
                            ]
                        }

                except Exception as search_error:
                    logger.error(
                        f"Error in web search process: {search_error}",
                        exc_info=True)

                    # Extract what we can from the initial response for fallback
                    if hasattr(initial_response,
                               'tool_calls') and initial_response.tool_calls:
                        try:
                            tool_args = initial_response.tool_calls[0]["args"]
                            answer = tool_args.get("answer", "")
                            return {
                                "messages": [
                                    AIMessage(
                                        content=
                                        f"{answer}\n\nNote: I encountered an error while searching for additional information on the web."
                                    )
                                ]
                            }
                        except:
                            pass

                    return {
                        "messages": [
                            AIMessage(
                                content=
                                "I encountered an error while searching the web. Let me answer based on what I already know.\n\n"
                                + initial_response.content if hasattr(
                                    initial_response, 'content') else
                                "I encountered an error while processing your web search query."
                            )
                        ]
                    }

            except Exception as initial_response_error:
                logger.error(
                    f"Error generating initial response: {initial_response_error}",
                    exc_info=True)
                return {
                    "messages": [
                        AIMessage(
                            content=
                            "I encountered an error while processing your query. Please try again with a different question."
                        )
                    ]
                }

        else:
            # Continuing reflection - this shouldn't happen in the current flow
            # but included for completeness
            logger.info(
                f"Continuing web search reflection, iteration {state.get('reflect_iterations', 0)}"
            )

            if state.get("reflection_data") and state.get("reflection_data", {}).get(
                    "final_answer"):
                final_answer = state.get("reflection_data", {}).get("final_answer")
                return {"messages": [AIMessage(content=final_answer)]}
            else:
                return {
                    "messages": [
                        AIMessage(
                            content=
                            "I'm still researching your query, but I don't have enough information yet. Let me provide what I know so far."
                        )
                    ]
                }

    except Exception as e:
        logger.error(f"Error in web_search_agent_node: {e}", exc_info=True)
        error_response = "I encountered an error while searching the web. Please try again with a different query."
        return {"messages": [AIMessage(content=error_response)]}


def tavily_web_search_agent_node(state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Enhanced web search agent using Tavily Search API with clean status."""
    try:
        # Check if Tavily is available
        if not tavily_available or not tavily_api_key:
            logger.warning("Tavily search not available")
            return {
                "messages": [
                    AIMessage(content="I'm sorry, but web search functionality is currently unavailable. Please check your TAVILY_API_KEY.")
                ]
            }
            
        # Get query and context
        current_query = state["messages"][-1].content
        conversation_context = build_conversation_context(state)
        
        logger.info(f"Performing Tavily search for: {current_query}")
        
        # Initialize search client with increased max_results
        tavily_search = TavilySearchResults(
            max_results=8,  # Increased from default
            api_key=tavily_api_key,
            include_raw_content=True,
            include_domains=[]
        )
        
        # Perform direct search
        try:
            search_results = tavily_search.invoke({"query": current_query})
            
            if not search_results or len(search_results) == 0:
                logger.warning("No search results found")
                return {
                    "messages": [
                        AIMessage(content=f"I searched for information about '{current_query}' but couldn't find relevant results. Let me answer based on my existing knowledge.")
                    ]
                }
                
            # Format search context for the LLM with increased excerpt length
            formatted_results = []
            for i, result in enumerate(search_results):
                if isinstance(result, dict):
                    url = result.get("url", "")
                    title = result.get("title", f"Result {i+1}")
                    # Increased content length for better context
                    content = result.get("content", result.get("text", ""))[:1200]
                    
                    if url and content:
                        formatted_results.append(f"[Result {i+1}]\nTitle: {title}\nURL: {url}\nContent: {content}\n")
            
            search_context = "\n\n".join(formatted_results)
            
            # Prepare the prompt for the LLM
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful AI assistant with access to web search results.
            
            User query: {question}
            
            Previous conversation context:
            {conversation_context}
            
            Here are the search results:
            {search_results}
            
            Based on these search results, please provide a detailed, accurate, and comprehensive answer.
            
            Important: 
            1. Always cite your sources properly using [1], [2], etc.
            2. Focus on the most relevant information from the search results
            3. Add a "Sources:" section at the end listing all the URLs used
            """)
            
            # Generate response using the appropriate model
            llm = get_model(state["model_name"])
            response = llm.invoke(
                prompt.format(
                    question=current_query,
                    conversation_context=conversation_context,
                    search_results=search_context
                )
            )
            
            # Check if the response already includes sources
            final_content = response.content
            
            if "Sources:" not in final_content and "[" not in final_content:
                # Add sources manually
                final_content += "\n\nSources:\n"
                for i, result in enumerate(search_results):
                    if isinstance(result, dict) and result.get("url"):
                        final_content += f"[{i+1}] {result.get('title', 'Result')}: {result.get('url')}\n"
            
            return {"messages": [AIMessage(content=final_content)]}
            
        except Exception as search_error:
            logger.error(f"Error in Tavily search: {search_error}")
            return {
                "messages": [
                    AIMessage(content="I encountered an error while searching for information. I'll answer based on my existing knowledge instead.")
                ]
            }
            
    except Exception as e:
        logger.error(f"Error in tavily_web_search_agent_node: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(content="I encountered an error while processing your web search. Please try again with a different question.")
            ]
        }


def image_generator_agent_node(
        state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Generates an image based on the user's query, using conversation context to inform the generation."""
    try:
        # Get the current query and build conversation context
        current_query = state["messages"][-1].content
        conversation_context = build_conversation_context(state)
        
        # Initialize the Gemma2-9b-it model through ChatGroq
        try:
            llm = ChatGroq(model="gemma2-9b-it", temperature=0.7)
        except Exception as model_error:
            logger.error(f"Error initializing Gemma2-9b-it model: {model_error}", exc_info=True)
            # Fallback to another model
            try:
                logger.info("Falling back to llama-3.3-70b-versatile model")
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
            except Exception as fallback_error:
                logger.error(f"Error initializing fallback model: {fallback_error}", exc_info=True)
                return {
                    "messages": [
                        AIMessage(
                            content="I'm sorry, but I couldn't access the language models needed for image generation. Please try again later."
                        )
                    ]
                }
        
        # Create system prompt for optimizing image generation prompts
        system_prompt = """
        You are an expert image prompt engineer. Your task is to convert user requests into 
        effective prompts for an image generation API. The API works best with specific 
        descriptions that include:
        
        - Subject matter (what/who should be in the image)
        - Style (photorealistic, cartoon, oil painting, etc.)
        - Mood or atmosphere
        - Lighting conditions
        - Composition details
        - Color palette or theme
        
        Return ONLY the optimized prompt text without any explanations or additional text.
        """
        
        # Craft the input for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {conversation_context}\n\nUser request: {current_query}\n\nCreate an optimized image generation prompt:")
        ]
        
        # Generate the optimized prompt
        try:
            prompt_response = llm.invoke(messages)
            optimized_prompt = prompt_response.content.strip()
            logger.info(f"Generated optimized image prompt: {optimized_prompt[:50]}...")
            
            # Check if Replicate API token is available
            if not replicate_api_token:
                logger.error(
                    "REPLICATE_API_TOKEN is not set in environment variables")
                return {
                    "messages": [
                        AIMessage(
                            content=
                            "Error: Replicate API token is not configured properly. Please check your environment variables."
                        )
                    ]
                }

            # Log that we're about to make the API call
            logger.info(
                f"Making Replicate API call with prompt: {optimized_prompt[:50]}..."
            )

            # Call the Replicate API to generate the image
            output = replicate.run(
                "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
                input={
                    "prompt": optimized_prompt,
                    "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                    "scheduler": "K_EULER",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "refine": "expert_ensemble_refiner",
                    "high_noise_frac": 0.8,
                })

            # Extract the image URL from the output
            if output and isinstance(output, list) and len(output) > 0:
                image_url = output[0]
                logger.info(f"Image generated successfully: {image_url}")
                response = f" {image_url}"
            else:
                logger.error("No image URL in the output")
                response = "I couldn't generate an image. Please try a different description."

            return {"messages": [AIMessage(content=response)]}
            
        except Exception as prompt_error:
            logger.error(f"Error generating image prompt: {prompt_error}", exc_info=True)
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error while processing your image generation request. Please try again with a more specific description."
                    )
                ]
            }
            
    except Exception as e:
        logger.error(f"Error in image_generator_agent_node: {e}", exc_info=True)
        response = f"Error generating image: {str(e)}"
        return {"messages": [AIMessage(content=response)]}


def default_agent_node(state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Handles general queries or RAG-based Q&A when a document is indexed."""
    try:
        logger.info("Starting default_agent_node processing")
        
        # Get the appropriate model
        llm = get_model(state["model_name"])
        logger.info(f"Using model: {state['model_name']}")
        
        # Get the current query and build conversation context
        current_query = state["messages"][-1].content
        logger.info(f"Processing query: {current_query[:50]}...")
        conversation_context = build_conversation_context(state)
        logger.info(f"Built conversation context of length: {len(conversation_context)}")
        
        # Retrieve context from indexed document if available
        context = ""
        if state["indexed"] and state["collection_name"]:
            try:
                logger.info("Retrieving context for default agent")
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                embeddings = OpenAIEmbeddings(api_key=openai_key)
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=state["collection_name"],
                    embeddings=embeddings)
                retrieved_docs = vector_store.similarity_search(current_query,
                                                                k=3)
                context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved {len(retrieved_docs)} document chunks")
            except Exception as context_error:
                logger.error(f"Error retrieving context: {context_error}",
                             exc_info=True)
        
        # Create the prompt template
        prompt_template = """
        Conversation history:
        {conversation_context}
        
        {context_section}
        
        Question: {question}
        Answer:
        """
        
        # Add context section if available
        context_section = f"""
        Based on the conversation history and document context, answer the question.
        Document context:
        {context}
        """ if context else "Based on the conversation history, answer the question."
        
        # Create the prompt and generate a response
        prompt = ChatPromptTemplate.from_template(prompt_template)
        response = llm.invoke(
            prompt.format(conversation_context=conversation_context,
                          context_section=context_section,
                          question=current_query))
        
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"Error in default_agent_node: {e}", exc_info=True)
        error_response = "I encountered an error while processing your query. Please try again with a different question."
        return {"messages": [AIMessage(content=error_response)]}


def deep_research_agent_node(
        state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Handle deep research on a query."""
    logger.info("Deep research requested - using placeholder implementation")
    try:
        # Get the current query and build conversation context
        current_query = state["messages"][-1].content
        conversation_context = build_conversation_context(state)
        
        # Get the appropriate model
        llm = get_model(state["model_name"])
        
        # Perform web search
        web_content = ""
        try:
            logger.info("Performing deep web search")
            exa_client = Exa(api_key=exa_key)
            search_results = exa_client.search(current_query,
                                               use_autoprompt=True,
                                               num_results=5)
            web_content = "\n\n".join([
                f"Source: {r.url}\nTitle: {r.title}\nExcerpt: {r.text[:800]}..."
                for r in search_results.results
            ])
            logger.info(
                f"Retrieved {len(search_results.results)} deep search results")
        except Exception as search_error:
            logger.error(f"Error in deep web search: {search_error}")
            web_content = "Web search failed, proceeding with model's knowledge only."
        
        # Retrieve RAG context if available
        rag_context = ""
        if state["indexed"] and state["collection_name"]:
            try:
                logger.info("Retrieving RAG context for deep research")
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                embeddings = OpenAIEmbeddings(api_key=openai_key)
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=state["collection_name"],
                    embeddings=embeddings)
                retrieved_docs = vector_store.similarity_search(current_query,
                                                                k=5)
                rag_context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs])
                logger.info(
                    f"Retrieved {len(retrieved_docs)} document chunks for deep research"
                )
            except Exception as rag_error:
                logger.error(
                    f"Error retrieving RAG context for deep research: {rag_error}"
                )
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_template("""
        You are performing deep research on a topic. Provide a comprehensive, well-structured answer.
        
        Conversation history:
        {conversation_context}
        
        Research question: {question}
        
        {web_section}
        
        {rag_section}
        
        Provide a detailed, well-structured answer that synthesizes information from all available sources.
        Include citations or references where appropriate.
        Organize your response with clear sections and highlight key findings.
        """)
        
        # Add web and RAG sections if available
        web_section = f"Web search findings:\n{web_content}" if web_content else "No web search results available."
        rag_section = f"Document context:\n{rag_context}" if rag_context else "No document context available."
        
        # Generate response
        research_llm = get_model(state["model_name"])
        response = research_llm.invoke(
            prompt.format(conversation_context=conversation_context,
                          question=current_query,
                          web_section=web_section,
                          rag_section=rag_section))
        
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"Error in deep_research_agent_node: {e}", exc_info=True)
        error_response = "I encountered an error while performing deep research. Please try again with a more specific query."
        return {"messages": [AIMessage(content=error_response)]}


def music_generator_agent_node(state: VaaniState) -> Dict[str, List[BaseMessage]]:
    """Generate music based on user query and conversation context."""
    try:
        # Debug log for MUSICFY_API_KEY in the function
        logger.info(f"MUSICFY_API_KEY in music_generator_agent_node: {'Available' if musicfy_api_key else 'Not available'}")
        
        # Check if Musicfy API key is available
        if not musicfy_api_key:
            logger.error("MUSICFY_API_KEY is not set in environment variables")
            # Try to get it directly from environment again as a fallback
            direct_key = os.getenv('MUSICFY_API_KEY')
            logger.info(f"Direct MUSICFY_API_KEY check: {'Available' if direct_key else 'Not available'}")
            
            if direct_key:
                # Use the directly fetched key
                api_key_to_use = direct_key
                logger.info("Using directly fetched MUSICFY_API_KEY")
            else:
                # Hardcode as last resort
                api_key_to_use = "cm8009x9i0029lb0cp5dqjczu"
                logger.info("Using hardcoded MUSICFY_API_KEY as last resort")
                
            # Continue with the hardcoded key
            logger.info("Proceeding with available API key")
        else:
            api_key_to_use = musicfy_api_key
            logger.info("Using global MUSICFY_API_KEY variable")
            
        # Get the current query and build conversation context
        user_query = state["messages"][-1].content if state["messages"] else ""
        conversation_context = build_conversation_context(state)
        
        # Initialize the Gemma2-9b-it model through ChatGroq
        try:
            llm = ChatGroq(model="gemma2-9b-it", temperature=0.7)
        except Exception as model_error:
            logger.error(f"Error initializing Gemma2-9b-it model: {model_error}", exc_info=True)
            # Fallback to another model
            try:
                logger.info("Falling back to llama-3.3-70b-versatile model")
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
            except Exception as fallback_error:
                logger.error(f"Error initializing fallback model: {fallback_error}", exc_info=True)
                return {
                    "messages": [
                        AIMessage(
                            content="I'm sorry, but I couldn't access the language models needed for music generation. Please try again later."
                        )
                    ]
                }
        
        # Create system prompt for optimizing music generation prompts
        system_prompt = """
        You are an expert music prompt engineer. Your task is to convert user requests into 
        effective prompts for a music generation API. The API works best with specific 
        descriptions that include:
        
        - Musical genre or style
        - Tempo and rhythm characteristics
        - Mood or emotional qualities
        - Instrumental focus
        - Any specific cultural or regional influences
        
        Return ONLY the optimized prompt text without any explanations or additional text.
        """
        
        # Craft the input for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {conversation_context}\n\nUser request: {user_query}\n\nCreate an optimized music generation prompt:")
        ]
        
        # Generate the optimized prompt
        try:
            prompt_response = llm.invoke(messages)
            optimized_prompt = prompt_response.content.strip()
            logger.info(f"Generated optimized music prompt: {optimized_prompt[:50]}...")
            
            # Log that we're about to make the API call
            logger.info(f"Making Musicfy API call for music generation with prompt: {optimized_prompt[:50]}...")
            
            # Call the Musicfy API with error handling
            try:
                url = "https://api.musicfy.lol/v1/generate-music"
                
                payload = {"prompt": optimized_prompt}
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key_to_use}",
                }
                
                response = requests.request("POST", url, json=payload, headers=headers)
                
                # Debug log for API response
                logger.info(f"Musicfy API response status code: {response.status_code}")
                logger.info(f"Musicfy API response headers: {response.headers}")
                logger.info(f"Musicfy API response text: {response.text[:100]}...")
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response
                    result = json.loads(response.text)
                    
                    if result and isinstance(result, list) and len(result) > 0 and "file_url" in result[0]:
                        music_url = result[0]["file_url"]
                        logger.info(f"Music generated successfully: {music_url}")
                        response_text = f"I've created music based on your request. Here it is: {music_url}"
                    else:
                        logger.error(f"Unexpected response format from Musicfy API: {result}")
                        response_text = "I couldn't generate music. The API returned an unexpected response format. Please try again with a different description."
                else:
                    logger.error(f"Musicfy API error: {response.status_code} - {response.text}")
                    response_text = f"I encountered an error while generating music. The API returned status code {response.status_code}. Please try again later."
                    
                return {"messages": [AIMessage(content=response_text)]}
                
            except Exception as api_error:
                logger.error(f"Musicfy API error: {api_error}", exc_info=True)
                error_message = str(api_error).lower()
                
                if "unauthorized" in error_message or "authentication" in error_message:
                    return {
                        "messages": [
                            AIMessage(
                                content="I'm sorry, but there's an authentication issue with the music generation service. The API key might be invalid or expired. Please contact the administrator."
                            )
                        ]
                    }
                elif "rate limit" in error_message:
                    return {
                        "messages": [
                            AIMessage(
                                content="I'm sorry, but we've hit the rate limit for the music generation service. Please try again later."
                            )
                        ]
                    }
                else:
                    return {
                        "messages": [
                            AIMessage(
                                content=f"I encountered an error while generating music: {str(api_error)}. Please try again with a different description."
                            )
                        ]
                    }
            
        except Exception as prompt_error:
            logger.error(f"Error generating music prompt: {prompt_error}", exc_info=True)
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error while processing your music generation request. Please try again with a more specific description that includes genre, mood, tempo, and instruments."
                    )
                ]
            }
            
    except Exception as e:
        logger.error(f"Error in music_generator_agent_node: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error while generating music. Please try again later or contact the administrator if the issue persists."
                )
            ]
        }


# Create and compile the graph
def create_graph():
    """Creates and compiles the LangGraph with proper error handling."""
    try:
        graph = StateGraph(VaaniState)
        graph.add_node("entry_node", entry_node)
        graph.add_node("summarizer", summarizer_node)
        graph.add_node("orchestrator", orchestrator_node)
        graph.add_node("rag_agent", rag_agent_node)
        graph.add_node("web_search_agent", tavily_web_search_agent_node if tavily_available and tavily_api_key else default_agent_node)
        graph.add_node("image_generator", image_generator_agent_node)
        graph.add_node("default_agent", default_agent_node)
        graph.add_node("deep_research", deep_research_agent_node)
        graph.add_node("music_generator", music_generator_agent_node)
        graph.add_edge(START, "entry_node")

        def entry_router(state):
            """Route based on state without modifying it"""
            if state["deep_research_requested"]:
                return "deep_research"
            if state["file_url"]:
                if is_image_file(state["file_url"]):
                    return "web_search_agent"
                elif is_document_file(
                        state["file_url"]) and not state["indexed"]:
                    return "rag_agent"
            return "orchestrator"

        graph.add_conditional_edges(
            "entry_node", entry_router, {
                "deep_research": "deep_research",
                "web_search_agent": "web_search_agent",
                "rag_agent": "rag_agent",
                "orchestrator": "summarizer"
            })
        graph.add_edge("summarizer", "orchestrator")

        def orchestrator_router(state):
            """Router function for orchestrator node"""
            return state["agent_name"]

        graph.add_conditional_edges(
            "orchestrator", orchestrator_router, {
                "rag_agent": "rag_agent",
                "web_search_agent": "web_search_agent",
                "image_generator": "image_generator",
                "music_generator": "music_generator",
                "default": "default_agent"
            })
        for agent in [
                "rag_agent", "web_search_agent", "image_generator", "music_generator", "default_agent", "deep_research"
        ]:
            graph.add_edge(agent, END)
        try:
            if SqliteSaver is not None:
                memory = SqliteSaver(connection_string="sqlite:///vaani.db")
                compiled_graph = graph.compile(checkpointer=memory)
                logger.info("Graph successfully compiled with SQLite checkpointer")
            else:
                logger.info("SqliteSaver not available, using in-memory checkpointer")
                compiled_graph = graph.compile()
            return compiled_graph
        except Exception as memory_error:
            logger.error(
                f"Error setting up memory with SQLite: {memory_error}")
            logger.info("Falling back to in-memory checkpointer")
            compiled_graph = graph.compile()
            return compiled_graph
    except Exception as e:
        logger.error(f"Error creating graph: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create agent graph: {str(e)}")


try:
    logger.info("Initializing Vaani.pro agent graph")
    graph = create_graph()
except Exception as e:
    logger.critical(f"Fatal error initializing agent graph: {e}",
                    exc_info=True)
    raise

if __name__ == "__main__":
    logger.info("agent.py executed successfully")