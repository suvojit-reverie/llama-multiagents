
import getpass
import os
import shutil
import uuid
import json
from langchain_aws import ChatBedrockConverse
from flask import Flask, request, jsonify, Response
import json
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
from langchain_groq import ChatGroq

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

# Load environment variables from .env file
load_dotenv()

import os
import shutil
# import sqlite3

# import pandas as pd
import requests
import re
import numpy as np
import openai
from langchain_core.tools import tool

# response = requests.get(
#     "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
# )
# response.raise_for_status()
# faq_text = response.text

# docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


def load_electricity_faq_docs(file_path: str) -> list:
    """
    Load electricity FAQ documents from a specified file.

    Args:
        file_path (str): The path to the file containing the electricity FAQ documents.

    Returns:
        list: A list of dictionaries, each containing the page content of the FAQ documents.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return [{"page_content": line.strip()} for line in file.readlines()]


# Usage
electricity_faq_docs = load_electricity_faq_docs("electricty_connection_rag_doc.txt")


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


faq_retriever = VectorStoreRetriever.from_docs(electricity_faq_docs, openai.Client())


# Initialize Qdrant client
# qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

# Create a collection in Qdrant
# def create_qdrant_collection(collection_name: str, vector_size: int):
#     try:
#         qdrant_client.create_collection(
#             collection_name=collection_name,
#             vectors_config={"size": vector_size, "distance": "Cosine"}
#         )
#     except Exception as e:
#         print(f"Failed to create collection {collection_name}: {e}")

# # Insert documents into Qdrant
# def insert_documents_to_qdrant(collection_name: str, docs: list, vectors: list):
#     points = [
#         PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"page_content": doc["page_content"]})
#         for doc, vector in zip(docs, vectors)
#     ]
#     qdrant_client.upsert(collection_name=collection_name, points=points)

# # Query Qdrant for similar documents
# def query_qdrant(collection_name: str, query_vector: list, k: int = 5) -> list[dict]:
#     search_result = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=k
#     )
#     return [
#         {"page_content": hit.payload["page_content"], "similarity": hit.score}
#         for hit in search_result
#     ]

# Create Qdrant collection and insert documents
# Check if the collection exists before creating it
# collection_name = "faq_collection"

# try:
#     qdrant_client.get_collection(collection_name)
# except Exception as e:
#     # logger.info(f"Collection {self._collection_name} not found. Creating a new one.")
#     qdrant_client.create_collection(
#         collection_name=collection_name,
#         vectors_config={
#             "size": 1536,
#             "distance": "Cosine"
#         }
#     )

# Update VectorStoreRetriever to use Qdrant
# class QdrantVectorStoreRetriever(VectorStoreRetriever):
#     def __init__(self, docs: list, oai_client, collection_name: str):
#         self._docs = docs
#         self._client = oai_client
#         self._collection_name = collection_name

#     @classmethod
#     def from_docs(cls, docs, oai_client, collection_name: str):
#         embeddings = oai_client.embeddings.create(
#             model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
#         )
#         vectors = [emb.embedding for emb in embeddings.data]
#         insert_documents_to_qdrant(collection_name, docs, vectors)
#         return cls(docs, oai_client, collection_name)

#     def query(self, query: str, k: int = 5) -> list[dict]:
#         embed = self._client.embeddings.create(
#             model="text-embedding-3-small", input=[query]
#         )
#         query_vector = embed.data[0].embedding
#         return query_qdrant(self._collection_name, query_vector, k)


# insert_documents_to_qdrant(collection_name, docs, retriever._arr.tolist())

# Use QdrantVectorStoreRetriever
# qdrant_retriever = QdrantVectorStoreRetriever.from_docs(electricity_faq_docs, openai.Client(), "faq_collection")


@tool
def set_language(language: str) -> str:
    """Based on user input for language return the language code
        Args:
        language (str): The language input of the user.

    Returns:
        str: A str containing language code.
    
    """
    language_mapping = {
        "english": "en",
        "hindi": "hi",
        "marathi": "mr",
        "kannada": "kn"
    }
    
    # Convert the input language to lowercase to make the comparison case-insensitive
    language = language.lower()
    
    # Return the corresponding language code or None if not found
    return language_mapping.get(language, "en")


@tool
def lookup_electricity_connection_policy(query: str) -> str:
    """Consult the electricity connection related documents to answer user questions. This document contains all the policy and FAQs related to electricity connection related queries"""

    docs = faq_retriever.query(query, k=2)

    # print the docs
    for doc in docs:
        print(doc["page_content"])

    return "\n\n".join([doc["page_content"] for doc in docs])


import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig


@tool
def get_user_information(phone_number: Optional[str] = None) -> dict:
    """
    Retrieve user information based on the phone number from various collections in the MongoDB database.

    Args:
        phone_number (Optional[str]): The phone number of the user. Defaults to None.

    Returns:
        dict: A dictionary containing user information from different collections.
    """
    user_info = {}

    if phone_number:
        # Check water connections
        water_connection = water_connections_collection.find_one(
            {"phone_number": phone_number}
        )
        if water_connection:
            user_info["water_connection"] = water_connection

        # Check electricity connections
        electricity_connection = electricity_connections_collection.find_one(
            {"phone_number": phone_number}
        )
        if electricity_connection:
            user_info["electricity_connection"] = electricity_connection

        # Check water issues
        water_issue = water_issues_collection.find_one({"phone_number": phone_number})
        if water_issue:
            user_info["water_issue"] = water_issue

        # Check electricity issues
        electricity_issue = electricity_connections_collection.find_one(
            {"phone_number": phone_number}
        )
        if electricity_issue:
            user_info["electricity_issue"] = electricity_issue

    if not user_info:
        return {"error": "No information found for the provided phone number."}

    return user_info


from datetime import date, datetime
from typing import Optional, Union


@tool
def handle_electricity_issue(issue_description: str) -> str:
    """
    Handle electricity-related issues by informing the user that the issue will be fixed within the next hour.

    Args:
        issue_description (str): A detailed description of the electricity issue the user is facing.

    Returns:
        str: A message indicating that the issue will be fixed within the next hour.
    """
    # Here you can add any logic to log the issue, notify a technician, etc.
    return f"We have received your report: '{issue_description}'. Our team will fix the issue within the next hour."


# tools for new water connection
# Initialize MongoDB client
client = MongoClient(os.getenv("MONGODB_URI"))
customer_support_db = client.get_database("customer_support_db")
water_connections_collection = customer_support_db.get_collection("water_connections")


@tool
def add_or_update_water_connection(
    address: str, request: str, phone_number: str
) -> str:
    """
    Add a new water connection record to the database or update an existing one if the phone number already exists.

    Args:
        address (str): The address where the new water connection is requested.
        request (str): Any additional information or requests from the user regarding the new water connection.
        phone_number (str): The phone number of the user requesting the new water connection.

    Returns:
        str: A message indicating whether the new water connection was successfully added or updated.
    """
    existing_connection = water_connections_collection.find_one(
        {"phone_number": phone_number}
    )

    if existing_connection:
        update_fields = {
            "address": address,
            "request": request,
            "status": "pending",
            "updated_at": datetime.now(),
        }
        result = water_connections_collection.update_one(
            {"phone_number": phone_number}, {"$set": update_fields}
        )
        if result.modified_count > 0:
            return f"Water connection request for phone number {phone_number} successfully updated."
        else:
            return f"No changes made to the water connection request for phone number {phone_number}."
    else:
        new_connection = {
            "address": address,
            "request": request,
            "phone_number": phone_number,
            "status": "pending",
            "created_at": datetime.now(),
        }
        result = water_connections_collection.insert_one(new_connection)
        if result.inserted_id:
            return f"New water connection request successfully added with ID {result.inserted_id}."
        else:
            return "Failed to add new water connection request."


@tool
def check_pending_requests(phone_number: str) -> list[dict]:
    """
    Check if there are any pending requests for a given phone number.

    Args:
        phone_number (str): The phone number to check for pending requests.

    Returns:
        list[dict]: A list of pending requests associated with the given phone number.
    """
    pending_requests = water_connections_collection.find(
        {"phone_number": phone_number, "status": "pending"}
    )
    return list(pending_requests)


@tool
def update_water_connection_request(
    request_id: str,
    address: Optional[str] = None,
    request: Optional[str] = None,
    phone_number: Optional[str] = None,
) -> str:
    """
    Update an existing water connection request in the database.

    Args:
        request_id (str): The ID of the water connection request to update.
        address (Optional[str]): The new address for the water connection. Defaults to None.
        request (Optional[str]): The new request details. Defaults to None.
        phone_number (Optional[str]): The new phone number for the request. Defaults to None.

    Returns:
        str: A message indicating whether the water connection request was successfully updated or not.
    """
    update_fields = {}
    if address:
        update_fields["address"] = address
    if request:
        update_fields["request"] = request
    if phone_number:
        update_fields["phone_number"] = phone_number

    if not update_fields:
        return "No fields to update."

    result = water_connections_collection.update_one(
        {"_id": request_id}, {"$set": update_fields}
    )

    if result.modified_count > 0:
        return f"Water connection request {request_id} successfully updated."
    else:
        return f"No water connection request found with ID {request_id} or no changes made."


# Tools for electricity issues
# Initialize MongoDB client for electricity connections
electricity_connections_collection = customer_support_db.get_collection(
    "electricity_connections"
)


@tool
def add_new_electricity_connection(
    address: str, request: str, phone_number: str
) -> str:
    """
    Add a new electricity connection record to the database.

    Args:
        address (str): The address where the new electricity connection is requested.
        request (str): Any additional information or requests from the user regarding the new electricity connection.
        phone_number (str): The phone number of the user requesting the new electricity connection.

    Returns:
        str: A message indicating whether the new electricity connection was successfully added or not.
    """
    new_connection = {
        "address": address,
        "request": request,
        "phone_number": phone_number,
        "status": "pending",
        "created_at": datetime.now(),
    }
    result = electricity_connections_collection.insert_one(new_connection)
    if result.inserted_id:
        return f"New electricity connection request successfully added with ID {result.inserted_id}."
    else:
        return "Failed to add new electricity connection request."


@tool
def check_electricity_request_status(phone_number: str) -> list[dict]:
    """
    Check the status of electricity connection requests for a given phone number.

    Args:
        phone_number (str): The phone number to check for electricity connection requests.

    Returns:
        list[dict]: A list of electricity connection requests associated with the given phone number.
    """
    requests = electricity_connections_collection.find({"phone_number": phone_number})
    return list(requests)


@tool
def check_pending_electricity_requests(phone_number: str) -> list[dict]:
    """
    Check if there are any pending electricity connection requests for a given phone number.

    Args:
        phone_number (str): The phone number to check for pending electricity connection requests.

    Returns:
        list[dict]: A list of pending electricity connection requests associated with the given phone number.
    """
    pending_requests = electricity_connections_collection.find(
        {"phone_number": phone_number, "status": "pending"}
    )
    return list(pending_requests)


@tool
def update_electricity_connection_request(
    request_id: str,
    address: Optional[str] = None,
    request: Optional[str] = None,
    phone_number: Optional[str] = None,
) -> str:
    """
    Update an existing electricity connection request in the database.

    Args:
        request_id (str): The ID of the electricity connection request to update.
        address (Optional[str]): The new address for the electricity connection. Defaults to None.
        request (Optional[str]): The new request details. Defaults to None.
        phone_number (Optional[str]): The new phone number for the request. Defaults to None.

    Returns:
        str: A message indicating whether the electricity connection request was successfully updated or not.
    """
    update_fields = {}
    if address:
        update_fields["address"] = address
    if request:
        update_fields["request"] = request
    if phone_number:
        update_fields["phone_number"] = phone_number

    if not update_fields:
        return "No fields to update."

    result = electricity_connections_collection.update_one(
        {"_id": request_id}, {"$set": update_fields}
    )

    if result.modified_count > 0:
        return f"Electricity connection request {request_id} successfully updated."
    else:
        return f"No electricity connection request found with ID {request_id} or no changes made."


# Tools for water issues

# Initialize MongoDB client for electricity connections
water_issues_collection = customer_support_db.get_collection("water_issues")


@tool
def register_or_update_water_issue(
    issue_description: str, address: str, phone_number: str
) -> str:
    """
    Register a new water-related issue or update an existing one in the database.

    Args:
        issue_description (str): A detailed description of the water issue the user is facing.
        address (str): The address where the issue is occurring.
        phone_number (str): The phone number of the user reporting the issue.

    Returns:
        str: A message indicating whether the water issue was successfully registered or updated, along with the ticket ID.
    """
    existing_issue = water_issues_collection.find_one({"phone_number": phone_number})

    if existing_issue:
        update_fields = {
            "issue_description": issue_description,
            "address": address,
            "status": "pending",
            "updated_at": datetime.now(),
        }
        result = water_issues_collection.update_one(
            {"phone_number": phone_number}, {"$set": update_fields}
        )
        if result.modified_count > 0:
            return f"Water issue for phone number {phone_number} successfully updated."
        else:
            return (
                f"No changes made to the water issue for phone number {phone_number}."
            )
    else:
        ticket_id = str(uuid.uuid4())
        new_issue = {
            "ticket_id": ticket_id,
            "issue_description": issue_description,
            "address": address,
            "phone_number": phone_number,
            "status": "pending",
            "created_at": datetime.now(),
        }
        result = water_issues_collection.insert_one(new_issue)
        if result.inserted_id:
            return f"Water issue successfully registered with ticket ID {ticket_id}."
        else:
            return "Failed to register water issue."


@tool
def check_water_issue_status(
    ticket_id: Optional[str] = None, phone_number: Optional[str] = None
) -> list[dict]:
    """
    Check the status of water-related issues by ticket ID or phone number.

    Args:
        ticket_id (Optional[str]): The ticket ID of the water issue to check. Defaults to None.
        phone_number (Optional[str]): The phone number associated with the water issue to check. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries containing the status and details of the water issues.
    """
    query = {}
    if ticket_id:
        query["ticket_id"] = ticket_id
    if phone_number:
        query["phone_number"] = phone_number

    if not query:
        return {"error": "Either ticket_id or phone_number must be provided."}

    issues = water_issues_collection.find(query)
    result = [
        {
            "ticket_id": issue["ticket_id"],
            "issue_description": issue["issue_description"],
            "address": issue["address"],
            "phone_number": issue["phone_number"],
            "status": issue["status"],
            "created_at": issue["created_at"],
        }
        for issue in issues
    ]

    if result:
        return result
    else:
        return {"error": "No water issues found with the provided criteria."}


# Tools for ambulance service

# Initialize MongoDB client for ambulance service
ambulance_service_collection = customer_support_db.get_collection("ambulance_service")


@tool
def book_ambulance_service(
    pickup_location: str, destination: str, phone_number: str
) -> str:
    """
    Book an ambulance service for the user.

    Args:
        pickup_location (str): The pickup location for the ambulance service.
        destination (str): The destination for the ambulance service.
        phone_number (str): The phone number of the user requesting the ambulance service.

    Returns:
        str: A message indicating whether the ambulance service was successfully booked or not.
    """
    booking_id = str(uuid.uuid4())
    new_booking = {
        "booking_id": booking_id,
        "pickup_location": pickup_location,
        "destination": destination,
        "phone_number": phone_number,
        "status": "booked",
        "created_at": datetime.now(),
    }
    result = ambulance_service_collection.insert_one(new_booking)
    if result.inserted_id:
        return f"Ambulance service successfully booked with booking ID {booking_id}."
    else:
        return "Failed to book ambulance service."


@tool
def check_ambulance_booking_status(booking_id: str) -> dict:
    """
    Check the status of an ambulance service booking.

    Args:
        booking_id (str): The booking ID of the ambulance service.

    Returns:
        dict: A dictionary containing the status and details of the ambulance service booking.
    """
    booking = ambulance_service_collection.find_one({"booking_id": booking_id})
    if booking:
        return {
            "booking_id": booking["booking_id"],
            "pickup_location": booking["pickup_location"],
            "destination": booking["destination"],
            "phone_number": booking["phone_number"],
            "status": booking["status"],
            "created_at": booking["created_at"],
        }
    else:
        return {
            "error": "No ambulance service booking found with the provided booking ID."
        }


@tool
def update_ambulance_booking(
    booking_id: str,
    pickup_location: Optional[str] = None,
    destination: Optional[str] = None,
    phone_number: Optional[str] = None,
) -> str:
    """
    Update an existing ambulance service booking in the database.

    Args:
        booking_id (str): The ID of the ambulance service booking to update.
        pickup_location (Optional[str]): The new pickup location for the ambulance service. Defaults to None.
        destination (Optional[str]): The new destination for the ambulance service. Defaults to None.
        phone_number (Optional[str]): The new phone number for the booking. Defaults to None.

    Returns:
        str: A message indicating whether the ambulance service booking was successfully updated or not.
    """
    update_fields = {}
    if pickup_location:
        update_fields["pickup_location"] = pickup_location
    if destination:
        update_fields["destination"] = destination
    if phone_number:
        update_fields["phone_number"] = phone_number

    if not update_fields:
        return "No fields to update."

    result = ambulance_service_collection.update_one(
        {"booking_id": booking_id}, {"$set": update_fields}
    )

    if result.modified_count > 0:
        return f"Ambulance service booking {booking_id} successfully updated."
    else:
        return f"No ambulance service booking found with ID {booking_id} or no changes made."


"""#### Utilities

Define helper functions to pretty print the messages in the graph while we debug it and to give our tool node error handling (by adding the error to the chat history).
"""

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    # print("Control is coming here")

    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]

        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print("Message repr: ", msg_repr)
            _printed.add(message.id)


def custom_print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:

        print(json.dumps(message, default=str))

        if isinstance(message, list):
            message = message[-1]

        if message.id not in _printed:
            if "response_metadata" in message:
                print("Agent Message: ", message)
            else:
                print("User Message: ", message)

            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print("Message repr: ", msg_repr)
            _printed.add(message.id)


from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


"""#### Agent

Next, define the assistant function. This function takes the graph state, formats it into a prompt, and then calls an LLM for it to predict the best response.
"""

# from langchain_openai import ChatOenai
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY"),
)

# llm = ChatBedrockConverse(
#     # model="us.meta.llama3-2-3b-instruct-v1:0",
#     model="meta.llama3-1-70b-instruct-v1:0",
#     temperature=0,
#     max_tokens=None,
#     # other params...
# )

"""#### Define Graph

Now, create the graph. The graph is the final assistant for this section.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition


import shutil
import uuid


from typing import Annotated

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "electricity_issue",
                "water_issue",
            ]
        ],
        update_dialog_stack,
    ]


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from pydantic import BaseModel, Field


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


def create_specialized_assistant(
    system_message: str,
    safe_tools: list,
    sensitive_tools: list,
    additional_examples: list = None,
) -> Runnable:
    examples = "\n\nSome examples for which you should CompleteOrEscalate:\n"
    if additional_examples:
        examples += "\n".join([f" - '{example}'" for example in additional_examples])
    else:
        examples += (
            " - 'what's the weather like this time of year?'\n"
            " - 'What flights are available?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Task completed!'"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message
                + "\nCurrent time: {time}."
                + '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
                + examples,
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

    tools = safe_tools + sensitive_tools
    return prompt | llm.bind_tools(tools + [CompleteOrEscalate])


# Electricity Issue Assistant
electricity_issue_system_message = (
"Context: You handle complaints related to existing services. You will collect the complaint details, submit the form via an API, and issue a complaint tracking ID."  
"Objective: Gather complaint details, probe into user issues when necessary, submit the form, and provide a complaint tracking ID."  
"Style & Tone: Empathetic, supportive, probing, and efficient."  
"Audience: Users raising complaints about electricity, water, or other related services."  
"Response Format: Ask for all relevant information (e.g., type of issue, account number, service details, etc.), submit the complaint, and provide a tracking ID."  
"Responsibilities:"  
"Ask the user to elaborate on their problem or complaint."  
"Gather all pertinent details to document the complaint thoroughly. If the user offers brief, insufficient, or vague responses, ask relevant probing questions to gain a deeper understanding of the issue."  
"Submit the complaint via the API"
"Provide a Complaint tracking ID."  
"Ask if the user needs help with anything else related to complaints, if no, conclude the conversation appropriately."  
"If the query is beyond your scope, transfer to the relevant agent or ask the user to be on hold while you trabsfer them to a live agent."
)
electricity_issue_safe_tools = [handle_electricity_issue]
electricity_issue_sensitive_tools = []
electricity_issue_runnable = create_specialized_assistant(
    electricity_issue_system_message,
    electricity_issue_safe_tools,
    electricity_issue_sensitive_tools,
)

# Water Issue Assistant
water_issue_system_message = (
"Context: You handle complaints related to existing services. You will collect the complaint details, submit the form via an API, and issue a complaint tracking ID."  
"Objective: Gather complaint details, probe into user issues when necessary, submit the form, and provide a complaint tracking ID."  
"Style & Tone: Empathetic, supportive, probing, and efficient."  
"Audience: Users raising complaints about electricity, water, or other related services."  
"Response Format: Ask for all relevant information (e.g., type of issue, account number, service details, etc., ask these bits of information one by one), submit the complaint, and provide a tracking ID Strictly keep your responses between 20 to 35 words at max."  
"Responsibilities:"  
"Ask the user to elaborate on their problem or complaint."  
"Probe to gather all pertinent details to document the complaint thoroughly. If the user offers brief, insufficient, or vague responses, ask relevant probing questions to gain a deeper understanding of the issue."  
"Submit the complaint via the API"
"Provide a Complaint tracking ID."  
"Ask if the user needs help with anything else related to complaints, if no, conclude the conversation appropriately."  
"If the query is beyond your scope, transfer to the relevant agent or ask the user to be on hold while you trabsfer them to a live agent."
)
water_issue_safe_tools = [register_or_update_water_issue, check_water_issue_status]
water_issue_sensitive_tools = []
water_issue_runnable = create_specialized_assistant(
    water_issue_system_message, water_issue_safe_tools, water_issue_sensitive_tools
)

# New Water Connection Assistant
new_water_connection_system_message = (
    "Context: You handle new water connection applications. You will guide users through the process, collect necessary information, submit the form via an API, and provide an application tracking ID."
    "Objective: Collect all required details for a water connection application, submit it, and issue a tracking ID."
    "Style & Tone: Patient, helpful, and informative."
    "Audience: Users applying for a new water connection."
    "Response Format: Ask for all relevant information (name, address, phone number, property details, etc.,) ask one piece of information at a time, submit the application, and provide a tracking ID. Strictly keep your responses between 20 to 35 words at max."
    "Responsibilities:"
    "- Confirm the user's request for a new water connection."
    "- Collect all necessary details (name, address, phone number, property location, etc. strictly one piece of information at a time)."
    "- Submit the application via the API and provide the user with a tracking ID."
    "- Ask if the user needs help with anything else related to electricity services, If no, conclude the conversation, If yes, go back to the main agent."
    "- If the query is beyond your scope, transfer to the relevant agent or ask the user to be on hold while you trabsfer them to a live agent."
)
new_water_connection_safe_tools = []
new_water_connection_sensitive_tools = [
    add_or_update_water_connection,
    check_pending_requests,
    update_water_connection_request,
]
new_water_connection_runnable = create_specialized_assistant(
    new_water_connection_system_message,
    new_water_connection_safe_tools,
    new_water_connection_sensitive_tools,
)

# New Electricity Connection Assistant
new_electricity_connection_system_message = (
    "Context: You handle new electricity connection applications. You will guide users through the process, collect necessary information, submit the form via an API, and provide an application tracking ID."
    "Objective: Collect all required details for an electricity connection application, submit it, and issue a tracking ID. If user has any queries, help them effeciently"
    "Style & Tone: Professional, helpful, informational, and detail-oriented."
    "Audience: Users applying for a new electricity connection."
    "Response Format: Ask for all relevant information (name, address, phone number, ID proof, etc.) one piece of information at a time, submit the application, and provide a tracking ID. Strictly keep your responses between 20 to 35 words at max."
    "Responsibilities:"
    "- Confirm the user's request for a new electricity connection."
    "- Collect all necessary details (name, address, phone number, etc. ne piece of information at a time)."
    "- Submit the application via the API and generate a tracking ID for the user."
    "- Ask if the user needs help with anything else related to electricity services, If no, conclude the conversation, If yes, go back to the main agent."
    "- If the query is beyond your scope, sk the user to be on hold while you transfer them to a live agent."
)
new_electricity_connection_safe_tools = [
    check_electricity_request_status,
    check_pending_electricity_requests,
]
new_electricity_connection_sensitive_tools = [
    add_new_electricity_connection,
    update_electricity_connection_request,
]
new_electricity_connection_runnable = create_specialized_assistant(
    new_electricity_connection_system_message,
    new_electricity_connection_safe_tools,
    new_electricity_connection_sensitive_tools,
)

# Electricity Connection Policy Assistant
electricity_connection_policy_system_message = (
    "You are a specialized assistant for handling queries related to electricity connection policies. "
    "The primary assistant delegates work to you whenever the user needs help with electricity connection policies. "
    "Consult the electricity connection policy documents to answer user questions accurately. "
    "If you need more information or the customer changes their mind, escalate the task back to the main assistant and ask the same prompt briefly."
    "If the query is beyond your scope, transfer to the relevant agent or ask the user to be on hold while you transfer them to a live agent."
)

electricity_connection_policy_safe_tools = [lookup_electricity_connection_policy]
electricity_connection_policy_sensitive_tools = []

electricity_connection_policy_runnable = create_specialized_assistant(
    electricity_connection_policy_system_message,
    electricity_connection_policy_safe_tools,
    electricity_connection_policy_sensitive_tools,
)

# Ambulance Service Assistant
ambulance_service_system_message = (
    "Context: You are the Ambulance Services Agent within the State Citizen Services bot on WhatsApp. Your primary role is to assist users who are reporting medical emergencies and need ambulance services. You must recognize the urgency in the user’s language. If the user mentions an emergency, uses the word “emergency,” or expresses anything urgent, you should immediately transfer the call to a live human agent. Let the user know to hold on while you arrange for a quick call from a live agent to handle the situation. If the user’s message does not convey urgency, you will gather the necessary details (name, full address, timing, etc.) and arrange ambulance services accordingly."
    "Objective: Handle medical emergency requests by recognizing urgency in the user’s language and acting appropriately. Transfer the conversation to a live human agent for urgent situations or arrange an ambulance for less urgent cases after collecting required information."
    "Style & Tone: Empathetic, calm, and responsive. Be urgent when required, but always supportive. Convey professionalism and understanding in medical emergencies."
    "Audience: General public reporting medical emergencies on WhatsApp. Ensure the language is clear and comforting, especially in distressing situations."
    "Response Format:"
    "If the user indicates an emergency (e.g., 'emergency,' 'urgent,' 'someone is hurt'), immediately transfer the call to a live agent for help."
    "If the user does not indicate urgency, ask for the necessary details such as name, full address, timing, and nature of the issue to arrange an ambulance."
    "In either case, clearly inform the user of the action you are taking."
    "Reassure the user that help is being dispatched."
    "Strictly keep your responses between 20 to 35 words at max."
    "Responsibilities:"
    "1. Recognize Urgency"
    "If the user mentions an emergency or urgency, respond immediately by arranging a live agent call transfer."
    "If the situation seems less urgent, proceed by asking for the necessary details to arrange an ambulance."
    "2. Urgent Response"
    "If urgent: Inform the user that a live agent will handle the situation and transfer the call."
    "3. Non-Urgent Response"
    "If not urgent: Collect details step-by-step (name, full address, time, and nature of the issue) and arrange ambulance service."
    "4. Reassure and End the Conversation"
    "Confirm that help is on the way and offer additional assistance if needed. End the conversation politely."
)
ambulance_service_safe_tools = []
ambulance_service_sensitive_tools = [
    book_ambulance_service,
    check_ambulance_booking_status,
    update_ambulance_booking,
]
ambulance_service_runnable = create_specialized_assistant(
    ambulance_service_system_message,
    ambulance_service_safe_tools,
    ambulance_service_sensitive_tools,
)

# Tools for fire brigade service

# Initialize MongoDB client for fire brigade service
fire_brigade_service_collection = customer_support_db.get_collection("fire_brigade_service")

@tool
def book_fire_brigade_service(location: str, phone_number: str, additional_info: Optional[str] = None) -> str:
    """
    Book a fire brigade service for the user.

    Args:
        location (str): The location of the fire.
        phone_number (str): The phone number of the user requesting the fire brigade service.
        additional_info (Optional[str]): Any additional information provided by the user. Defaults to None.

    Returns:
        str: A message indicating whether the fire brigade service was successfully booked or not.
    """
    request_id = str(uuid.uuid4())
    new_request = {
        "request_id": request_id,
        "location": location,
        "phone_number": phone_number,
        "additional_info": additional_info,
        "status": "requested",
        "created_at": datetime.now(),
    }
    result = fire_brigade_service_collection.insert_one(new_request)
    if result.inserted_id:
        return f"Fire brigade service successfully requested with request ID {request_id}."
    else:
        return "Failed to request fire brigade service."

@tool
def check_fire_brigade_request_status(request_id: str) -> dict:
    """
    Check the status of a fire brigade service request.

    Args:
        request_id (str): The request ID of the fire brigade service.

    Returns:
        dict: A dictionary containing the status and details of the fire brigade service request.
    """
    request = fire_brigade_service_collection.find_one({"request_id": request_id})
    if request:
        return {
            "request_id": request["request_id"],
            "location": request["location"],
            "phone_number": request["phone_number"],
            "additional_info": request.get("additional_info"),
            "status": request["status"],
            "created_at": request["created_at"],
        }
    else:
        return {
            "error": "No fire brigade service request found with the provided request ID."
        }

@tool
def update_fire_brigade_request(
    request_id: str,
    location: Optional[str] = None,
    phone_number: Optional[str] = None,
    additional_info: Optional[str] = None,
) -> str:
    """
    Update an existing fire brigade service request in the database.

    Args:
        request_id (str): The ID of the fire brigade service request to update.
        location (Optional[str]): The new location of the fire. Defaults to None.
        phone_number (Optional[str]): The new phone number for the request. Defaults to None.
        additional_info (Optional[str]): The new additional information for the request. Defaults to None.

    Returns:
        str: A message indicating whether the fire brigade service request was successfully updated or not.
    """
    update_fields = {}
    if location:
        update_fields["location"] = location
    if phone_number:
        update_fields["phone_number"] = phone_number
    if additional_info:
        update_fields["additional_info"] = additional_info

    if not update_fields:
        return "No fields to update."

    result = fire_brigade_service_collection.update_one(
        {"request_id": request_id}, {"$set": update_fields}
    )

    if result.modified_count > 0:
        return f"Fire brigade service request {request_id} successfully updated."
    else:
        return f"No fire brigade service request found with ID {request_id} or no changes made."


# Fire Brigade Service Assistant
fire_brigade_service_system_message = (
    "You are a specialized assistant for handling fire brigade service requests. "
    "The primary assistant delegates work to you whenever the user needs help with requesting fire brigade services. "
    "Collect necessary details from the user such as name, phone number, location of the fire, and any additional information. (collect these bits of information one at a time). "
    "Confirm the request and provide the user with a reference number. "
    "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
    "If the query is beyond your scope, ask the user to be on hold while you transfer them to a live agent who handles emergencies."
)
fire_brigade_service_safe_tools = []
fire_brigade_service_sensitive_tools = [
    book_fire_brigade_service,
    check_fire_brigade_request_status,
    update_fire_brigade_request,
]
fire_brigade_service_runnable = create_specialized_assistant(
    fire_brigade_service_system_message,
    fire_brigade_service_safe_tools,
    fire_brigade_service_sensitive_tools,
)



class ToAmbulanceServiceAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle ambulance service bookings."""

    pickup_location: str = Field(
        description="The pickup location for the ambulance service."
    )
    destination: str = Field(description="The destination for the ambulance service.")
    phone_number: str = Field(
        description="The phone number of the user requesting the ambulance service."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_location": "123 Main St, Springfield",
                "destination": "456 Elm St, Springfield",
                "phone_number": "123-456-7890",
            }
        }


class ToElectricityIssueAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle electricity-related issues."""

    issue_description: str = Field(
        description="A detailed description of the electricity issue the user is facing."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "issue_description": "The lights in my room are flickering and the power outlets are not working."
            }
        }


class ToWaterIssueAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle water-related issues."""

    issue_description: str = Field(
        description="A detailed description of the water issue the user is facing."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "issue_description": "The water pressure in my room is very low and there's no hot water."
            }
        }


class ToNewWaterConnection(BaseModel):
    """Transfers work to a specialized assistant to handle new water connection requests."""

    address: str = Field(
        description="The address where the new water connection is requested."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the new water connection."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "address": "123 Main St, Springfield",
                "request": "I need a new water connection for my newly constructed house.",
            }
        }


class ToNewElectricityConnection(BaseModel):
    """Transfers work to a specialized assistant to handle new electricity connection requests."""

    address: str = Field(
        description="The address where the new electricity connection is requested."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the new electricity connection."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "address": "456 Elm St, Springfield",
                "request": "I need a new electricity connection for my newly constructed house.",
            }
        }


class ToElectricityConnectionPolicyAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle electricity connection policy queries."""

    query: str = Field(
        description="The user's query related to electricity connection policies."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the requirements for a new electricity connection?"
            }
        }

class ToFireBrigadeServiceAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle fire brigade service requests."""

    location: str = Field(description="The location of the fire.")
    phone_number: str = Field(description="The phone number of the user requesting the fire brigade service.")
    additional_info: Optional[str] = Field(description="Any additional information provided by the user.", default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "location": "789 Pine St, Springfield",
                "phone_number": "987-654-3210",
                "additional_info": "The fire is spreading quickly and there are people trapped inside."
            }
        }


# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# for openai
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Context: You are the first point of contact for a state government’s citizen services bot on WhatsApp. Your role is to identify the user’s intent—whether they want to apply for a new electricity connection, a new water connection, raise a complaint, track an existing application, request emergency services, or have another query. Based on their response, you will route them to the appropriate specialized agent."
            "Objective: Efficiently determine the user's intent and route them to the right agent for their needs."
            "Style & Tone: Friendly, neutral, and efficient. Keep your responses between 20 to 35 words at max."
            "Audience: General public using WhatsApp to access government services."
            "Response Format: Simple questions to identify intent, followed by clear confirmation and transfer to the relevant agent."
            "Guidelines:"
            "- When using languages other than English, make sure to use colloquial, conversational, and everyday language; do not be overly formal, and use English words wherever necessary."
            "- Do not translate the following words from English; keep it as is: Connection, Service, REgistration, Application, Tracking, Complaint, Emergency, Fire Brigade."
            "Responsibilities:"
            "First check if you know the user's preferred language, if no, greet the user and introduce yourself using this prompt: नमस्ते, Citizen Services Helpline में आपका स्वागत है! कृपया मुझे अपनी पसंदीदा भाषा बताएं: हिंदी, इंग्लिश, मराठी, या कन्नडा. (Once the user responds with their preferred language, strictly continue the conversation in that language for the remainder of the conversation). If you do know the user's preferred language, ask how else can you help themin their preferred language."
            "Ask the user how you can assist them: if they need help with electricity connection, water connection, complaint registration, medical emergency services, or anything else."            "- Confirm the user's intent and seamlessly pass the conversation to the relevant agent."
            "If the query is outside your scope, retry for a clear intent, and if the user intent is still out of scope, end the conversation politely saying you cannot help the user at the moment.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


primary_assistant_tools = [
    # TavilySearchResults(max_results=1),
    # lookup_policy,
    set_language,
    # lookup_electricity_connection_policy,  # Added tool for handling electricity connection related queries
]

assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToElectricityIssueAssistant,
        ToWaterIssueAssistant,
        ToNewWaterConnection,
        ToNewElectricityConnection,
        ToElectricityConnectionPolicyAssistant,  # Added tool for handling electricity connection related queries
        ToAmbulanceServiceAssistant,  # Added tool for handling ambulance service bookings
        ToFireBrigadeServiceAssistant,  # Added tool for handling fire brigade service requests
    ]
)

"""#### Create Assistant

We're about ready to create the graph. In the previous section, we made the design decision to have a shared `messages` state between all the nodes. This is powerful in that each delegated assistant can see the entire user journey and have a shared context. This, however, means that weaker LLMs can easily get mixed up about there specific scope. To mark the "handoff" between the primary assistant and one of the delegated workflows (and complete the tool call from the router), we will add a `ToolMessage` to the state.


#### Utility

Create a function to make an "entry" node for each workflow, stating "the current assistant ix `assistant_name`".
"""

from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


"""#### Define Graph

Now it's time to start building our graph. As before, we'll start with a node to pre-populate the state with the user's current information.
"""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": get_user_information.invoke({})}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

"""Now we'll start adding our specialized workflows. Each mini-workflow looks very similar to our full graph in [Part 3](#part-3-conditional-interrupt), employing 5 nodes:

1. `enter_*`: use the `create_entry_node` utility you defined above to add a ToolMessage signaling that the new specialized assistant is at the helm
2. Assistant: the prompt + llm combo that takes in the current state and either uses a tool, asks a question of the user, or ends the workflow (return to the primary assistant)
3. `*_safe_tools`: "read-only" tools the assistant can use without user confirmation.
4. `*_sensitive_tools`: tools with "write" access that require user confirmation (and will be assigned an `interrupt_before` when we compile the graph)
5. `leave_skill`: _pop_ the `dialog_state` to signal that the *primary assistant* is back in control

Because of their similarities, we _could_ define a factory function to generate these. Since this is a tutorial, we'll define them each explicitly.

First, make the **flight booking assistant** dedicated to managing the user journey for updating and canceling flights.
"""


def add_assistant_workflow(
    builder,
    entry_node_name: str,
    entry_node_label: str,
    assistant_runnable: Runnable,
    safe_tools: list,
    sensitive_tools: list,
    dialog_state: str,
):
    builder.add_node(
        entry_node_name,
        create_entry_node(entry_node_label, dialog_state),
    )
    builder.add_node(dialog_state, Assistant(assistant_runnable))
    builder.add_edge(entry_node_name, dialog_state)
    builder.add_node(
        f"{dialog_state}_safe_tools",
        create_tool_node_with_fallback(safe_tools),
    )
    builder.add_node(
        f"{dialog_state}_sensitive_tools",
        create_tool_node_with_fallback(sensitive_tools),
    )

    def route_assistant(state: State):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        safe_toolnames = [t.name for t in safe_tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return f"{dialog_state}_safe_tools"
        return f"{dialog_state}_sensitive_tools"

    builder.add_edge(f"{dialog_state}_sensitive_tools", dialog_state)
    builder.add_edge(f"{dialog_state}_safe_tools", dialog_state)
    builder.add_conditional_edges(
        dialog_state,
        route_assistant,
        [
            f"{dialog_state}_safe_tools",
            f"{dialog_state}_sensitive_tools",
            "leave_skill",
            END,
        ],
    )


# Electricity issue assistant
add_assistant_workflow(
    builder,
    "enter_electricity_issue",
    "Electricity Issue Assistant",
    electricity_issue_runnable,
    electricity_issue_safe_tools,
    electricity_issue_sensitive_tools,
    "electricity_issue",
)

# Water issue assistant
add_assistant_workflow(
    builder,
    "enter_water_issue",
    "Water Issue Assistant",
    water_issue_runnable,
    water_issue_safe_tools,
    water_issue_sensitive_tools,
    "water_issue",
)

# New water connection assistant
add_assistant_workflow(
    builder,
    "enter_new_water_connection",
    "New Water Connection Assistant",
    new_water_connection_runnable,
    new_water_connection_safe_tools,
    new_water_connection_sensitive_tools,
    "new_water_connection",
)

# New electricity connection assistant
add_assistant_workflow(
    builder,
    "enter_new_electricity_connection",
    "New Electricity Connection Assistant",
    new_electricity_connection_runnable,
    new_electricity_connection_safe_tools,
    new_electricity_connection_sensitive_tools,
    "new_electricity_connection",
)

# Add the electricity connection policy assistant workflow
add_assistant_workflow(
    builder,
    "enter_electricity_connection_policy",
    "Electricity Connection Policy Assistant",
    electricity_connection_policy_runnable,
    electricity_connection_policy_safe_tools,
    electricity_connection_policy_sensitive_tools,
    "electricity_connection_policy",
)

# Add the ambulance service assistant workflow
add_assistant_workflow(
    builder,
    "enter_ambulance_service",
    "Ambulance Service Assistant",
    ambulance_service_runnable,
    ambulance_service_safe_tools,
    ambulance_service_sensitive_tools,
    "ambulance_service",
)

# Add the fire brigade service assistant workflow
add_assistant_workflow(
    builder,
    "enter_fire_brigade_service",
    "Fire Brigade Service Assistant",
    fire_brigade_service_runnable,
    fire_brigade_service_safe_tools,
    fire_brigade_service_sensitive_tools,
    "fire_brigade_service",
)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

"""Next, create the **car rental assistant** graph to own all car rental needs."""


# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToElectricityIssueAssistant.__name__:
            return "enter_electricity_issue"
        elif tool_calls[0]["name"] == ToWaterIssueAssistant.__name__:
            return "enter_water_issue"
        elif tool_calls[0]["name"] == ToNewWaterConnection.__name__:
            return "enter_new_water_connection"
        elif tool_calls[0]["name"] == ToNewElectricityConnection.__name__:
            return "enter_new_electricity_connection"
        elif tool_calls[0]["name"] == ToElectricityConnectionPolicyAssistant.__name__:
            return "enter_electricity_connection_policy"
        elif tool_calls[0]["name"] == ToAmbulanceServiceAssistant.__name__:
            return "enter_ambulance_service"
        elif tool_calls[0]["name"] == ToFireBrigadeServiceAssistant.__name__:
            return "enter_fire_brigade_service"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_electricity_issue",
        "enter_water_issue",
        "enter_new_water_connection",
        "enter_new_electricity_connection",
        "enter_electricity_connection_policy",  # Add this line
        "enter_ambulance_service",  # Add this line
        "enter_fire_brigade_service",  # Add this line for fire brigade service
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "electricity_issue",
    "water_issue",
    "new_water_connection",
    "new_electricity_connection",
    "electricity_connection_policy",
    "ambulance_service",
    "fire_brigade_service",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = MemorySaver()
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "ambulance_service_sensitive_tools",
        "new_electricity_connection_sensitive_tools",
    ],
)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


if os.getenv("APP_TYPE") == "terminal":
    # #####################################################################
    _printed = set()

    print("Starting conversation with the primary assistant.")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        events = part_4_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        snapshot = part_4_graph.get_state(config)
        while snapshot.next:
            # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
            # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
            # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changes.\n\n"
                )
            except:
                user_input = "y"
            if user_input.strip().lower() == "y":
                # Just continue
                result = part_4_graph.invoke(
                    None,
                    config,
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = part_4_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = part_4_graph.get_state(config)

    print("Conversation complete.")
    ###############################################################3


if os.getenv("APP_TYPE") == "api":

    class HumanMessage:
        def __init__(self, message):
            self.message = message

        def to_dict(self):
            return {"message": self.message}

    app = Flask(__name__)

    def get_last_response(responses):
        if not responses:
            return None

        # get the last response
        last_response = responses[-1]
        # print(f"last response: {last_response}")

        # get messages from the last response
        messages = last_response.get("messages")

        if messages and len(messages) > 1:
            ai_message = messages[-1]

            if isinstance(ai_message, AIMessage):
                return ai_message.content
        return None

    # Initialize a dictionary to store conversation states
    conversation_states = {}

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.json
        user_message = data.get("message")
        conversation_id = data.get("conversation_id")

        if not user_message:
            return jsonify({"error": "message is required"}), 400

        if not conversation_id or conversation_id.strip() == "":
            # Create a new conversation ID
            conversation_id = str(uuid.uuid4())
            config = {
                "configurable": {
                    "passenger_id": "3442 587242",
                    "thread_id": conversation_id,
                }
            }
            conversation_states[conversation_id] = config
        else:
            # Retrieve the existing conversation state
            config = conversation_states.get(conversation_id)
            if not config:
                return jsonify({"error": "Invalid conversation ID"}), 400

        _printed = set()
        events = part_4_graph.stream(
            {"messages": ("user", user_message)}, config, stream_mode="values"
        )
        responses = []
        for event in events:
            _print_event(event, _printed)
            responses.append(event)

        snapshot = part_4_graph.get_state(config)
        while snapshot.next:
            user_input = "y"  # Automatically approve actions for simplicity
            if user_input.strip().lower() == "y":
                result = part_4_graph.invoke(None, config)
            else:
                result = part_4_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = part_4_graph.get_state(config)

        bot_response = get_last_response(responses=responses)

        return jsonify({"conversation_id": conversation_id, "response": bot_response,"lang":"hi"})

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)
