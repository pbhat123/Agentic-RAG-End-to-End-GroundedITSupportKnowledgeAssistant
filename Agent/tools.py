"""Agent Tools for RAG-based Support Assistant"""

import json
import os
from typing import List, Dict, Any
from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Collection of tools for the support assistant agent.
class SupportTicketTools:
    def __init__(self, tickets_path: str = '../../data/synthetic_tickets.json'):
        # Load tickets
        with open(tickets_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)
        
        # Call Embedding model to create embeddings for vector DB storage
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        self._setup_vectorstore()

    # Create vector store for RAG
    def _setup_vectorstore(self):
        documents = []
        for ticket in self.tickets:
            content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
            doc = Document(
                page_content=content,
                metadata={
                    'ticket_id': ticket['ticket_id'],
                    'category': ticket['category'],
                    'priority': ticket['priority'],
                    'title': ticket['title']
                }
            )
            documents.append(doc)
        # Local persistence on disk for reuse
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./agent_vectorstore"
        )

    # Tool 1 - RAG Retrieval - Search similar support tickets
    def search_similar_tickets(self, query: str) -> str:
        results = self.vectorstore.similarity_search(query, k=3)
        if not results:
            return "No similar tickets found."
        output = "Found similar tickets:\n\n"
        for i, doc in enumerate(results, 1):
            output += f"--- Ticket {i} ---\n"
            output += doc.page_content + "\n\n"
        return output

    # Tool 2 - Ticket Lookup - Get specific ticket by ID
    def get_ticket_by_id(self, ticket_id: str) -> str:
        ticket_id = ticket_id.upper().strip()
        for ticket in self.tickets:
            if ticket['ticket_id'] == ticket_id:
                return f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Created: {ticket['created_date']}
Resolved: {ticket['resolved_date']}"""
        return f"Ticket {ticket_id} not found in the database."

    # Tool 3 - Category Search - Filter tickets by category
    def search_by_category(self, category: str) -> str:
        category = category.strip()
        matching = [t for t in self.tickets if t['category'].lower() == category.lower()]
        if not matching:
            available = list(set(t['category'] for t in self.tickets))
            return f"No tickets found in category '{category}'. Available categories: {', '.join(available)}"
        output = f"Found {len(matching)} tickets in category '{category}':\n\n"
        for ticket in matching:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} (Priority: {ticket['priority']})\n"
        return output

    # Tool 4 - Ticket Statistics - Get insights about ticket database
    def get_ticket_statistics(self, input: str = "") -> str:
        total = len(self.tickets)
        categories = {}
        priorities = {}
        for ticket in self.tickets:
            categories[ticket['category']] = categories.get(ticket['category'], 0) + 1
            priorities[ticket['priority']] = priorities.get(ticket['priority'], 0) + 1
        output = f"Ticket Database Statistics:\n"
        output += f"Total Tickets: {total}\n\n"
        output += "By Category:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {cat}: {count}\n"
        output += "\nBy Priority:\n"
        for pri, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {pri}: {count}\n"
        return output

    # Tool Mapping
    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Use this tool to search for similar support tickets based on a problem description or question.
Input should be a clear description of the issue or question.
This is the PRIMARY tool for answering "how to fix" or "similar issue" questions."""
            ),
            Tool(
                name="GetTicketByID",
                func=self.get_ticket_by_id,
                description="""Use this tool to retrieve details of a specific ticket by its ID.
Input should be a ticket ID like 'TICK-001'.
Use this when the user mentions a specific ticket number."""
            ),
            Tool(
                name="SearchByCategory",
                func=self.search_by_category,
                description="""Use this tool to find all tickets in a specific category.
Input should be a category name like 'Authentication', 'Database', 'Payment', etc.
Use this when the user wants to see all issues of a certain type."""
            ),
            Tool(
                name="GetTicketStatistics",
                func=self.get_ticket_statistics,
                description="""Use this tool to get statistics about the ticket database including total count, 
category distribution, and priority breakdown.
No input needed - just call this tool.
Use this when the user asks for statistics, summaries, or overview of tickets."""
            )
        ]
