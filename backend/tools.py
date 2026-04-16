"""
Custom tools for the Research Assistant Agent.

Uses Wikipedia for real information search.
SSL certificates are configured via system environment variables.
"""
import os
from datetime import datetime
from langchain_core.tools import tool
import wikipedia

# Configure Wikipedia with a proper user agent
wikipedia.set_user_agent('ResearchAssistant/1.0 (Learning LangChain/LangGraph)')

print("[OK] Wikipedia search configured")


@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about any topic.
    Use this for facts about people, places, events, technology, science, history, etc.
    """
    try:
        # Search and get summary
        summary = wikipedia.summary(query, sentences=5)
        return f"Wikipedia Summary for '{query}':\n\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return f"The query '{query}' is ambiguous. Did you mean one of: {options}?"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'. Try a different search term."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@tool
def get_wikipedia_page(title: str) -> str:
    """
    Get detailed content from a specific Wikipedia page.
    Use this when you need more information than the summary provides.
    """
    try:
        page = wikipedia.page(title)
        content = page.content[:3000]
        return f"Wikipedia Page: {page.title}\n\nURL: {page.url}\n\nContent:\n{content}..."
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return f"The title '{title}' is ambiguous. Options: {options}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found with title '{title}'."
    except Exception as e:
        return f"Error fetching page: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Examples: '2 + 2', '100 * 0.15', '2024 - 1956', 'pow(2, 10)'
    """
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "len": len
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    Use this when the user asks about the current time or date.
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


@tool
def word_count(text: str) -> str:
    """
    Count the number of words and characters in a text.
    """
    words = len(text.split())
    chars = len(text)
    return f"Text statistics: {words} words, {chars} characters"


# All tools available to the agent
ALL_TOOLS = [search_wikipedia, get_wikipedia_page, calculate, get_current_time, word_count]

print(f"[OK] Loaded {len(ALL_TOOLS)} tools: {', '.join(t.name for t in ALL_TOOLS)}")
