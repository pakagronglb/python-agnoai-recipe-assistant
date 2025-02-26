import json
from textwrap import dedent
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.models.openai import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

class SearchResult(BaseModel):
    title: str = Field(..., title='Title of the search result')
    snippet: str = Field(..., title='Snippet of the search result')
    link: str = Field(..., title='Link to the search result')

class SearchResults(BaseModel):
    results: list[SearchResult] = Field(..., title='List of search results')

def get_recipe(url: str) -> str:
    """
    Fetches the content of a webpage and returns its text content.
    Args:
        url (str): The URL of the webpage to fetch.
    Returns:
        str: The text content of the webpage, with newlines, carriage returns, 
             tabs, and multiple spaces replaced by single spaces.
    Raises:
        Exception: If there is an error during the HTTP request or parsing.
    """    
    try:
        response = httpx.get(url, headers={'user-agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('    ', '')
    except Exception as e:
        return str(e)
    
def search_google(
        query: str, 
        date_restrict: str = None,
        exact_terms: str = None,
        exclude_terms: str = None,
        link_site: str = None,
        site_search: str = None,
        num: int = 10,
        start: int = 1,
    ) -> list[dict]:
    """  
    Google Custom Search API

    Args:
        query (str): search query
        date_restrict (str, optional): Restricts results to URLs based on date. Supported values include:
            d[number]: requests results from the specified number of past days.
            w[number]: requests results from the specified number of past weeks.
            m[number]: requests results from the specified number of past months.
            y[number]: requests results from the specified number of past years.
        exact_terms (str, optional): Identifies a phrase that all documents in the search results must contain.
        exclude_terms (str, optional): Identifies a word or phrase that should not appear in any documents in the search results.
        link_site (str, optional): Specifies that all search results should contain a link to a particular URL.
        site_search (str, optional): Specifies all search results should be pages from a given site.
        num (int, optional): Number of search results to return. Default is 10. Maximum is 10.
        start (int, optional): The index of the first result to return. Default is 1.
            
    Returns:
        list[dict]: list of search results

    """
    API_KEY = os.getenv('custom_search_api_key')
    CX_ID = os.getenv('search_engine_id')
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': CX_ID,
        'q': query,
        'num': num,
        'start': start,
        'dateRestrict': date_restrict,
        'exactTerms': exact_terms,
        'excludeTerms': exclude_terms,
        'linkSite': link_site,
        'siteSearch': site_search,
    }
    response = httpx.get(url, params=params)
    if response.status_code == 200:
        # return response.json()
        json_results = response.json()

        lst = []
        for item in json_results.get('items', []):
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            lst.append(SearchResult(title=title, snippet=snippet, link=link))
        return SearchResults(results=lst).model_dump_json()
    return response.text

def setup_storage(table_name: str, target_dir: str = None) -> SqliteAgentStorage:
    """ Set Agno SQLite agent storage """
    if target_dir is None:
        target_dir = './agent_sessions.db'

    storage = SqliteAgentStorage(
        db_file=target_dir,
        table_name=table_name
    )
    return storage

def recipe_agent() -> Agent:
    agent_storage = setup_storage(table_name='recipe_agent_sessions')

    agent = Agent(
        name='Recipe Assistant Agent',
        model=OpenAIChat(id='gpt-4o-mini'),
        storage=agent_storage,
        description=dedent("""
            You are an agent that is created to help user with recipes from the web.
                        
            If user ask questions not related to recipes, simply reply "I'm unable to help with that". Ask for a different langauge is okay
                        
            1. Use search_google function to find recipes page from the web then present the user with the options (name and URL)               
            2. Ask user to choose one of the options or ask for more options.
            3. Use get_recipe function to get the recipe from the URL and present it to the user.
        """),    
        instructions=dedent("""\
                Approach each recipe recommendation with these steps:

                1. Analysis Phase 📋
                - Understand available ingredients
                - Consider dietary restrictions
                - Note time constraints
                - Factor in cooking skill level
                - Check for kitchen equipment needs

                2. Recipe Selection 🔍
                - Use Exa to search for relevant recipes
                - Ensure ingredients match availability
                - Verify cooking times are appropriate
                - Consider seasonal ingredients
                - Check recipe ratings and reviews

                3. Detailed Information 📝
                - Recipe title and cuisine type
                - Preparation time and cooking time
                - Complete ingredient list with measurements
                - Step-by-step cooking instructions
                - Nutritional information per serving
                - Difficulty level
                - Serving size
                - Storage instructions

                4. Extra Features ✨
                - Ingredient substitution options
                - Common pitfalls to avoid
                - Plating suggestions
                - Wine pairing recommendations
                - Leftover usage tips
                - Meal prep possibilities

                Presentation Style:
                - Use clear markdown formatting
                - Present ingredients in a structured list
                - Number cooking steps clearly
                - Add emoji indicators for:
                🌱 Vegetarian
                🌿 Vegan
                🌾 Gluten-free
                🥜 Contains nuts
                ⏱️ Quick recipes
                - Include tips for scaling portions
                - Note allergen warnings
                - Highlight make-ahead steps
                - Suggest side dish pairings"""
        ),
        tools=[search_google, get_recipe],
        add_history_to_messages=True,
        debug_mode=False,
        show_tool_calls=True
    )
    return agent
