"""
agents.py (Day 2)

Contains the agentic logic for the Brain Dump Sanctuary.
This file provides the classes that app.py imports.

- QuestionAgent: Socratic question generation
- PerspectiveAgent: Multi-angle analysis
- SearchAgent: Web search + synthesis (MOCKED for Day 2)

This file uses simple, direct LLM calls for the Day 2 demo.
It is designed to be plug-and-play with app.py.

Future Work: These classes can be refactored into nodes
in a more complex LangGraph orchestration graph.
"""

import os
import json
import requests

# We'll use Google Gemini as the LLM (modern google-genai SDK)
from google import genai
from google.genai import types

from dotenv import load_dotenv

# Tavily for web search
from tavily import TavilyClient

# --- API Key Configuration ---
# Get your free key from: https://aistudio.google.com/app/apikey
try:
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception:
    API_KEY = None

def get_genai_client():
    """Helper to get or create a google.genai.Client instance."""
    api_key = os.getenv("GOOGLE_API_KEY") or API_KEY
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing GenAI Client: {e}")
        return None

if not API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. Please set it in your environment.")
    API_KEY = "YOUR_API_KEY_HERE"

# --- Tavily API Key Configuration ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not found. SearchAgent will use mock data.")
    print("Get your API key from: https://app.tavily.com")
    TAVILY_API_KEY = None

# --- Perplexity API Key Configuration ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    print("ℹ️ PERPLEXITY_API_KEY not found. Will use Tavily or mock data.")
    print("Get your API key from: https://www.perplexity.ai/")
    PERPLEXITY_API_KEY = None


# ============== 1. Socratic Question Agent ==============

class QuestionAgent:
    """
    Generates Socratic questions to expand on a vague idea.
    """
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        self.system_prompt = """
You are a Socratic tutor. A user has a vague brain dump idea. 
Your goal is to generate 5 insightful, open-ended questions to help them 
explore this idea, discover its core, and understand their own curiosity.

- Do not answer the questions.
- Provide ONLY the list of questions.
- Return the questions as a JSON list of strings.

Example:
User Idea: "Why do we forget things we just read?"
Your Response:
[
    "What kind of material do you find you forget most often?",
    "What is your state of mind when you are reading?",
    "Are you trying to memorize facts, or understand a concept?",
    "What is the difference between remembering a fact and understanding an idea?",
    "How does this relate to the 'forgetting curve'?"
]
"""

    def generate_questions(self, idea: str) -> list[str]:
        """Generates Socratic questions for a given idea."""
        client = get_genai_client()
        if not client:
            return ["Error: GOOGLE_API_KEY is not set.", "Please add it to your environment."]

        try:
            # Combine system prompt and user input
            prompt = f"{self.system_prompt}\n\nUser Idea: \"{idea}\""
            
            # Generate response using modern google-genai SDK
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            questions_json_string = response.text
            questions = json.loads(questions_json_string)
            
            # The LLM might return a dict {"questions": [...]}, or just [...]
            if isinstance(questions, dict):
                # Try to find the list value
                for key, value in questions.items():
                    if isinstance(value, list):
                        return value
            elif isinstance(questions, list):
                return questions
            
            return ["Error: Could not parse questions from LLM response."]

        except Exception as e:
            print(f"Error in QuestionAgent: {e}")
            return [
                "An error occurred while generating questions.",
                "Is your API key set correctly?",
                f"Details: {e}"
            ]


# ============== 2. Multi-Perspective Agent ==============

class PerspectiveAgent:
    """
    Analyzes a controversial topic from multiple angles.
    Can optionally use web search results for context.
    """
    def __init__(self, model="gemini-2.5-flash", search_agent=None):
        self.model = model
        self.search_agent = search_agent
        self.system_prompt = """
You are a multi-perspective analyst. A user has a topic,
which may be controversial or nuanced.
Your goal is to provide three distinct, concise viewpoints:
1.  **Skeptical View:** The critical or cautious perspective.
2.  **Optimistic View:** The positive or enthusiastic perspective.
3.  **Nuanced View:** A balanced, synthetic view that includes trade-offs or a "third way".

You MUST return a JSON object with exactly three keys:
"skeptical", "optimistic", and "nuanced".
Each value should be a short paragraph (2-4 sentences).
"""

    def analyze(self, topic: str) -> dict[str, str]:
        """Analyzes a topic from multiple perspectives with optional web context."""
        client = get_genai_client()
        if not client:
            return {
                "skeptical": "Error: GOOGLE_API_KEY not set.",
                "optimistic": "Please add your API key to environment.",
                "nuanced": "The agent cannot run without an API key."
            }

        try:
            # Get web context if search_agent is available
            web_context = ""
            sources = []
            if self.search_agent and self.search_agent.use_real_search:
                print(f"🌐 Fetching web context for multi-perspective analysis...")
                articles = self.search_agent.get_relevant_articles(topic)
                if articles:
                    web_context = "\n\nRelevant Web Context:\n"
                    for i, article in enumerate(articles, 1):
                        web_context += f"{i}. {article['title']}: {article['snippet']}\n"
                        sources.append(article)
            
            # Combine system prompt and user input with optional web context
            prompt = f"{self.system_prompt}\n\nTopic: \"{topic}\"{web_context}"
            
            # Generate response using modern google-genai SDK
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            perspectives_json_string = response.text
            perspectives = json.loads(perspectives_json_string)
            
            # Ensure the keys are always present to prevent errors in app.py
            perspectives.setdefault('skeptical', 'No skeptical view generated.')
            perspectives.setdefault('optimistic', 'No optimistic view generated.')
            perspectives.setdefault('nuanced', 'No nuanced view generated.')
            
            # Add sources if available
            if sources:
                perspectives['sources'] = sources
            
            return perspectives
            
        except Exception as e:
            print(f"Error in PerspectiveAgent: {e}")
            return {
                "skeptical": f"An error occurred: {e}",
                "optimistic": "Please check your API key and model access.",
                "nuanced": "The PerspectiveAgent failed to run."
            }


# ============== 3. Web Search Agent ==============

class SearchAgent:
    """
    Real web search using Perplexity Sonar (preferred) or Tavily API + Gemini synthesis.
    Falls back to mock data if API keys are missing.
    """
    def __init__(self):
        self.use_perplexity = bool(PERPLEXITY_API_KEY)
        self.use_tavily = bool(TAVILY_API_KEY) and not self.use_perplexity
        self.use_real_search = self.use_perplexity or self.use_tavily
        
        if self.use_perplexity:
            self.perplexity_api_key = PERPLEXITY_API_KEY
            print("✅ Initialized SearchAgent with Perplexity Sonar API")
        elif self.use_tavily:
            self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
            print("✅ Initialized SearchAgent with Tavily API")
        else:
            print("⚠️ Initialized MOCK SearchAgent (no Perplexity or Tavily key)")
    
    def deep_dive(self, topic: str) -> dict:
        """
        Searches web using Perplexity Sonar or Tavily, then synthesizes.
        Falls back to mock if no API key.
        """
        if not self.use_real_search:
            return self._mock_search(topic)
        
        if self.use_perplexity:
            return self._perplexity_search(topic)
        else:
            return self._tavily_search(topic)
    
    def _perplexity_search(self, topic: str) -> dict:
        """Search using Perplexity Sonar API"""
        try:
            print(f"🔍 Searching Perplexity Sonar for: {topic}")
            
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Provide comprehensive, well-sourced answers."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide a comprehensive summary about: {topic}"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            summary = data.get('choices', [{}])[0].get('message', {}).get('content', 'No summary generated')
            
            # Extract citations if available
            sources = []
            if 'citations' in data:
                sources = [{"title": f"Source {i+1}", "url": "#", "snippet": cite} 
                          for i, cite in enumerate(data['citations'][:5])]
            
            return {
                "summary": summary,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Perplexity search failed: {e}")
            return {
                "summary": f"Search failed: {str(e)}. Please check your Perplexity API key.",
                "sources": []
            }
    
    def _tavily_search(self, topic: str) -> dict:
        """
        Searches web using Tavily, then synthesizes with Gemini.
        """
        try:
            # Step 1: Search with Tavily
            print(f"🔍 Searching Tavily for: {topic}")
            search_results = self.tavily.search(
                query=topic,
                max_results=5,
                search_depth="advanced"
            )
            
            # Step 2: Extract sources
            sources = []
            context = ""
            for result in search_results.get('results', []):
                sources.append({
                    'title': result.get('title', 'Untitled'),
                    'url': result.get('url', '#'),
                    'snippet': result.get('content', '')[:200] + "..."
                })
                context += f"\n\n{result.get('content', '')}"
            
            # Step 3: Synthesize with Gemini
            print(f"🤖 Synthesizing with Gemini...")
            synthesis_prompt = f"""You are a research synthesizer. Based on the following web search results about "{topic}", 
write a comprehensive 3-4 paragraph summary that:
1. Answers the core question
2. Highlights key findings and debates
3. Cites different viewpoints if applicable

Web Search Results:
{context[:4000]}

Provide ONLY the summary text, no preamble."""

            client = get_genai_client()
            if client:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=synthesis_prompt
                )
                summary = response.text
            else:
                summary = "Error: GOOGLE_API_KEY is not configured for Gemini synthesis."
            
            return {
                "summary": summary,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Tavily search failed: {e}")
            return {
                "summary": f"Search failed: {str(e)}. Please check your Tavily API key.",
                "sources": []
            }
    
    def get_relevant_articles(self, topic: str, max_results: int = 3) -> list[dict]:
        """
        Gets top articles for a topic (used by PerspectiveAgent).
        Returns: [{'title': str, 'url': str, 'snippet': str}, ...]
        """
        if not self.use_real_search:
            return []
        
        try:
            if self.use_perplexity:
                print(f"📰 Fetching articles with Perplexity for: {topic}")
                # Perplexity doesn't have a dedicated article fetch, use deep_dive results
                result = self._perplexity_search(topic)
                return result.get('sources', [])[:max_results]
            else:
                print(f"📰 Fetching articles for: {topic}")
                search_results = self.tavily.search(
                    query=topic,
                    max_results=max_results,
                    search_depth="basic"
                )
                
                articles = []
                for result in search_results.get('results', []):
                    articles.append({
                        'title': result.get('title', 'Untitled'),
                        'url': result.get('url', '#'),
                        'snippet': result.get('content', '')[:300] + "..."
                    })
                
                return articles
            
        except Exception as e:
            print(f"❌ Article fetch failed: {e}")
            return []
    
    def _mock_search(self, topic: str) -> dict:
        """Fallback mock implementation"""
        print(f"🔍 MOCK SEARCH: Deep dive for '{topic}' (using hardcoded data)")
        
        # Simulate different responses for different topics
        if "dream" in topic.lower():
            return {
                "summary": (
                    "Dreams are a complex neurological phenomenon, primarily occurring during "
                    "REM sleep. Research suggests they are crucial for memory consolidation, "
                    "emotional regulation, and problem-solving. The 'fading' "
                    "is attributed to the brain's different neurochemical state during sleep, "
                    "which is not optimized for encoding new memories."
                ),
                "sources": [
                    {"title": "The Science of Dreaming - Scientific American", "url": "https://www.scientificamerican.com/article/the-science-of-dreaming/"},
                    {"title": "Why We Dream - Psychology Today", "url": "https://www.psychologytoday.com/us/basics/dreaming"}
                ]
            }
        elif "llm" in topic.lower():
            return {
                "summary": (
                    "The debate on LLM 'understanding' is central to AI research. "
                    "One view holds they are 'stochastic parrots,' brilliantly matching "
                    "statistical patterns without true comprehension. "
                    "The opposing view suggests that at their scale, these models "
                    "develop emergent, internal world models, representing a new "
                    "form of understanding."
                ),
                "sources": [
                    {"title": "On the Dangers of Stochastic Parrots - FAccT '21", "url": "https://dl.acm.org/doi/10.1145/3442188.3445922"},
                    {"title": "Sparks of AGI: Early experiments with GPT-4", "url": "https://arxiv.org/abs/2303.12712"}
                ]
            }
        else:
            return {
                "summary": (
                    f"This is a mock summary about '{topic}'. This agent successfully simulated a "
                    "web search. In a real application, this text would be "
                    "dynamically generated by an LLM based on live search results "
                    "from a tool like Tavily or SerpAPI."
                ),
                "sources": [
                    {"title": "Mock Source 1 - Wikipedia", "url": "https://en.wikipedia.org/wiki/Main_Page"},
                    {"title": "Mock Source 2 - Example.com", "url": "https://example.com"}
                ]
            }

# --- How to refactor to LangGraph (Future Work) ---
"""
To implement your full "LangGraph" vision, you would:

1.  Define a State:
    class BrainDumpState(TypedDict):
        topic: str
        socratic_questions: list[str]
        perspectives: dict
        deep_dive: dict

2.  Create Nodes:
    - Each agent's method (e.g., `generate_questions`) becomes a node.
    - def question_node(state: BrainDumpState):
    -   questions = QuestionAgent().generate_questions(state['topic'])
    -   return {"socratic_questions": questions}
    - ... (similar nodes for perspective_node and search_node)

3.  Build the Graph:
    - workflow = StateGraph(BrainDumpState)
    - workflow.add_node("socratic", question_node)
    - workflow.add_node("perspectives", perspective_node)
    - workflow.add_node("search", search_node)
    - workflow.set_entry_point("socratic") # or a router
    - workflow.add_edge("socratic", "perspectives")
    - workflow.add_edge("perspectives", "search")
    - workflow.add_edge("search", END)

4.  Compile & Run:
    - app = workflow.compile()
    - # In app.py, you'd call:
    - # results = app.invoke({"topic": "Why do we dream?"})
    - # And then display results['socratic_questions'], etc.
    
This simple class-based approach is used for the Day 2 demo
as it directly matches your existing app.py implementation.
"""


# ============== 4. Feed Agent (using Perplexity Sonar) ==============

class FeedAgent:
    """
    Generates AI-powered summaries for feed cards using Perplexity Sonar.
    If Perplexity is not available, falls back to SearchAgent.
    Also fetches relevant images using Tavily image search.
    """
    def __init__(self, search_agent=None):
        self.search_agent = search_agent
        self.use_perplexity = bool(PERPLEXITY_API_KEY)
        self.perplexity_api_key = PERPLEXITY_API_KEY
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        
        if self.use_perplexity:
            print("✅ Initialized FeedAgent with Perplexity Sonar")
        else:
            print("ℹ️ FeedAgent will use SearchAgent for summaries")
        
        if self.tavily_client:
            print("📸 Initialized FeedAgent with Tavily Image Search")
        else:
            print("⚠️ Tavily API key not configured. Image search disabled.")
    
    def generate_summary(self, topic: str) -> dict:
        """
        Generate a comprehensive summary using Perplexity Sonar.
        Returns: {"summary": str, "sources": list}
        """
        if self.use_perplexity:
            return self._perplexity_summary(topic)
        elif self.search_agent:
            return self.search_agent.deep_dive(topic)
        else:
            return {
                "summary": "Unable to generate summary. Please configure Perplexity or Tavily API.",
                "sources": []
            }
    
    def _perplexity_summary(self, topic: str) -> dict:
        """Generate summary using Perplexity Sonar API"""
        try:
            print(f"🧠 Generating Sonar summary for: {topic}")
            
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a brilliant synthesizer of knowledge. Given a brain dump topic, 
provide a comprehensive yet concise summary that:
1. Explains the core concept clearly
2. Provides practical insights
3. Connects to broader contexts
4. Sparks further curiosity

Keep the tone engaging and thought-provoking."""
                    },
                    {
                        "role": "user",
                        "content": f"Provide a comprehensive summary about this brain dump topic: {topic}"
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.7,
                "top_p": 0.9,
                "search_domain_filter": ["perplexity.com"]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            summary = data.get('choices', [{}])[0].get('message', {}).get('content', 'No summary generated')
            
            # Extract citations if available
            sources = []
            if 'citations' in data:
                sources = [{"title": f"Source {i+1}", "url": "#", "snippet": cite} 
                          for i, cite in enumerate(data['citations'][:3])]
            
            return {
                "summary": summary,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Perplexity summary generation failed: {e}")
            # Fallback to SearchAgent if available
            if self.search_agent:
                print("Falling back to SearchAgent...")
                return self.search_agent.deep_dive(topic)
            
            return {
                "summary": f"Error generating summary: {str(e)}",
                "sources": []
            }
    
    def search_images(self, topic: str, max_results: int = 3) -> list[str]:
        """
        Search for relevant images using Tavily image search API.
        
        Args:
            topic: The topic to search for images
            max_results: Maximum number of images to return (default: 3)
            
        Returns:
            List of image URLs (up to max_results)
        """
        if not self.tavily_client:
            print("⚠️ Tavily API key not configured. Cannot search for images.")
            return []
        
        try:
            print(f"📸 Searching for images related to: {topic}")
            
            # Use Tavily to search for images
            response = self.tavily_client.search(
                query=topic,
                max_results=max_results,
                include_images=True
            )
            
            # Extract image URLs from results
            image_urls = []
            
            # Try to extract images from the response
            if 'images' in response:
                for img in response['images'][:max_results]:
                    if isinstance(img, dict) and 'url' in img:
                        image_urls.append(img['url'])
                    elif isinstance(img, str):
                        image_urls.append(img)
            
            # If no images found in dedicated images field, try results
            if not image_urls and 'results' in response:
                for result in response['results'][:max_results]:
                    if isinstance(result, dict) and 'image' in result:
                        image_urls.append(result['image'])
            
            print(f"✓ Found {len(image_urls)} relevant images")
            return image_urls[:max_results]
            
        except Exception as e:
            print(f"❌ Image search failed: {e}")
            return []


# ============== 5. Brain Dump Generation Agent ==============

class GenerationAgent:
    """
    Generates creative braindumps based on a cluster's theme and entries.
    Takes a cluster name and list of entries, uses Gemini to synthesize
    a new braindump that fits the cluster and would be interesting to read.
    """
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        self.system_prompt = """
You are a creative brainstorming agent. You've been given a cluster of related thoughts/brain dumps,
along with the cluster's theme.

Your task is to generate ONE new, engaging brain dump entry that:
1. Fits naturally with the theme and existing entries
2. Is inspired by the existing entries but presents a NEW angle or question
3. Is concise (1-2 sentences), matching the style of the existing entries
4. Introduces something the user might find interesting to explore
5. Does NOT simply repeat or combine existing entries

Generate ONLY the new brain dump text itself - no preamble, no explanation.
Just the thoughtful question or observation that belongs in this cluster.
"""

    def generate_braindump(self, cluster_name: str, entries: list[str]) -> str:
        """
        Generates a new braindump for a cluster.
        
        Args:
            cluster_name: The name/label of the cluster (e.g., "Dreams and Consciousness")
            entries: List of existing brain dump texts in this cluster
            
        Returns:
            Generated brain dump text as a string
        """
        client = get_genai_client()
        if not client:
            return "Error: GOOGLE_API_KEY is not set. Please add it to your environment."

        try:
            # Format entries for the prompt
            entries_str = "\n".join([f"- {entry}" for entry in entries])
            
            # Build the prompt
            prompt = f"""{self.system_prompt}

Cluster Theme: "{cluster_name}"

Existing entries in this cluster:
{entries_str}

Generate a new, creative brain dump entry that fits this cluster:"""
            
            # Generate response using modern google-genai SDK
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8
                )
            )
            generated_text = response.text.strip()
            
            # Clean up any extra quotes or markers
            if generated_text.startswith('"') and generated_text.endswith('"'):
                generated_text = generated_text[1:-1]
            
            return generated_text
            
        except Exception as e:
            print(f"Error in GenerationAgent: {e}")
            return f"Error generating braindump: {str(e)}"