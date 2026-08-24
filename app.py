"""
Brain Dump Sanctuary - Day 2 Complete System
Includes: Streamlit UI + LangGraph Agent + Multi-Perspective Analysis

File structure:
braindump_sanctuary/
├── app.py (THIS FILE - run with: streamlit run app.py)
├── agents.py (LangGraph workflows)
├── braindump_core.py (from Day 1)
└── requirements.txt
"""

# ============== app.py - MAIN STREAMLIT APP ==============
import streamlit as st
import sys
from datetime import datetime
import pytz
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

# Import Day 1 components
from braindump_core import BrainDumpDB, EmbeddingEngine, ClusterEngine, create_knowledge_graph

# Import Day 2 components
from agents import QuestionAgent, SearchAgent, GenerationAgent, FeedAgent

# Page config
st.set_page_config(
    page_title="Brain Dump Sanctuary",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = BrainDumpDB()
if 'embedder' not in st.session_state:
    st.session_state.embedder = EmbeddingEngine()
if 'clusterer' not in st.session_state:
    st.session_state.clusterer = ClusterEngine(min_cluster_size=2)
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = SearchAgent()
if 'question_agent' not in st.session_state:
    st.session_state.question_agent = QuestionAgent()
if 'generation_agent' not in st.session_state:
    st.session_state.generation_agent = GenerationAgent()
if 'feed_agent' not in st.session_state:
    st.session_state.feed_agent = FeedAgent(search_agent=st.session_state.search_agent)

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("Transform stale lists into actionable curiosity")
    
    st.divider()
    
    # Tab Switcher
    tab_selection = st.radio(
        "Navigate",
        ["Home", "Feed"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Add new brain dump
    st.subheader("💭 New Brain Dump")
    
    # Use a unique key that changes when we clear
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    new_dump = st.text_area(
        "What's on your mind?",
        placeholder="Why do dreams feel so real?",
        height=100,
        label_visibility="collapsed",
        key=f"dump_input_{st.session_state.input_key}"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        add_clicked = st.button("Add to Sanctuary", type="primary", width='stretch')
    with col2:
        clear_clicked = st.button("Clear", width='stretch')
    
    if add_clicked:
        if new_dump.strip():
            if not st.session_state.get('adding', False):
                st.session_state.adding = True
                dump_id, is_duplicate = st.session_state.db.add_dump(new_dump.strip())
                st.session_state.input_key += 1
                if is_duplicate:
                    st.warning("This thought already exists in your sanctuary! 🔄")
                else:
                    st.success("Added! ✨")
                st.rerun()
            else:
                st.session_state.adding = False
        else:
            st.error("Can't add empty thought!")
    
    if clear_clicked:
        if not st.session_state.get('clearing_input', False):
            st.session_state.clearing_input = True
            st.session_state.input_key += 1
            st.rerun()
        else:
            st.session_state.clearing_input = False
    
    st.divider()
    
    # Actions
    st.subheader("⚙️ Actions")
    
    if st.button("🔄 Refresh Clusters", width='stretch'):
        st.info("Clusters will refresh on the Home tab")
    
    if st.button("🗑️ Clear All Dumps", width='stretch'):
        if not st.session_state.get('clearing', False):
            st.session_state.clearing = True
            # Clear all dumps and clusters from Neo4j
            with st.session_state.db.driver.session() as session:
                session.run("MATCH (d:Dump) DETACH DELETE d")
                session.run("MATCH (c:Cluster) DETACH DELETE c")
            st.session_state.input_key = 0
            st.success("All dumps cleared!")
            st.rerun()
        else:
            st.session_state.clearing = False
    
    # Stats
    st.divider()
    dumps = st.session_state.db.get_all_dumps()
    st.metric("Total Brain Dumps", len(dumps))

# ============== HELPER FUNCTIONS ==============

def render_brain_dump_table(dumps, cluster_labels_map):
    """Render brain dumps as a table with cluster labels and timestamps"""
    if not dumps:
        st.info("No brain dumps yet.")
        return
    
    # Prepare data for table
    table_data = []
    for dump_id, text, cluster_id, created_at in dumps:  # Already sorted by DESC in get_all_dumps()
        cluster_label = "Unclustered"
        if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
            cluster_label = cluster_labels_map[cluster_id]['label']
        
        # Format timestamp - handle Neo4j datetime objects and convert to IST
        if created_at:
            try:
                # Neo4j returns a neo4j.time.DateTime object
                # Convert to IST (Indian Standard Time: UTC+5:30)
                ist = pytz.timezone('Asia/Kolkata')
                
                # Parse the datetime string if needed
                if isinstance(created_at, str):
                    # Try to parse ISO format datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    # Assume it's a datetime object
                    dt = created_at
                
                # Make it timezone-aware if it isn't already
                if dt.tzinfo is None:
                    # Assume UTC if no timezone
                    dt = pytz.UTC.localize(dt)
                
                # Convert to IST
                dt_ist = dt.astimezone(ist)
                
                # Format as readable string
                created_at_str = dt_ist.strftime("%d %b %Y, %I:%M %p IST")
            except Exception as e:
                print(f"Timestamp conversion error: {e}")
                created_at_str = str(created_at) if created_at else "Unknown"
        else:
            created_at_str = "Unknown"
        
        table_data.append({
            "Brain Dump": text,
            "Cluster Label": cluster_label,
            "Created At": created_at_str
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, width='stretch', hide_index=True)


def render_feed_card(dump_id, text, cluster_id, cluster_labels_map):
    """Render a single blog-style card for a brain dump with agent outputs"""
    with st.container(border=True):
        # Title
        st.markdown(f"### 💭 {text}")
        
        # Cluster label badge
        if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
            cluster_label = cluster_labels_map[cluster_id]['label']
            st.markdown(f"🏷️ **{cluster_label}**")
        
        st.divider()
        
        # Check if we have cached feed data
        cached_data = st.session_state.db.get_feed_cache(dump_id)
        
        if cached_data:
            # Use cached summary and questions
            st.markdown("**Summary:**")
            st.write(cached_data['summary'])
            questions = cached_data['questions']
            image_urls = cached_data.get('image_urls', [])
            
            # If cached data exists but images are missing, generate them
            if not image_urls:
                with st.spinner("📸 Fetching related images..."):
                    try:
                        image_urls = st.session_state.feed_agent.search_images(text, max_results=3)
                        # Update cache with new image URLs
                        st.session_state.db.save_feed_cache(dump_id, cached_data['summary'], questions, image_urls)
                    except Exception as e:
                        print(f"Error searching for images: {str(e)}")
                        image_urls = []
        else:
            # Generate summary using Feed Agent
            with st.spinner("🧠 Generating Sonar summary and images..."):
                try:
                    result = st.session_state.feed_agent.generate_summary(text)
                    summary = result['summary']
                    st.markdown("**Summary:**")
                    st.write(summary)
                    
                    # Generate questions
                    questions = st.session_state.question_agent.generate_questions(text)
                    
                    # Search for related images
                    image_urls = st.session_state.feed_agent.search_images(text, max_results=3)
                    
                    # Cache summary, questions, and images
                    st.session_state.db.save_feed_cache(dump_id, summary, questions, image_urls)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    summary = "Unable to generate summary. Please try again."
                    questions = []
                    image_urls = []
        
        # Display images if available
        if image_urls:
            st.divider()
            st.markdown("**Related Images:**")
            cols = st.columns(min(3, len(image_urls)))  # Create up to 3 columns
            for idx, image_url in enumerate(image_urls[:3]):
                with cols[idx]:
                    try:
                        st.image(image_url, width='stretch')
                    except Exception as e:
                        st.caption(f"Could not load image: {image_url}")
        
        st.divider()
        
        # Display questions (either cached or just generated)
        st.markdown("**Questions for Reflection:**")
        if questions:
            for i, q in enumerate(questions, 1):
                st.markdown(f"{i}. {q}")
        else:
            st.info("No questions available for this brain dump.")


def render_home():
    """Render the Home tab with cluster map, text input, and brain dump table"""
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("*Where racing thoughts become structured curiosity*")
    
    dumps = st.session_state.db.get_all_dumps()
    
    if len(dumps) == 0:
        st.info("👋 Welcome! Add your first brain dump using the sidebar.")
        
        with st.expander("🎯 Try these example dumps"):
            examples = [
                "Why do dreams feel so real but fade so quickly?",
                "How does quantum entanglement actually work?",
                "Are LLMs actually understanding or just pattern matching?",
                "What causes the smell of rain on dry ground?",
                "Why does time feel faster as we age?",
            ]
            for ex in examples:
                if st.button(f"Add: {ex}", key=ex):
                    dump_id, is_duplicate = st.session_state.db.add_dump(ex)
                    st.rerun()
    else:
        # Cluster Map Section
        st.subheader("🗺️ Semantic Cluster Map")
        
        if len(dumps) < 3:
            st.warning("⚠️ Add at least 3 brain dumps to see meaningful clusters")
        else:
            try:
                col1, col2 = st.columns([4, 1])
                with col2:
                    force_refresh = st.button("🔄 Refresh", help="Recalculate embeddings and clusters")
                
                # Check if embeddings already exist for all dumps
                existing_embeddings = st.session_state.db.get_embeddings()
                existing_ids = {emb[0] for emb in existing_embeddings}
                all_dump_ids = {d[0] for d in dumps}
                
                # Check if cluster labels already exist
                existing_labels = st.session_state.db.get_all_cluster_labels()
                
                # Check if reducer exists (used for cached path)
                has_reducer = st.session_state.clusterer.reducer is not None
                
                # Only recalculation if we have new dumps without embeddings OR force refresh OR no reducer yet
                need_recalculation = force_refresh or not (existing_ids >= all_dump_ids) or not has_reducer
                
                if need_recalculation:
                    with st.spinner("Generating embeddings and clustering..."):
                        # Generate embeddings
                        texts = [d[1] for d in dumps]
                        embeddings = st.session_state.embedder.embed(texts)
                        
                        # Update embeddings in DB
                        for i, (dump_id, _, _, _) in enumerate(dumps):
                            st.session_state.db.update_embedding(dump_id, embeddings[i])
                        
                        # Cluster
                        clusters, coords_2d = st.session_state.clusterer.fit_predict(embeddings)
                        
                        # Track which clusters have changed
                        clusters_with_changes = set()
                        
                        # Check for cluster ID changes for each dump
                        for i, (dump_id, _, old_cluster_id, _) in enumerate(dumps):
                            new_cluster_id = clusters[i]
                            if old_cluster_id != new_cluster_id:
                                clusters_with_changes.add(new_cluster_id)
                                if old_cluster_id is not None and old_cluster_id != -1:
                                    clusters_with_changes.add(old_cluster_id)
                        
                        # Update clusters in DB
                        for i, (dump_id, _, _, _) in enumerate(dumps):
                            st.session_state.db.update_cluster(dump_id, clusters[i])
                        
                        # Auto-generate cluster labels for new clusters or changed clusters
                        cluster_labels_dict = {}
                        unique_clusters = set(clusters) - {-1}
                        
                        if unique_clusters:
                            with st.spinner("Generating cluster labels with Gemini..."):
                                for cluster_id in unique_clusters:
                                    # Re-label if cluster is new OR if it had changes
                                    if cluster_id not in existing_labels or cluster_id in clusters_with_changes:
                                        cluster_dumps = [dumps[i][1] for i in range(len(dumps)) if clusters[i] == cluster_id]
                                        label = st.session_state.clusterer.generate_cluster_label(cluster_dumps)
                                        cluster_labels_dict[cluster_id] = label
                                        st.session_state.db.save_cluster_label(cluster_id, label)
                                    else:
                                        cluster_labels_dict[cluster_id] = existing_labels[cluster_id]['label']
                        
                        # Generate and cache feed data for new/uncached dumps
                        with st.spinner("Generating feed summaries and questions..."):
                            for i, (dump_id, text, _, _) in enumerate(dumps):
                                # Check if feed cache already exists
                                if not st.session_state.db.get_feed_cache(dump_id):
                                    try:
                                        # Generate summary
                                        summary_result = st.session_state.feed_agent.generate_summary(text)
                                        summary = summary_result['summary']
                                        
                                        # Generate questions
                                        questions = st.session_state.question_agent.generate_questions(text)
                                        
                                        # Cache both
                                        st.session_state.db.save_feed_cache(dump_id, summary, questions)
                                    except Exception as e:
                                        print(f"Warning: Could not generate feed cache for {dump_id}: {e}")
                                        # Continue without caching for this dump
                else:
                    # Use existing embeddings and clusters
                    st.info("📦 Using cached embeddings and clusters")
                    
                    # Load existing embeddings
                    embeddings_list = []
                    for dump_id, _, _, _ in dumps:
                        matching_emb = next((emb[1] for emb in existing_embeddings if emb[0] == dump_id), None)
                        if matching_emb is not None:
                            embeddings_list.append(matching_emb)
                    
                    embeddings = np.array(embeddings_list)
                    
                    # Get clusters from database
                    clusters = np.array([d[2] if d[2] is not None else -1 for d in dumps])
                    
                    # Get 2D coordinates for visualization
                    coords_2d = st.session_state.clusterer.reducer.fit_transform(embeddings)
                    
                    # Load existing cluster labels
                    cluster_labels_dict = {k: v['label'] for k, v in existing_labels.items()}
                    unique_clusters = set(clusters) - {-1}
                
                # Visualize with labels
                fig = create_knowledge_graph(dumps, coords_2d, clusters, cluster_labels_dict, embeddings)
                st.plotly_chart(fig, width='stretch')
                        
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
                if st.checkbox("Show technical details"):
                    st.exception(e)
        
        st.divider()
        
        # Consolidated Cluster Generation Section
        st.subheader("🧬 Generate New Brain Dumps for All Clusters")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("*Generate one new braindump for each cluster in a single click*")
        
        with col2:
            if st.button("✨ Generate All", key="gen_all_clusters", help="Generate a new braindump for each cluster", type="primary"):
                try:
                    # Get all clusters with their dumps
                    all_cluster_labels = st.session_state.db.get_all_cluster_labels()
                    
                    if all_cluster_labels:
                        generated_count = 0
                        failed_clusters = []
                        
                        with st.spinner("🤖 Generating braindumps for all clusters..."):
                            generated_dumps = []  # Track generated dumps and their cluster IDs
                            
                            for cluster_id, cluster_info in all_cluster_labels.items():
                                cluster_label = cluster_info['label']
                                
                                # Skip clusters with None label
                                if cluster_label is None:
                                    failed_clusters.append(f"Cluster {cluster_id} (no label)")
                                    continue
                                
                                # Get dumps in this cluster
                                cluster_dumps = st.session_state.db.get_cluster_dumps(cluster_id)
                                if cluster_dumps:
                                    cluster_dump_texts = [d[1] for d in cluster_dumps]
                                    
                                    try:
                                        # Generate the braindump
                                        generated_text = st.session_state.generation_agent.generate_braindump(
                                            cluster_name=cluster_label,
                                            entries=cluster_dump_texts
                                        )
                                        
                                        # Check if there was an error
                                        if not generated_text.startswith("Error"):
                                            # Add to database with cluster assignment
                                            new_dump_id, is_duplicate = st.session_state.db.add_generated_dump(generated_text, cluster_id)
                                            
                                            if not is_duplicate:
                                                # Compute embedding for the generated dump immediately
                                                # so it doesn't trigger a full recalculation on rerun
                                                generated_embedding = st.session_state.embedder.embed([generated_text])[0]
                                                st.session_state.db.update_embedding(new_dump_id, generated_embedding)
                                                
                                                generated_dumps.append((new_dump_id, generated_text, cluster_id))
                                                generated_count += 1
                                            else:
                                                # Duplicate found, skip this generated dump
                                                pass
                                        else:
                                            failed_clusters.append(cluster_label)
                                    except Exception as e:
                                        print(f"Error generating for {cluster_label}: {e}")
                                        failed_clusters.append(cluster_label)
                        
                        # Show results
                        if generated_count > 0:
                            st.success(f"✨ Generated {generated_count} new brain dump{'s' if generated_count != 1 else ''}!")
                        
                        if failed_clusters:
                            st.warning(f"⚠️ Could not generate for: {', '.join(failed_clusters)}")
                        
                        if generated_count > 0:
                            st.rerun()
                    else:
                        st.info("💡 No clusters available yet.")
                except Exception as e:
                    st.error(f"Error generating braindumps: {str(e)}")
        
        st.divider()
        
        # Brain Dump Table Section
        st.subheader("📋 All Brain Dumps")
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        render_brain_dump_table(dumps, cluster_labels_map)


def render_feed():
    """Render the Feed tab with blog-style cards for 5 most recent brain dumps"""
    st.title("📰 Feed")
    st.markdown("*Agent-generated insights for your most recent thoughts*")
    
    dumps = st.session_state.db.get_all_dumps()
    
    if len(dumps) == 0:
        st.info("👋 No brain dumps yet. Add one using the sidebar to get started!")
    else:
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        
        # Initialize search mode state if not exists
        if 'search_mode' not in st.session_state:
            st.session_state.search_mode = False
        if 'selected_dump_id' not in st.session_state:
            st.session_state.selected_dump_id = None
        
        # Search and filter section
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search brain dumps",
                placeholder="Type to search...",
                key="feed_search",
                help="Search using fuzzy matching (handles typos)"
            )
        
        # Perform fuzzy matching using fuzzywuzzy
        suggestions = []
        if search_query.strip():
            # Calculate fuzzy match score for each dump
            for dump_id, text, cluster_id, created_at in dumps:
                # Use first 100 chars of text for matching
                match_text = text[:100]
                # Use token_sort_ratio for better matching with word order variations
                score = fuzz.token_sort_ratio(search_query.lower(), match_text.lower())
                suggestions.append({
                    'dump_id': dump_id,
                    'text': text,
                    'cluster_id': cluster_id,
                    'score': score,
                    'match_text': match_text
                })
            
            # Sort by score (higher = better match) and take top 8
            suggestions = sorted(suggestions, key=lambda x: x['score'], reverse=True)[:8]
            
            if suggestions:
                st.write(f"**Found {len(suggestions)} best matches:**")
                
                # Create dropdown options (truncated text)
                suggestion_texts = [
                    s['text'][:70] + "..." if len(s['text']) > 70 else s['text'] 
                    for s in suggestions
                ]
                
                # Show suggestions as a dropdown with match score
                selected_idx = st.selectbox(
                    "Select a brain dump",
                    range(len(suggestions)),
                    format_func=lambda i: f"({suggestions[i]['score']}%) {suggestion_texts[i]}",
                    key="feed_suggestions_dropdown",
                    help="Suggestions ranked by relevance (higher % = better match)"
                )
                
                if selected_idx is not None:
                    st.session_state.search_mode = True
                    st.session_state.selected_dump_id = suggestions[selected_idx]['dump_id']
            else:
                st.warning(f"No brain dumps found for '{search_query}'")
        else:
            # No search query - clear search mode
            st.session_state.search_mode = False
            st.session_state.selected_dump_id = None
        
        st.divider()
        
        # Display selected dump or recent dumps
        if st.session_state.search_mode and st.session_state.selected_dump_id:
            # Show single selected dump with back button
            col1, col2 = st.columns([4, 1])
            
            with col2:
                if st.button("← Back to Feed", key="back_to_feed"):
                    st.session_state.search_mode = False
                    st.session_state.selected_dump_id = None
                    st.rerun()
            
            # Find the selected dump
            selected_dump = next(
                (d for d in dumps if d[0] == st.session_state.selected_dump_id),
                None
            )
            
            if selected_dump:
                dump_id, text, cluster_id, created_at = selected_dump
                st.markdown("### Single Brain Dump View")
                render_feed_card(dump_id, text, cluster_id, cluster_labels_map)
        else:
            # Show 5 most recent dumps (default view)
            recent_dumps = dumps[:5]
            st.markdown(f"Showing **{len(recent_dumps)}** most recent brain dumps")
            st.divider()
            
            for dump_id, text, cluster_id, created_at in recent_dumps:
                render_feed_card(dump_id, text, cluster_id, cluster_labels_map)
                st.markdown("")  # Spacing between cards


# ============== MAIN APP ==============
# Main content
st.markdown("")  # Spacing

# Get all dumps
dumps = st.session_state.db.get_all_dumps()

# Render selected tab
if tab_selection == "Home":
    render_home()
else:
    render_feed()

# Footer
st.divider()
st.markdown("*Built with Google Gemini, Tavily Search, and Streamlit*")