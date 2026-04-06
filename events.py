"""
events.py — Event correlation for NarrativeScope.

Uses the Wikipedia API to fetch 3-5 background context/events related 
to the searched topic to overlay on the time-series chart.
"""

import wikipediaapi
import logging

logger = logging.getLogger(__name__)

def fetch_wikipedia_events(topic: str, max_events: int = 3) -> list:
    """
    Search Wikipedia for the given topic and extract a few key sentences 
    to act as 'events' for the timeline correlation visual.
    
    Returns a list of dictionaries with 'title', 'summary', and 'url'.
    """
    if not topic or len(topic) < 3:
        return []
        
    user_agent = "NarrativeScope/1.0 (Research Internship Project; +https://simppl.org/)"
    try:
        wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI, user_agent=user_agent)
    except TypeError:
        # Fallback for older versions of wikipediaapi
        wiki = wikipediaapi.Wikipedia('en')
        
    try:
        # Get the page for the topic
        page = wiki.page(topic)
        
        # If the page does not exist, we try a fallback or return empty
        if not page.exists():
            # If the search was too specific, try just the first major word
            fallback_topic = topic.split()[0]
            if len(fallback_topic) > 3:
                page = wiki.page(fallback_topic)
            if not page.exists():
                return []
                
        # We will treat section headings or paragraphs as 'events'
        events = []
        
        # Add the main summary as an overarching context Event
        main_summary = page.summary.split('. ')[0][:100] + "..." if page.summary else ""
        if main_summary:
            events.append({
                'title': page.title,
                'summary': main_summary,
                'url': page.fullurl,
                'type': 'Main Context'
            })
            
        # Try to pull dates or recent events from 'History' or 'In the news' style sections if possible
        # Since Wikipedia formatting varies wildly, we'll extract top-level sections as related entities
        count = 1
        for section in page.sections:
            if section.title.lower() in ['see also', 'references', 'external links', 'citations']:
                continue
            section_summary = section.text.split('. ')[0][:100] + "..." if section.text else ""
            if section_summary and len(section_summary) > 20:
                events.append({
                    'title': f"{page.title} - {section.title}",
                    'summary': section_summary,
                    'url': f"{page.fullurl}#{section.title.replace(' ', '_')}",
                    'type': 'Related Event'
                })
                count += 1
                if count >= max_events:
                    break
        
        return events
    except Exception as e:
        logger.error(f"Error fetching Wikipedia events for '{topic}': {e}")
        return []
