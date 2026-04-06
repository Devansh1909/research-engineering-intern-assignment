import asyncio
from playwright.async_api import async_playwright
import os
import time

async def capture_screenshots():
    os.makedirs('screenshots', exist_ok=True)
    print("Navigating to dashboard...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use a high-res viewport for good screenshots
        context = await browser.new_context(viewport={'width': 1440, 'height': 1080})
        page = await context.new_page()
        
        # Go to app
        await page.goto("http://localhost:8501")
        # Wait for Streamlit to load completely
        await page.wait_for_selector('text=NarrativeScope', timeout=15000)
        # Give it a few extra seconds to render the UI components completely
        await page.wait_for_timeout(3000)
        
        print("Capturing hero.png...")
        await page.screenshot(path="screenshots/hero.png")
        
        print("Capturing search.png...")
        # Scroll down to search
        await page.evaluate("window.scrollBy(0, 500)")
        await page.wait_for_timeout(1000)
        await page.screenshot(path="screenshots/search.png")
        
        print("Searching for a narrative...")
        # Type into the search box
        await page.fill("input[aria-label='What narrative are you investigating?']", "information manipulation")
        await page.press("input[aria-label='What narrative are you investigating?']", "Enter")
        
        # Wait for results to load
        print("Waiting for search results...")
        await page.wait_for_timeout(8000)
        
        print("Capturing timeseries.png...")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.35)")
        await page.wait_for_timeout(2000)
        await page.screenshot(path="screenshots/timeseries.png")
        
        print("Capturing clusters.png...")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.55)")
        await page.wait_for_timeout(2000)
        await page.screenshot(path="screenshots/clusters.png")
        
        print("Capturing network.png...")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.75)")
        await page.wait_for_timeout(2000)
        await page.screenshot(path="screenshots/network.png")
        
        print("Capturing live_mode.png...")
        # Open the sidebar
        try:
            await page.click("[data-testid='collapsedControl']", timeout=2000)
            await page.wait_for_timeout(1000)
        except:
            pass # Sidebar might already be open
        
        await page.screenshot(path="screenshots/live_mode.png")
        
        await browser.close()
        print("Successfully captured all screenshots!")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
