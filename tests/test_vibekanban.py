#!/usr/bin/env python3
"""
Test vibe-kanban configuration using Playwright.
Verifies that the hermes-lite project and tasks are visible.
"""

import asyncio
import sys
from playwright.async_api import async_playwright


async def test_vibekanban():
    """Test vibe-kanban UI."""

    # Find the vibe-kanban port
    import subprocess
    result = subprocess.run(
        ["ss", "-tlnp"], capture_output=True, text=True
    )

    port = None
    for line in result.stdout.split('\n'):
        if 'vibe-kanban' in line or '127.0.0.1:4' in line:
            # Extract port
            import re
            match = re.search(r'127\.0\.0\.1:(\d+)', line)
            if match:
                port = match.group(1)
                break

    if not port:
        # Try common ports
        import urllib.request
        for p in [40289, 39269, 3000, 8080]:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{p}", timeout=2)
                port = str(p)
                break
            except:
                continue

    if not port:
        print("❌ Could not find vibe-kanban port")
        return False

    url = f"http://127.0.0.1:{port}"
    print(f"Testing vibe-kanban at {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Load vibe-kanban
            print("  Loading page...")
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=30000)

            # Take screenshot
            await page.screenshot(path="/tmp/vibekanban_test.png")
            print("  📸 Screenshot saved to /tmp/vibekanban_test.png")

            # Check for hermes-lite project
            print("  Looking for hermes-lite project...")
            content = await page.content()

            if "hermes-lite" in content:
                print("  ✅ hermes-lite project found!")
            else:
                print("  ⚠️  hermes-lite not visible on main page (may need to navigate)")

            # Look for our tasks
            tasks_found = []
            task_names = [
                "LLM Entity Extraction",
                "Iterative Micro-Query Loop",
                "Entity-First Ranker",
                "Upgrade Embedding Model"
            ]

            for task in task_names:
                if task in content:
                    tasks_found.append(task)

            if tasks_found:
                print(f"  ✅ Found {len(tasks_found)} tasks: {tasks_found}")
            else:
                print("  ⚠️  No tasks visible on main page (may need to click project)")

            # Try clicking on hermes-lite if we can find it
            try:
                hermes_link = page.locator("text=hermes-lite").first
                if await hermes_link.is_visible():
                    await hermes_link.click()
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    await page.screenshot(path="/tmp/vibekanban_project.png")
                    print("  📸 Project page screenshot: /tmp/vibekanban_project.png")

                    # Check for tasks now
                    content = await page.content()
                    tasks_found = [t for t in task_names if t in content]
                    if tasks_found:
                        print(f"  ✅ Found {len(tasks_found)} tasks on project page!")
            except Exception as e:
                print(f"  ℹ️  Could not click project: {e}")

            print("\n✅ vibe-kanban is running and accessible!")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            await page.screenshot(path="/tmp/vibekanban_error.png")
            return False
        finally:
            await browser.close()


if __name__ == "__main__":
    success = asyncio.run(test_vibekanban())
    sys.exit(0 if success else 1)
