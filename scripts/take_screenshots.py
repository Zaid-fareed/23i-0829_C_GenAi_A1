"""Drive the running web app in a real browser (Microsoft Edge via Playwright), run every workspace on a demo image,
save screenshots for the report and print layout measurements (is the Run button on screen? does the page scroll?
how big are the image tiles?).

  pip install playwright            # uses the Edge already installed on Windows (no browser download)
  python scripts/take_screenshots.py --base http://localhost:5173 --out report/figures --size 1366x768
"""
import argparse
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:3000")
ap.add_argument("--out", default="report/figures")
ap.add_argument("--size", default="1366x768")
ap.add_argument("--prefix", default="app_")
ap.add_argument("--only", default="", help="comma-separated workspace keys to capture")
a = ap.parse_args()
W, H = (int(v) for v in a.size.split("x"))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

WORKSPACES = [  # key, sidebar label, sample thumbnail, run button text
    ("universal", "Universal Restoration", "pet_Maine_Coon_2.jpg", "Run model"),
    ("hard", "Hard-Routed Restoration", "pet_Maine_Coon_2.jpg", "Run model"),
    ("soft", "Soft Mixture-of-Experts", "pet_Maine_Coon_2.jpg", "Run model"),
    ("sketch", "Face-to-Sketch Generator", "face_image1396_style1.jpg", "Generate sketch"),
]
wanted = set(filter(None, a.only.split(",")))

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": W, "height": H})
    page.goto(a.base, wait_until="networkidle")
    for key, label, sample, run_text in WORKSPACES:
        if wanted and key not in wanted:
            continue
        page.locator("nav button", has_text=label).first.click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / f"{a.prefix}{key}_empty.png"))
        page.locator(f'button[title="{sample}"]').first.click()
        if key != "sketch":  # salt-and-pepper, high severity is the default showcase
            page.get_by_role("button", name="High", exact=True).click()
        run = page.get_by_role("button", name=re.compile(run_text)).first
        box = run.bounding_box()
        run.click()
        page.wait_for_selector("text=/Done ·/", timeout=60000)
        page.wait_for_timeout(700)
        page.screenshot(path=str(out / f"{a.prefix}{key}.png"))
        m = page.evaluate("""() => {
            const scrolling = [...document.querySelectorAll('*')].filter(e => {
                const s = getComputedStyle(e); return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 2; })
                .map(e => `${e.getBoundingClientRect().x < 300 ? 'SIDEBAR' : (e.getBoundingClientRect().x < 620 ? 'LEFT controls' : 'RIGHT results')} +${e.scrollHeight - e.clientHeight}px`);
            const tiles = [...document.querySelectorAll('figure img')].map(i => { const r = i.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; });
            return {pageOverflow: document.documentElement.scrollHeight - window.innerHeight, scrolling, tiles};
        }""")
        print(f"[{key}] viewport {W}x{H} | Run button bottom edge at y={box['y'] + box['height']:.0f} "
              f"({'visible' if box['y'] + box['height'] <= H else 'OFF SCREEN'}) | page overflow {m['pageOverflow']}px | "
              f"inner scroll: {m['scrolling'] or 'none'} | tiles (w x h): {m['tiles']}")
    browser.close()
print("saved screenshots to", out)
