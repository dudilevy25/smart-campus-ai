import os
import re
from datetime import datetime
from collections import defaultdict

VIZ_DIR = "visualization"
OUT_HTML = os.path.join(VIZ_DIR, "index.html")

# סדר עדיפות להצגה בכל שורה
PLOT_ORDER = ["hist", "box", "trend"]

def parse_images():
    """
    מחזיר:
    - features: dict[str, dict[str, str]]  -> {"Temperature": {"hist": "05_temperature_boxplot.png", "trend": ...}, ...}
    - correlations: list[str]              -> ["16_correlation_heatmap.png", ...]
    - others: list[str]                    -> כל השאר (כולל Occupancy Count, All Features Trend)
    """
    features = defaultdict(dict)
    correlations = []
    others = []

    if not os.path.isdir(VIZ_DIR):
        raise SystemExit(f"Folder '{VIZ_DIR}' not found. Run your EDA script first.")

    imgs = [f for f in os.listdir(VIZ_DIR) if f.lower().endswith(".png")]
    imgs.sort()  # סדר לפי 01_, 02_, ...

    # תבנית 1: NN_(hist|box|trend)_(Feature).png
    patt_kft = re.compile(r"^\d+_(hist|box|trend)_(.+)\.png$", re.IGNORECASE)
    # תבנית 2: NN_(Feature)_(hist|box|trend).png  <-- זו התבנית של data_analysis.py
    patt_fk = re.compile(r"^\d+_(.+)_(hist|box|trend)\.png$", re.IGNORECASE)

    for img in imgs:
        base_lower = img.lower()

        # כל הקורולציות ייכנסו ל-General (לא סקשן נפרד)
        if "correlation_heatmap" in base_lower:
            correlations.append(img)
            continue

        m1 = patt_kft.match(img)
        m2 = patt_fk.match(img)

        if m1:
            kind = m1.group(1).lower()
            feature_original = m1.group(2)
            features[feature_original][kind] = img
        elif m2:
            feature_original = m2.group(1)
            kind = m2.group(2).lower()
            features[feature_original][kind] = img
        else:
            # למשל: 02_occupancy_histogram.png, 21_all_features_trend.png, וכו'
            others.append(img)

    return features, correlations, others

def img_card(src: str, title: str):
    # index.html נמצא בתוך visualization/, לכן src הוא רק שם הקובץ
    return f"""
      <div class="card" tabindex="0" role="button" aria-label="הגדל {title}" data-img="{src}">
        <img src="{src}" alt="{title}" />
        <div class="caption">{title}</div>
      </div>
    """

def row_for_feature(name: str, kinds_map: dict):
    # בניית שורה לפי סדר קבוע: hist → box → trend (מה שקיים)
    cards = []
    label_map = {"hist": "Histogram", "box": "Boxplot", "trend": "Trend"}

    for k in PLOT_ORDER:
        if k in kinds_map:
            cards.append(img_card(kinds_map[k], f"{label_map[k]} — {name}"))

    if not cards:
        return ""

    return f"""
    <section class="feature-row" id="feature-{name}">
      <h2>{name}</h2>
      <div class="row-grid">
        {''.join(cards)}
      </div>
    </section>
    """

def nice_title_from_filename(img: str) -> str:
    base = img.lower()
    if "all_features_trend" in base:
        return "All Features — Trend"
    if "correlation_heatmap" in base:
        return "Correlation Heatmap"
    # כותרת גנרית מנורמלת
    pretty = img.split("_", 1)[-1].rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
    return pretty.title()

def automated_reports_section():
    """
    קישורים לדוחות האוטומטיים (אם קיימים):
    - reports/eda_ydata_profiling.html
    - reports/eda_sweetviz.html
    - reports/autoviz/   (תיקייה עם קבצי HTML מרובים)
    """
    links = []

    # index.html יושב בתוך visualization/, לכן נשתמש ב- ../reports/...
    ydata_path = os.path.join("reports", "eda_ydata_profiling.html")
    sweetviz_path = os.path.join("reports", "eda_sweetviz.html")
    autoviz_dir = os.path.join("reports", "autoviz")

    if os.path.exists(ydata_path):
        links.append('<a href="../reports/eda_ydata_profiling.html" target="_blank" rel="noopener">📊 ydata-profiling Report</a>')
    if os.path.exists(sweetviz_path):
        links.append('<a href="../reports/eda_sweetviz.html" target="_blank" rel="noopener">🍭 Sweetviz Report</a>')
    if os.path.isdir(autoviz_dir):
        links.append('<a href="../reports/autoviz" target="_blank" rel="noopener">⚡ AutoViz Report (folder)</a>')

    if not links:
        return ""

    return f"""
    <section class="feature-row" id="auto-reports">
      <h2>Automated Reports</h2>
      <ul class="links">
        {''.join(f'<li>{lnk}</li>' for lnk in links)}
      </ul>
    </section>
    """

def build_html(features, correlations, others):
    # מיון פיצ'רים אלפביתי
    feature_names = sorted(features.keys(), key=lambda s: s.lower())

    rows = []
    for name in feature_names:
        rows.append(row_for_feature(name, features[name]))

    # General = כל מה שלא נכנס לשורות פיצ'רים + כל הקורולציות
    general_imgs = []
    general_imgs.extend(others)
    general_imgs.extend(correlations)

    general_cards = []
    for img in general_imgs:
        general_cards.append(img_card(img, nice_title_from_filename(img)))

    general_row = ""
    if general_cards:
        general_row = f"""
        <section class="feature-row" id="general">
          <h2>General</h2>
          <div class="row-grid">
            {''.join(general_cards)}
          </div>
        </section>
        """

    # מקטע דוחות אוטומטיים
    auto_reports = automated_reports_section()

    html = f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EDA Report</title>
  <style>
    :root {{
      --border: #e5e7eb;
      --text: #111827;
      --muted: #656b72;
      --bg: #ffffff;
      --shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}
    html, body {{
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(255,255,255,0.85);
      backdrop-filter: blur(6px);
      border-bottom: 1px solid var(--border);
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 16px 20px 28px;
    }}
    h1 {{
      font-size: 24px;
      margin: 8px 0 4px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 6px;
    }}
    .toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px 0 12px;
    }}
    .toc a {{
      padding: 6px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      text-decoration: none;
      color: var(--text);
      font-size: 13px;
      box-shadow: var(--shadow);
      background: white;
    }}
    .feature-row {{
      margin: 26px 0;
    }}
    .feature-row > h2 {{
      font-size: 18px;
      margin: 0 0 10px;
    }}
    .row-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 14px;
    }}
    @media (max-width: 900px) {{
      .row-grid {{
        grid-template-columns: repeat(2, minmax(200px, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .row-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      box-shadow: var(--shadow);
      background: #fff;
      cursor: zoom-in;
      outline: none;
    }}
    .card:focus {{
      box-shadow: 0 0 0 3px rgba(59,130,246,0.35);
    }}
    .card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }}
    .caption {{
      font-size: 13px;
      margin-top: 6px;
      color: var(--muted);
    }}
    .links {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    .links a {{
      display: inline-block;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      text-decoration: none;
      color: var(--text);
      background: #fff;
      box-shadow: var(--shadow);
    }}

    /* Lightbox */
    .lightbox {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.75);
      z-index: 50;
      padding: 16px;
    }}
    .lightbox.open {{
      display: flex;
    }}
    .lightbox img {{
      max-width: 96vw;
      max-height: 92vh;
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      background: #fff;
    }}
    .lightbox .close {{
      position: fixed;
      top: 14px;
      left: 16px;
      color: #fff;
      font-size: 16px;
      background: rgba(0,0,0,0.35);
      padding: 6px 10px;
      border-radius: 999px;
      cursor: pointer;
      user-select: none;
    }}
  </style>
</head>
<body>
  <header>
    <div class="container">
      <h1>EDA Report</h1>
      <div class="meta">נוצר בתאריך: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
      <div class="toc">
        {"".join(f'<a href="#feature-{name}">{name}</a>' for name in feature_names)}
        {'<a href="#general">General</a>' if general_cards else ''}
        {'<a href="#auto-reports">Automated Reports</a>' if auto_reports else ''}
      </div>
    </div>
  </header>

  <main class="container">
    {"".join(rows)}
    {general_row}
    {auto_reports}
  </main>

  <!-- Lightbox -->
  <div class="lightbox" id="lightbox" aria-hidden="true">
    <div class="close" id="lightboxClose" aria-label="סגור">סגור ✕</div>
    <img id="lightboxImg" alt="preview" />
  </div>

  <script>
    (function() {{
      const lb = document.getElementById('lightbox');
      const lbImg = document.getElementById('lightboxImg');
      const lbClose = document.getElementById('lightboxClose');

      function openLightbox(src) {{
        lbImg.src = src;
        lb.classList.add('open');
        lb.setAttribute('aria-hidden', 'false');
      }}
      function closeLightbox() {{
        lb.classList.remove('open');
        lb.setAttribute('aria-hidden', 'true');
        lbImg.src = '';
      }}

      document.addEventListener('click', function(e) {{
        const card = e.target.closest('.card');
        if (card && card.dataset.img) {{
          openLightbox(card.dataset.img);
        }} else if (e.target === lb || e.target === lbImg || e.target === lbClose) {{
          closeLightbox();
        }}
      }});

      document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') closeLightbox();
      }});
    }})();
  </script>
</body>
</html>
"""
    return html

def main():
    features, correlations, others = parse_images()
    html = build_html(features, correlations, others)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report written to: {OUT_HTML}")

if __name__ == "__main__":
    main()
