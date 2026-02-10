import re
import math
import json
import uuid
import asyncio
import logging
import time
import os
import socket
import threading
import base64
import subprocess
from datetime import datetime

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response
from playwright.async_api import async_playwright
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import redis
from google.cloud import bigquery, storage
import vertexai
from vertexai.generative_models import GenerativeModel

# ==================== 설정 ====================
PROJECT_ID = "chillgram-deploy"
DATASET_ID = "raw"
TABLE_ID = "crawl_events"
GCS_BUCKET = "chillgram-deploy-analysis-pdfs"
VERTEX_LOCATION = "us-central1"

# 프록시 설정
PROXY_HOST = "kr.decodo.com"
PROXY_PORT = 10001
PROXY_USER = "user-spjhm4tfwt-sessionduration-30"
PROXY_PASS = "7bznkJZ07~y5xIkleP"
LOCAL_PROXY_PORT = 18080  # 8080은 uvicorn이 사용하므로 다른 포트

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Redis 연결 (연결 실패 시 graceful fallback)
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("Redis 연결 성공")
except Exception:
    redis_client = None
    logger.warning("Redis 연결 실패 — 인메모리 캐시로 대체합니다")

# Redis 대체용 인메모리 캐시
_memory_cache: dict[str, dict] = {}


# ==================== 유틸리티 ====================
def extract_product_id(coupang_url: str) -> str | None:
    """쿠팡 URL에서 product_id를 추출합니다."""
    match = re.search(r"/products/(\d+)", coupang_url)
    return match.group(1) if match else None


# ==================== Redis / 캐시 ====================
REFRESH_DAYS = 90  # 이 기간이 지나면 다시 크롤링

def check_duplicate(product_id: str) -> bool:
    """중복 체크 — 90일 이상 지난 데이터는 만료 처리."""
    status = get_status(product_id)
    if not status:
        return False

    # 진행 중인 작업이 있으면 중복으로 처리
    if status.get("status") in ("queued", "crawling", "saving", "analyzing", "generating_pdf"):
        return True

    # done 상태인데 90일 지났으면 만료 → 다시 크롤링
    if status.get("status") == "done":
        updated_at = status.get("updated_at", "")
        try:
            last_update = datetime.fromisoformat(updated_at)
            if (datetime.now() - last_update).days >= REFRESH_DAYS:
                logger.info(f"[REFRESH] {product_id}: {REFRESH_DAYS}일 경과 → 재크롤링")
                return False
        except (ValueError, TypeError):
            pass
        return True

    # error 상태면 재시도 허용
    if status.get("status") == "error":
        return False

    return True


def set_status(product_id: str, status: str, **extra):
    key = f"coupang:{product_id}:status"
    value = {"status": status, "updated_at": datetime.now().isoformat(), **extra}
    if redis_client:
        redis_client.set(key, json.dumps(value))
    else:
        _memory_cache[key] = value


def get_status(product_id: str) -> dict | None:
    key = f"coupang:{product_id}:status"
    if redis_client:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    return _memory_cache.get(key)


# ==================== BigQuery 저장 ====================
def save_to_bigquery(product_id: str, product_name: str, reviews: list[dict]):
    """크롤링한 리뷰를 BigQuery에 적재합니다."""
    logger.info(f"BigQuery 적재 시작: {len(reviews)}개 리뷰")
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    rows = []
    for r in reviews:
        data_json = json.dumps({
            "product_id": product_id,
            "product_name": product_name,
            "review": r.get("content", ""),
            "review_date": r.get("date", ""),
        }, ensure_ascii=False)
        rows.append({
            "data": data_json,
            "message_id": str(uuid.uuid4()),
            "publish_time": datetime.utcnow().isoformat(),
            "attributes": None,
            "subscription_name": "api_upload",
        })

    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        logger.error(f"BigQuery 적재 오류: {errors}")
        raise RuntimeError(f"BigQuery insert failed: {errors}")

    logger.info(f"BigQuery 적재 완료: {len(rows)}건")


def fetch_reviews_from_bigquery(product_id: str) -> tuple[str, list[str]]:
    """BigQuery에서 해당 product_id의 리뷰를 조회합니다."""
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT data
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE JSON_EXTRACT_SCALAR(data, '$.product_id') = @product_id
        ORDER BY publish_time DESC
        LIMIT 1000
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product_id", "STRING", product_id)
        ]
    )
    rows = client.query(query, job_config=job_config).result()

    product_name = ""
    review_texts = []
    for row in rows:
        try:
            data = json.loads(row.data)
            if not product_name:
                product_name = data.get("product_name", "")
            review_texts.append(data.get("review", ""))
        except (json.JSONDecodeError, TypeError):
            continue

    logger.info(f"BigQuery 조회: {len(review_texts)}개 리뷰 (product_id={product_id})")
    return product_name, review_texts


# ==================== Vertex AI 분석 ====================
def analyze_with_vertex(product_name: str, review_texts: list[str]) -> dict:
    """Vertex AI Gemini로 리뷰를 분석합니다."""
    logger.info(f"Vertex AI 분석 시작: {len(review_texts)}개 리뷰")

    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    model = GenerativeModel("gemini-2.5-flash")

    max_for_analysis = min(200, len(review_texts))
    sample = review_texts[:max_for_analysis]
    combined = "\n---\n".join(sample)

    prompt = f"""
당신은 전문 데이터 분석가입니다. 다음 "{product_name}" 제품 리뷰 {max_for_analysis}개를 종합 분석해주세요.

## 출력 형식: 반드시 아래 JSON 구조로만 출력하세요. 다른 텍스트 없이 JSON만 출력하세요.

{{
  "product_name": "{product_name}",
  "sentiment_analysis": {{
    "overall_sentiment_distribution": {{
      "positive": "00%",
      "neutral": "00%",
      "negative": "00%"
    }},
    "overall_sentiment_score": "0.0/10",
    "sentiment_trend_summary": "요약 문장"
  }},
  "keyword_analysis": {{
    "top_10_positive_keywords": [
      {{"keyword": "키워드", "frequency": 0}}
    ],
    "top_10_negative_keywords": [
      {{"keyword": "키워드", "frequency": 0}}
    ]
  }},
  "review_summary": {{
    "three_line_summary": ["요약1", "요약2", "요약3"],
    "representative_positive_reviews": ["리뷰1", "리뷰2", "리뷰3"],
    "representative_negative_reviews": ["리뷰1", "리뷰2", "리뷰3"]
  }},
  "insights": {{
    "top_3_customer_satisfaction_points": ["포인트1", "포인트2", "포인트3"],
    "top_3_customer_dissatisfaction_points": ["포인트1", "포인트2", "포인트3"],
    "improvement_suggestions": ["제안1", "제안2", "제안3"],
    "top_3_marketing_leverage_points": ["포인트1", "포인트2", "포인트3"]
  }},
  "action_items": {{
    "immediate_improvements": ["항목1", "항목2"],
    "marketing_messages": ["메시지1", "메시지2"],
    "competitive_advantages": ["강점1", "강점2"]
  }}
}}

## 리뷰 데이터
{combined}
"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 8192},
    )

    raw_text = response.text.strip()
    # JSON 블록 추출 (```json ... ``` 감싸져 있을 수 있음)
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if json_match:
        analysis = json.loads(json_match.group())
    else:
        raise ValueError(f"Gemini 응답에서 JSON을 파싱할 수 없습니다: {raw_text[:200]}")

    logger.info("Vertex AI 분석 완료")
    return analysis


# ==================== 대시보드 HTML 생성 ====================
def generate_dashboard_html(analysis: dict) -> str:
    """분석 결과로 동적 HTML 대시보드를 생성합니다."""
    product_name = analysis.get("product_name", "제품")
    sentiment = analysis.get("sentiment_analysis", {})
    keywords = analysis.get("keyword_analysis", {})
    insights = analysis.get("insights", {})
    review_summary = analysis.get("review_summary", {})
    action_items = analysis.get("action_items", {})

    distribution = sentiment.get("overall_sentiment_distribution", {})
    positive = int(distribution.get("positive", "0%").replace("%", ""))
    neutral = int(distribution.get("neutral", "0%").replace("%", ""))
    negative = int(distribution.get("negative", "0%").replace("%", ""))
    score = sentiment.get("overall_sentiment_score", "0/10")
    score_num = float(score.split("/")[0])
    score_class = "score-high" if score_num >= 8 else "score-medium" if score_num >= 6 else "score-low"

    pos_keywords = keywords.get("top_10_positive_keywords", [])
    neg_keywords = keywords.get("top_10_negative_keywords", [])
    pos_kw_names = [k["keyword"] for k in pos_keywords]
    pos_kw_freq = [k["frequency"] for k in pos_keywords]
    neg_kw_names = [k["keyword"] for k in neg_keywords]
    neg_kw_freq = [k["frequency"] for k in neg_keywords]

    # 인사이트 리스트
    satisfaction = insights.get("top_3_customer_satisfaction_points", [])
    dissatisfaction = insights.get("top_3_customer_dissatisfaction_points", [])
    marketing = insights.get("top_3_marketing_leverage_points", [])
    improvements = insights.get("improvement_suggestions", [])

    # 리뷰 요약
    summary_lines = review_summary.get("three_line_summary", [])
    pos_reviews = review_summary.get("representative_positive_reviews", [])
    neg_reviews = review_summary.get("representative_negative_reviews", [])

    # 액션 아이템
    immediate = action_items.get("immediate_improvements", [])
    marketing_msg = action_items.get("marketing_messages", [])
    advantages = action_items.get("competitive_advantages", [])

    def li_items(items):
        return "\n".join(f"<li>{item}</li>" for item in items)

    def blockquote_items(items):
        return "\n".join(f'<blockquote>"{item}"</blockquote>' for item in items)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{product_name} - 리뷰 분석 리포트</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh; color: #fff; padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 40px 0 30px; }}
        .header h1 {{
            font-size: 2.2rem; font-weight: 700;
            background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header p {{ color: #94A3B8; font-size: 1rem; }}
        .section {{
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px; padding: 28px; margin-bottom: 28px;
        }}
        .section-title {{
            font-size: 1.3rem; font-weight: 600; margin-bottom: 20px;
            padding-bottom: 12px; border-bottom: 2px solid rgba(99,102,241,0.3);
        }}
        .charts-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px;
        }}
        .chart-box {{
            background: rgba(255,255,255,0.02); border-radius: 14px;
            padding: 18px; border: 1px solid rgba(255,255,255,0.05); min-height: 320px;
        }}
        .chart-label {{ font-size: 1rem; font-weight: 500; margin-bottom: 12px; color: #E2E8F0; }}
        .insights-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px;
        }}
        .insight-card {{
            background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px; padding: 18px;
        }}
        .insight-card.green {{ background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); }}
        .insight-card.red {{ background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.3); }}
        .insight-card.yellow {{ background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); }}
        .insight-label {{
            font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 10px; color: #A5B4FC;
        }}
        .insight-card.green .insight-label {{ color: #10B981; }}
        .insight-card.red .insight-label {{ color: #EF4444; }}
        .insight-card.yellow .insight-label {{ color: #F59E0B; }}
        ul {{ list-style: none; }}
        ul li {{
            padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.93rem; color: #CBD5E1; line-height: 1.6;
        }}
        ul li:last-child {{ border-bottom: none; }}
        ul li::before {{ content: "→ "; color: #6366F1; }}
        blockquote {{
            background: rgba(255,255,255,0.03); border-left: 3px solid #6366F1;
            padding: 10px 14px; margin: 8px 0; border-radius: 0 8px 8px 0;
            font-size: 0.9rem; color: #94A3B8; font-style: italic;
        }}
        .score-badge {{
            display: inline-block; padding: 6px 14px; border-radius: 20px;
            font-weight: 600; font-size: 1.1rem;
        }}
        .score-high {{ background: linear-gradient(135deg, #10B981, #059669); }}
        .score-medium {{ background: linear-gradient(135deg, #F59E0B, #D97706); }}
        .score-low {{ background: linear-gradient(135deg, #EF4444, #DC2626); }}
        .summary-box {{
            background: rgba(99,102,241,0.08); border-radius: 12px;
            padding: 18px; margin-bottom: 18px;
        }}
        .summary-box p {{ color: #CBD5E1; line-height: 1.8; font-size: 0.95rem; }}
        .footer {{
            text-align: center; padding: 30px 0; color: #64748B; font-size: 0.85rem;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{product_name} 리뷰 분석 리포트</h1>
        <p>Vertex AI Gemini 2.5 Flash | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <!-- 감성 점수 & 요약 -->
    <div class="section">
        <div class="section-title">
            감성 분석
            <span class="score-badge {score_class}" style="margin-left:16px;">감성 점수: {score}</span>
        </div>
        <p style="color:#94A3B8;margin-bottom:16px;">{sentiment.get("sentiment_trend_summary", "")}</p>
        <div class="summary-box">
            <p>{"<br>".join(summary_lines)}</p>
        </div>
        <div class="charts-grid">
            <div class="chart-box">
                <div class="chart-label">감성 분포</div>
                <div id="sentiment-chart" style="height:280px;"></div>
            </div>
            <div class="chart-box">
                <div class="chart-label">긍정 키워드 TOP 10</div>
                <div id="pos-keywords" style="height:280px;"></div>
            </div>
            <div class="chart-box">
                <div class="chart-label">부정 키워드 TOP 10</div>
                <div id="neg-keywords" style="height:280px;"></div>
            </div>
        </div>
    </div>

    <!-- 대표 리뷰 -->
    <div class="section">
        <div class="section-title">대표 리뷰</div>
        <div class="charts-grid">
            <div class="insight-card green">
                <div class="insight-label">긍정 리뷰</div>
                {blockquote_items(pos_reviews)}
            </div>
            <div class="insight-card red">
                <div class="insight-label">부정 리뷰</div>
                {blockquote_items(neg_reviews)}
            </div>
        </div>
    </div>

    <!-- 인사이트 -->
    <div class="section">
        <div class="section-title">인사이트 & 액션 아이템</div>
        <div class="insights-grid">
            <div class="insight-card green">
                <div class="insight-label">고객 만족 포인트</div>
                <ul>{li_items(satisfaction)}</ul>
            </div>
            <div class="insight-card red">
                <div class="insight-label">개선 필요 사항</div>
                <ul>{li_items(dissatisfaction)}</ul>
            </div>
            <div class="insight-card">
                <div class="insight-label">개선 제안</div>
                <ul>{li_items(improvements)}</ul>
            </div>
            <div class="insight-card green">
                <div class="insight-label">마케팅 활용 포인트</div>
                <ul>{li_items(marketing)}</ul>
            </div>
            <div class="insight-card yellow">
                <div class="insight-label">즉시 개선 항목</div>
                <ul>{li_items(immediate)}</ul>
            </div>
            <div class="insight-card">
                <div class="insight-label">마케팅 메시지</div>
                <ul>{li_items(marketing_msg)}</ul>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Generated by BigQuery + Vertex AI Gemini 2.5 Flash</p>
        <p>CHILLGRAM Review Analysis | {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
</div>

<script>
Plotly.newPlot('sentiment-chart', [{{
    values: [{positive}, {neutral}, {negative}],
    labels: ['긍정 ({positive}%)', '중립 ({neutral}%)', '부정 ({negative}%)'],
    type: 'pie', hole: 0.55,
    marker: {{ colors: ['#10B981', '#F59E0B', '#EF4444'] }},
    textinfo: 'label', textfont: {{ color: '#fff', size: 12 }},
    hoverinfo: 'label+percent'
}}], {{
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: {{ color: '#fff' }}, showlegend: false,
    margin: {{ t: 20, b: 20, l: 20, r: 20 }},
    annotations: [{{ text: '<b>{score}</b>', font: {{ size: 22, color: '#fff' }}, showarrow: false }}]
}}, {{responsive: true}});

Plotly.newPlot('pos-keywords', [{{
    x: {json.dumps(pos_kw_freq[::-1])},
    y: {json.dumps(pos_kw_names[::-1], ensure_ascii=False)},
    type: 'bar', orientation: 'h',
    marker: {{ color: 'rgba(16,185,129,0.8)', line: {{ color: '#10B981', width: 1 }} }},
    text: {json.dumps(pos_kw_freq[::-1])}, textposition: 'outside',
    textfont: {{ color: '#10B981', size: 11 }}
}}], {{
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: {{ color: '#CBD5E1' }},
    xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
    yaxis: {{ showgrid: false, tickfont: {{ size: 11 }} }},
    margin: {{ t: 10, b: 20, l: 70, r: 40 }}, bargap: 0.25
}}, {{responsive: true}});

Plotly.newPlot('neg-keywords', [{{
    x: {json.dumps(neg_kw_freq[::-1])},
    y: {json.dumps(neg_kw_names[::-1], ensure_ascii=False)},
    type: 'bar', orientation: 'h',
    marker: {{ color: 'rgba(239,68,68,0.8)', line: {{ color: '#EF4444', width: 1 }} }},
    text: {json.dumps(neg_kw_freq[::-1])}, textposition: 'outside',
    textfont: {{ color: '#EF4444', size: 11 }}
}}], {{
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: {{ color: '#CBD5E1' }},
    xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
    yaxis: {{ showgrid: false, tickfont: {{ size: 11 }} }},
    margin: {{ t: 10, b: 20, l: 70, r: 40 }}, bargap: 0.25
}}, {{responsive: true}});
</script>
</body>
</html>"""
    return html


# ==================== HTML → PDF (Playwright) ====================
async def html_to_pdf(html_content: str) -> bytes:
    """Playwright로 HTML을 PDF로 변환합니다."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        # Plotly 차트 렌더링 대기
        await page.wait_for_timeout(3000)
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"},
        )
        await browser.close()
    return pdf_bytes


# ==================== GCS 저장/조회 ====================
def save_pdf_to_gcs(product_id: str, pdf_bytes: bytes) -> str:
    """PDF를 GCS에 저장하고 경로를 반환합니다."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob_path = f"pdfs/{product_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    logger.info(f"GCS 저장 완료: gs://{GCS_BUCKET}/{blob_path}")
    return blob_path


def get_pdf_from_gcs(product_id: str) -> bytes | None:
    """GCS에서 최신 PDF를 다운로드합니다."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix=f"pdfs/{product_id}/"))
    if not blobs:
        return None
    latest = sorted(blobs, key=lambda b: b.name, reverse=True)[0]
    logger.info(f"GCS 조회: {latest.name}")
    return latest.download_as_bytes()


# ==================== 프록시 & Xvfb (run_crawler_test.py 방식) ====================
def setup_xvfb(display=":99"):
    os.environ["DISPLAY"] = display
    xvfb_proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    logger.info(f"Xvfb 가상 디스플레이 시작됨 (DISPLAY={display})")
    return xvfb_proc


def _pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
        try:
            dst.close()
        except Exception:
            pass


def _handle_client(client_sock):
    try:
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = client_sock.recv(4096)
            if not chunk:
                client_sock.close()
                return
            request += chunk

        first_line = request.split(b"\r\n")[0].decode("utf-8", errors="replace")
        method, target, _ = first_line.split(" ", 2)

        upstream = socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=15)
        auth = base64.b64encode(f"{PROXY_USER}:{PROXY_PASS}".encode()).decode()

        if method == "CONNECT":
            connect_req = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\nProxy-Authorization: Basic {auth}\r\nConnection: keep-alive\r\n\r\n"
            upstream.sendall(connect_req.encode())

            response = b""
            while b"\r\n\r\n" not in response:
                chunk = upstream.recv(4096)
                if not chunk:
                    break
                response += chunk

            if b"200" in response.split(b"\r\n")[0]:
                client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                t1 = threading.Thread(target=_pipe, args=(client_sock, upstream), daemon=True)
                t2 = threading.Thread(target=_pipe, args=(upstream, client_sock), daemon=True)
                t1.start()
                t2.start()
                t1.join()
                t2.join()
            else:
                client_sock.sendall(response)
                client_sock.close()
                upstream.close()
        else:
            header_end = request.find(b"\r\n\r\n")
            headers_part = request[:header_end]
            body_part = request[header_end:]
            auth_header = f"Proxy-Authorization: Basic {auth}\r\n".encode()
            first_line_end = headers_part.find(b"\r\n") + 2
            modified_request = headers_part[:first_line_end] + auth_header + headers_part[first_line_end:] + body_part
            upstream.sendall(modified_request)

            t1 = threading.Thread(target=_pipe, args=(client_sock, upstream), daemon=True)
            t2 = threading.Thread(target=_pipe, args=(upstream, client_sock), daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    except Exception:
        try:
            client_sock.close()
        except Exception:
            pass


_proxy_server = None


def start_local_proxy():
    global _proxy_server
    if _proxy_server is not None:
        return _proxy_server

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", LOCAL_PROXY_PORT))
    server.listen(50)

    def accept_loop():
        while True:
            try:
                client_sock, _ = server.accept()
                t = threading.Thread(target=_handle_client, args=(client_sock,), daemon=True)
                t.start()
            except Exception:
                break

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    time.sleep(1)
    logger.info(f"로컬 프록시 포워더 시작됨 (localhost:{LOCAL_PROXY_PORT} → {PROXY_HOST}:{PROXY_PORT})")
    _proxy_server = server
    return server


# ==================== 크롤링 (undetected_chromedriver + 프록시) ====================
def get_total_review_pages(driver, max_reviews):
    try:
        count_span = driver.find_element(By.CSS_SELECTOR, "span.count")
        if count_span:
            count_text = count_span.text.strip().replace(",", "")
            match = re.search(r"(\d+)", count_text)
            if match:
                total_reviews = int(match.group(1))
                original_reviews = total_reviews
                total_reviews = min(total_reviews, max_reviews)
                total_pages = math.ceil(total_reviews / 10)
                logger.info(f"총 리뷰 수: {original_reviews}, 크롤링 대상: {total_reviews}, 페이지: {total_pages}")
                return total_pages
    except Exception as e:
        logger.error(f"리뷰 수를 찾을 수 없습니다: {e}")
    return 0


def crawl_all_coupang_reviews(product_url, max_reviews=20):
    all_reviews = []
    count = 0

    # Xvfb & 프록시 시작
    xvfb_proc = setup_xvfb()
    start_local_proxy()

    logger.info("Chrome 브라우저 시작 중 (undetected-chromedriver + Proxy)...")
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--proxy-server=http://127.0.0.1:{LOCAL_PROXY_PORT}")
    options.add_argument("--lang=ko-KR")

    driver = uc.Chrome(options=options, headless=False, version_main=144)

    try:
        # 쿠팡 메인 페이지 먼저 방문 (쿠키 획득)
        logger.info("쿠팡 메인 페이지 방문 중 (쿠키 획득)...")
        driver.get("https://www.coupang.com")
        time.sleep(5)
        logger.info(f"메인 페이지 타이틀: {driver.title}")

        # 상품 페이지 접속
        logger.info(f"페이지 접속 시도: {product_url}")
        driver.get(product_url)
        time.sleep(5)

        logger.info(f"페이지 타이틀: {driver.title}")

        # Access Denied 체크
        if "Access Denied" in driver.title:
            logger.error("Access Denied - 페이지 접근이 차단되었습니다")
            return []

        # 리뷰 섹션 찾기
        try:
            review_section = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "sdpReview"))
            )
        except Exception:
            logger.error("리뷰 섹션을 찾을 수 없습니다")
            return []

        driver.execute_script("arguments[0].scrollIntoView();", review_section)
        logger.info("리뷰 섹션 발견")
        time.sleep(1)

        # 베스트순 정렬
        try:
            best_btn = driver.find_element(By.XPATH, "//button[contains(text(), '베스트순')]")
            if best_btn:
                best_btn.click()
                time.sleep(1)
                logger.info("베스트순 정렬 적용됨")
        except Exception:
            pass

        num_pages = get_total_review_pages(driver, max_reviews)

        if num_pages == 0:
            logger.error("페이지 수가 0입니다.")
            return []

        for page_num in range(1, num_pages + 1):
            logger.info(f"{page_num}/{num_pages} 페이지 크롤링 중")

            if page_num != 1:
                try:
                    page_btn = driver.find_element(
                        By.XPATH, f"//*[@id='sdpReview']//button[span[text()='{page_num}']]"
                    )
                    driver.execute_script("arguments[0].scrollIntoView();", page_btn)
                    time.sleep(0.2)
                    page_btn.click()
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 클릭 실패: {e}")
                    continue

            reviews = driver.find_elements(By.CSS_SELECTOR, "#sdpReview article")

            for review in reviews:
                try:
                    try:
                        product_el = review.find_element(By.CSS_SELECTOR, "div[class*='twc-line-clamp']")
                        product_name = product_el.text.strip()
                    except Exception:
                        product_name = ""

                    full_text = review.text or ""

                    if product_name and product_name in full_text:
                        idx = full_text.find(product_name) + len(product_name)
                        review_text = full_text[idx:].strip()

                        for suffix in ["도움이 돼요신고하기", "신고하기", "도움이 돼요"]:
                            if review_text.endswith(suffix):
                                review_text = review_text[:-len(suffix)].strip()

                        if "맛 만족도" in review_text:
                            review_text = review_text[:review_text.find("맛 만족도")].strip()
                    else:
                        review_text = ""

                    if review_text and len(review_text) > 5:
                        all_reviews.append({
                            "product": product_name,
                            "content": review_text,
                        })
                        count += 1
                except Exception:
                    continue

        logger.info(f"크롤링 완료: {count}개 리뷰")

    except Exception as e:
        logger.error(f"크롤링 중 오류: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        try:
            xvfb_proc.terminate()
        except Exception:
            pass

    return all_reviews


# ==================== 비동기 파이프라인 ====================
async def background_pipeline(product_id: str, coupang_url: str, max_reviews: int):
    """등록 시 백그라운드로 실행되는 전체 파이프라인."""
    try:
        # 1. 크롤링 (동기 함수 → asyncio.to_thread)
        set_status(product_id, "crawling")
        logger.info(f"[PIPELINE] 크롤링 시작: {product_id}")
        reviews = await asyncio.to_thread(crawl_all_coupang_reviews, coupang_url, max_reviews)

        if not reviews:
            set_status(product_id, "error", message="리뷰를 찾을 수 없습니다")
            return

        product_name = reviews[0].get("product", "Unknown")

        # 2. BigQuery 적재
        set_status(product_id, "saving")
        logger.info(f"[PIPELINE] BigQuery 적재: {product_id}")
        await asyncio.to_thread(save_to_bigquery, product_id, product_name, reviews)

        # 3. Vertex AI 분석
        set_status(product_id, "analyzing")
        logger.info(f"[PIPELINE] Vertex AI 분석: {product_id}")
        review_texts = [r["content"] for r in reviews]
        analysis = await asyncio.to_thread(analyze_with_vertex, product_name, review_texts)

        # 4. 대시보드 HTML 생성
        set_status(product_id, "generating_pdf")
        logger.info(f"[PIPELINE] PDF 생성: {product_id}")
        html = generate_dashboard_html(analysis)

        # 5. PDF 변환
        pdf_bytes = await html_to_pdf(html)

        # 6. GCS 저장
        pdf_path = await asyncio.to_thread(save_pdf_to_gcs, product_id, pdf_bytes)

        # 7. 완료
        set_status(product_id, "done", pdf_path=pdf_path, review_count=len(reviews))
        logger.info(f"[PIPELINE] 완료: {product_id} ({len(reviews)}개 리뷰)")

    except Exception as e:
        logger.error(f"[PIPELINE] 오류: {product_id} — {e}")
        set_status(product_id, "error", message=str(e))


# ==================== API 엔드포인트 ====================
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/register")
async def register_product(request: Request, background_tasks: BackgroundTasks):
    """쿠팡 URL을 등록하고 비동기로 크롤링+분석+PDF 생성을 시작합니다."""
    body = await request.json()
    coupang_url = body.get("coupang_url", "")
    max_reviews = body.get("max_reviews", 100)

    product_id = extract_product_id(coupang_url)
    if not product_id:
        return {"status": "error", "message": "유효하지 않은 쿠팡 URL입니다"}

    if check_duplicate(product_id):
        return {"status": "already_exists", "product_id": product_id}

    # 비동기 파이프라인 시작
    background_tasks.add_task(background_pipeline, product_id, coupang_url, max_reviews)
    set_status(product_id, "queued")

    return {"status": "crawling_started", "product_id": product_id}


@app.get("/status/{product_id}")
def get_product_status(product_id: str):
    """크롤링/분석 진행 상태를 확인합니다."""
    status = get_status(product_id)
    if not status:
        return {"status": "not_found", "product_id": product_id}
    return {"product_id": product_id, **status}


@app.post("/analyze")
async def analyze_product(request: Request):
    """분석 완료된 제품의 PDF를 반환합니다."""
    body = await request.json()
    product_id = body.get("product_id", "")

    if not product_id:
        return {"status": "error", "message": "product_id가 필요합니다"}

    status_info = get_status(product_id)
    if not status_info:
        return {"status": "not_found", "message": "등록되지 않은 제품입니다"}

    current_status = status_info.get("status", "")
    if current_status != "done":
        return {
            "status": "processing",
            "current_step": current_status,
            "message": f"아직 처리 중입니다 (현재 단계: {current_status})",
        }

    # GCS에서 PDF 가져오기
    pdf_bytes = await asyncio.to_thread(get_pdf_from_gcs, product_id)
    if not pdf_bytes:
        return {"status": "error", "message": "PDF 파일을 찾을 수 없습니다"}

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="analysis_{product_id}.pdf"'},
    )


@app.post("/crawl")
async def crawl_reviews(request: Request):
    """기존 호환용 크롤링 전용 엔드포인트."""
    body = await request.json()
    product_id = body.get("product_id")
    max_reviews = body.get("max_reviews", 20)

    if not product_id:
        return {"error": "product_id is required"}

    product_url = f"https://www.coupang.com/vp/products/{product_id}"
    reviews = await asyncio.to_thread(crawl_all_coupang_reviews, product_url, max_reviews)

    if not reviews:
        return {"error": "리뷰를 찾을 수 없습니다"}

    product_name = reviews[0]["product"]
    data = [{"review": r["content"], "ts": r["date"]} for r in reviews]

    return {"name": product_name, "data": data}
