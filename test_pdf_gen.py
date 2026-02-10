"""
BigQuery에 있는 신라면 데이터로 PDF 생성 테스트
VM에서 실행: python test_pdf_gen.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from main import (
    fetch_reviews_from_bigquery,
    analyze_with_vertex,
    save_analysis_to_gcs,
    generate_dashboard_html,
    html_to_pdf,
    save_pdf_to_gcs,
)

PRODUCT_ID = "9264507739"


async def main():
    # 1. BigQuery에서 리뷰 가져오기
    print(f"[1/5] BigQuery에서 리뷰 조회 중... (product_id={PRODUCT_ID})")
    product_name, review_texts = fetch_reviews_from_bigquery(PRODUCT_ID)
    print(f"  → {len(review_texts)}개 리뷰, 상품명: {product_name}")

    if not review_texts:
        print("[ERROR] BigQuery에 리뷰가 없습니다.")
        return

    # 2. Vertex AI 분석
    print("[2/5] Vertex AI 분석 중...")
    analysis = analyze_with_vertex(product_name, review_texts)
    print("  → 분석 완료")

    # 3. 분석 JSON GCS 저장
    print("[3/5] 분석 JSON GCS 저장 중...")
    analysis_path = save_analysis_to_gcs(PRODUCT_ID, analysis)
    print(f"  → {analysis_path}")

    # 4. HTML → PDF
    print("[4/5] HTML 생성 + PDF 변환 중...")
    html = generate_dashboard_html(analysis)
    pdf_bytes = await html_to_pdf(html)
    print(f"  → PDF 생성 완료 ({len(pdf_bytes):,} bytes)")

    # 5. GCS 저장
    print("[5/5] PDF GCS 저장 중...")
    pdf_path = save_pdf_to_gcs(PRODUCT_ID, pdf_bytes)
    print(f"  → {pdf_path}")

    print(f"\n=== 완료 ===")
    print(f"GCS PDF: gs://chillgram-deploy-analysis-pdfs/{pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
