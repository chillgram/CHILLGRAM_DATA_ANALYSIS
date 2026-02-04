import re
import math
from fastapi import FastAPI, Request
from playwright.async_api import async_playwright

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

async def get_total_review_pages(page, max_reviews):
    try:
        count_span = await page.wait_for_selector("span.count", timeout=3000)
        count_text = (await count_span.inner_text()).strip()
        count_text = count_text.replace(',', '')
        match = re.search(r"(\d+)", count_text)
        if match:
            total_reviews = int(match.group(1))
            original_reviews = total_reviews
            total_reviews = min(total_reviews, max_reviews)
            total_pages = math.ceil(total_reviews / 10)
            print(f"[INFO] 총 리뷰 수: {original_reviews}, 크롤링 대상: {total_reviews}, 필요한 페이지 수: {total_pages}")
            return total_pages
    except:
        print("[ERROR] 리뷰 수를 찾을 수 없습니다.")
    return 0

async def crawl_all_coupang_reviews(product_url, max_reviews=20):
    all_reviews = []
    count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = await browser.new_page()
        await page.set_extra_http_headers({
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
        })

        await page.goto(product_url)
        await page.wait_for_timeout(2000)

        # '베스트순' 버튼 클릭
        try:
            await page.get_by_role("button", name="베스트순").click()
            await page.wait_for_timeout(2000)
        except:
            print("[WARN] 베스트순 버튼 클릭 실패 또는 이미 적용됨")

        num_pages = await get_total_review_pages(page, max_reviews)

        for page_num in range(1, num_pages + 1):
            print(f"\n=== {page_num}페이지 리뷰 수집 중 ===")

            if page_num != 1:
                btn = page.locator("#sdpReview").locator(
                    f"button:has(span:text-is('{page_num}'))"
                ).first

                await btn.scroll_into_view_if_needed()
                await btn.wait_for(state="visible", timeout=5000)
                await btn.click()

                await page.wait_for_timeout(500)

            await page.wait_for_selector("#sdpReview", timeout=10000)
            await page.wait_for_selector("#sdpReview article", timeout=10000)

            reviews_locator = page.locator("#sdpReview article")
            n = await reviews_locator.count()
            print("이번 페이지 article 개수:", n)

            for i in range(n):
                review = reviews_locator.nth(i)

                # 1. 작성자
                author_loc = review.locator("span[data-member-id]")
                author = (await author_loc.first.inner_text()).strip() if await author_loc.count() > 0 else "Unknown"

                # 2. 작성일
                date_loc = review.locator("div.twc-text-\\[14px\\]\\/\\[15px\\]")
                date = (await date_loc.first.inner_text()).strip() if await date_loc.count() > 0 else ""

                # 3. 상품명 (twc-line-clamp 클래스)
                product_loc = review.locator("div[class*='twc-line-clamp']")
                product_name = (await product_loc.first.inner_text()).strip() if await product_loc.count() > 0 else "Unknown"

                # 4. 전체 텍스트에서 리뷰 본문 추출
                full_text = (await review.text_content()) or ""

                # 상품명 이후 텍스트 = 리뷰 본문 + 부가정보
                if product_name and product_name in full_text:
                    idx = full_text.find(product_name) + len(product_name)
                    review_text = full_text[idx:].strip()

                    # 끝부분 부가정보 제거 (맛 만족도, 도움이 돼요 등)
                    for suffix in ["도움이 돼요신고하기", "신고하기", "도움이 돼요"]:
                        if review_text.endswith(suffix):
                            review_text = review_text[:-len(suffix)].strip()

                    # "맛 만족도" 이후 제거
                    if "맛 만족도" in review_text:
                        review_text = review_text[:review_text.find("맛 만족도")].strip()
                else:
                    review_text = ""

                if review_text:
                    all_reviews.append({
                        "author": author,
                        "date": date,
                        "product": product_name,
                        "content": review_text
                    })
                    count += 1

        print(f"[DONE] 총 크롤링한 리뷰 수: {count}\n")
        await browser.close()

    return all_reviews

@app.post("/")
async def crawl_reviews(request: Request):
    body = await request.json()
    product_id = body.get("product_id")
    max_reviews = body.get("max_reviews", 20)

    if not product_id:
        return {"error": "product_id is required"}

    product_url = f"https://www.coupang.com/vp/products/{product_id}"
    reviews = await crawl_all_coupang_reviews(product_url, max_reviews)

    if not reviews:
        return {"error": "리뷰를 찾을 수 없습니다"}

    product_name = reviews[0]["product"]
    data = [{"review": r["content"], "ts": r["date"]} for r in reviews]

    return {"name": product_name, "data": data}
