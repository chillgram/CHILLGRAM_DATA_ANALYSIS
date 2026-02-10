\# 파일 설명

main.py : VM 에 redis 로 중복 체크 => 크롤링 => BigQuery 적재 => vertex AI => JSON 을 GCS 저장 => PDF 를 GCS 저장

db.save.py : 로컬 호스트 호출해서 (상품번호, 크롤링 갯수) 인자 넣고 json 파일로 저장

