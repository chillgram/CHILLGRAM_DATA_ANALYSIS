import json
import urllib.request

def save_reviews(product_id, max_reviews=30, output_file='reviews.json'):
    req = urllib.request.Request(
        'http://localhost:8080/',
        data=json.dumps({'product_id': product_id, 'max_reviews': max_reviews}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode('utf-8'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'저장 완료: {data["name"]}')
    print(f'리뷰 수: {len(data["data"])}개')
    print(f'파일: {output_file}')
    return data

if __name__ == "__main__":
    save_reviews('7225189423', max_reviews=30)
