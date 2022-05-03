import json
import requests
from urllib.parse import urljoin

#URL = 'http://ketiair.com:10022/'
URL = 'http://ketiair.com:10022/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'

example_re = json.dumps(
    {
        "task": "relation_extraction",
        "doc":{
            "text":"안중근은 이토 히로부미를 암살하여 러시아 헌병에게 붙잡혔고 1910년 3월 26일 오전 10시에 살인의 죄형으로 관동주 뤼순형무소에서 교수형으로 순국하였다."
        },
        "arg_pairs":[
            [
                [0,2],
                [33,51]
            ]
        ]
    }
)

example_ner = json.dumps(
    {
        "task": "named_entity_recognition",
        "doc":{
            "text":"대전시는 올해 지역산업 맞춤형 일자리 창출 지원 사업에 국비 33억원을 확보했다고 30일 발표했다."
        }
    }
)

example_cr = json.dumps(
    {
        "task": "coreference_resolution",
        "doc":{
            "text":"식약처가 철저하게 관리하고 있는 ‘화장품 안전기준’에 따르면 THB는 염모제에 사용이 허용되는 합법적인 성분이다. 모다모다가 THB를 발색 샴푸에 처음 사용한 것도 아니다. 사실 THB는 염색약의 색을 조절하는 용도로 1950년 대에도 유럽에서 사용되어왔던 전통적인 성분이다. 1961년 미국 염모제 특허에도 THB가 등장한다. 실제로 THB는 국제화장품성분사전에 등재되어 널리 활용되고 있는 범용 염모제 성분이다."
        },
        "arg_pairs":[
            [
                [34,36]
            ]
        ]
    }
)

headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=example_re, headers=headers)
print(response.status_code)
print(response.request)
print(response.json())
print(response.raise_for_status())
