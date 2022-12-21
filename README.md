# NLP_modules

KE-T5모델을 활용하여 Relation Extraction, Named Entity Recognition, Coreference Resolution 학습 및 테스트

## Dataset

1. Relation Extraction \
KLUE RE Dataset: Huggingface dataloader 활용
2. Named Entity Recognition \
KLUE NER Dataset: Huggingface dataloader 활용
3. Coreference Resolution \
NIKL CR Dataset: NIKL Dataset Download

## Training

KLUE RE Dataset은 CRE 학습 시에 자동으로 데이터를 다운로드 합니다. \
기본적으로 ke-t5-base모델을 활용했으며, 학습 시에 check point와 hf 모델 저장을 같이 합니다.

```bash
python -m torch.distributed.run --nproc_per_node=2 train.py \
    --pre_trained_model "KETI-AIR/ke-t5-base"\
    --task "klue_re" \
    --batch_size 16 \
    --epochs 30
```

task = ['klue_re', 'klue_ner', 'nikl_cr']

## Test

app/app.py를 활용하여 flask를 통해 각 task별로 test를 실행할 수 있습니다. \
학습한 모델이 없는 경우, 사전에 학습한 모델을 자동으로 다운받아 실행합니다. (각각 ke-t5-base모델, 30epoch 학습) \

```bash
python app/app.py
```

## Evaluation

### 정량적 평가

| task | base model | f1 score  |
| --- | --- | --- |
| Relation Extraction | ke-t5-base | 70.56  |
| Named Entity Recognition | ke-t5-base | 96.93  |
| Coreference Resolution | ke-t5-base | 99.29  |
|  |  |  |

### 정성적 평가

Relation Extraction

Text: 제2총군은 태평양 전쟁 말기에 일본 본토에 상륙하려는 연합군에게 대항하기 위해 설립된 일본 제국 육군의 총군이었다.

```json
{"result":
    [{
        "subject": "제2총군",
        "relation": "org:member_of",
        "object": "일본 제국 육군"
    }]
}
```

Text: 안중근은 이토 히로부미를 암살하여 러시아 헌병에게 붙잡혔고 1910년 3월 26일 오전 10시에 살인의 죄형으로 관동주 뤼순형무소에서 교수형으로 순국하였다.

```json
{"result":
    [{
        "subject": "안중근",
        "relation": "per:date_of_death",
        "object": "1910년 3월 26일 오전 10시"
    }]
}
```

Text: 지난달 31일 서울 광진구에서 현빈과 손예진이 결혼시을 올렸으며 현재 허니문 여행을 즐기고 있다.

```json
{"result":
    [{
        "subject": "현빈",
        "relation": "per:spouse",
        "object": "손예진"
    }]
}
```

Text: 삼성전자는 KDDI, NTT 도코모 등 통신사를 통해 어제(21일) 갤럭시S22 시리즈를 일본에 공식 출시했고, 출시에 앞서 이달 7일부터 20일까지 2주간 사전 판매를 진행했다.

```json
{"result":
    [{
        "subject": "삼성전자",
        "relation": "org:product",
        "object": "갤럭시S22"
    }]
}
```

Named Entity Recognition

Text: 대전시는 올해 지역산업 맞춤형 일자리 창출 지원 사업에 국비 33억원을 확보했다고 30일 발표했다.

```json
{"result":
    {"text": "(OG:대전시)는 (DT:올해) 지역산업 맞춤형 일자리 창출 지원 사업에 국비 (QT:33억원을) 확보했다고 (DT:30일) 발표했다."}
}
```

Coreference Resolution

Text: 김인철 부총리 겸 교육부 장관 후보자가 배우자가 서울예술고 강사로 근무한 기간에 같은 학교 비상임이사로 재직한 것으로 확인됐다. 당시 김 후보자는 기간제 교사 임용 등 학교 주요안건을 결정하는 과정에 참여해 ‘이해충돌’ 아니냐는 지적이 나온다. 25일 강민정 더불어민주당 의원이 서울시교육청으로부터 받은 자료에 따르면 김 후보자의 배우자 이모씨(60)는 2003년부터 2020년 말까지 2018년을 제외하고 서울예술고에서 실기 ‘성악’ 강의를 했다. 2003년 3월부터 2010년 2월까지 수업시수는 ‘자료없음’으로 제출했고, 2010년 3월부터 2020년 12월말까지는 주1~2시간 수업했다. 이씨가 서울예술고에서 받은 급여는 연간 95만원에서 200만원 정도로 파악됐다.

```json
{"result":
    [
        {"text": "text: (#김인철 부총리 겸 교육부 장관 후보자가) 배우자가 서울예술고 강사로 근무한 기간에 같은 학교 비상임이사로 재직한 것으로 확인됐다. 당시 (#김 후보자는) 기간제 교사 임용 등 학교 주요안건을 결정하는 과정에 참여해 ‘이해충돌’ 아니냐는 지적이 나온다. 25일 강민정 더불어민주당 의원이 서울시교육청으로부터 받은 자료에 따르면 (#김 후보자의) 배우자 이모씨(60)는 2003년부터 2020년 말까지 2018년을 제외하고 서울예술고에서 실기 ‘성악’ 강의를 했다. 2003년 3월부터 2010년 2월까지 수업시수는 ‘자료없음’으로 제출했고, 2010년 3월부터 2020년 12월말까지는 주1~2시간 수업했다. 이씨가 서울예술고에서 받은 급여는 연간 95만원에서 200만원 정도로 파악됐다."}, \
        {'text': 'text: 김인철 부총리 겸 교육부 장관 후보자가 (#배우자)가 서울예술고 강사로 근무한 기간에 같은 학교 비상임이사로 재직한 것으로 확인됐다. 당시 김 후보자는 기간제 교사 임용 등 학교 주요안건을 결정하는 과정에 참여해 ‘이해충돌’ 아니냐는 지적이 나온다. 25일 강민정 더불어민주당 의원이 서울시교육청으로부터 받은 자료에 따르면 (#김 후보자의 배우자 이모씨)(60)는 2003년부터 2020년 말까지 2018년을 제외하고 서울예술고에서 실기 ‘성악’ 강의를 했다. 2003년 3월부터 2010년 2월까지 수업시수는 ‘자료없음’으로 제출했고, 2010년 3월부터 2020년 12월말까지는 주1~2시간 수업했다. (#이씨가) 서울예술고에서 받은 급여는 연간 95만원에서 200만원 정도로 파악됐다.'}
    ]
}
```

## Acknowledgement
본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다. (정보통신기획평가원, 2022-0-00320, 상황인지 및 사용자 이해를 통한 인공지능 기반 1:1 복합대화 기술 개발)
