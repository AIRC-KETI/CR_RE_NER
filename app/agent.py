import json
import sys
import os
import regex as re
import numpy as np

import torch

from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.meta import _KLUE_RE_RELATIONS, _KLUE_NER_IOB2_TAGS, _DEFAULT_SPAN_TAGS
from models.models import *
from models.models import load_model

class Service:
    task = [
        {
            'name': [
                "relation_extraction",
                "named_entity_recognition",
                "coreference_resolution"
            ],
            'description': ''
        }
    ]

    def __init__(self):
        self.service = MainService()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.service.do_service(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400

class MainService(object):
    def __init__(self):
        self.config = json.load(open("config.json", "r"))

        self.service_manager = {
            "relation_extraction": {
                "model": self.load_service_manager(self.config["relation_extraction"]),
                "service": self.extract
            },
            "named_entity_recognition": {
                "model": self.load_service_manager(self.config["named_entity_recognition"]),
                "service": self.recognize
            },
            "coreference_resolution": {
                "model": self.load_service_manager(self.config["coreference_resolution"]),
                "service": self.resolve
            }
        }
        pre_trained_tokenizer = self.config.get("pre_trained_tokenizer", "KETI-AIR/ke-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_tokenizer)

    @staticmethod
    def load_service_manager(model_cfg):
        model_class = load_model(model_cfg["model_name"])
        model = model_class.from_pretrained(model_cfg["hf_path"])
        model.eval()

        return model

    def do_service(self, content):
        task = content.get('task', None)
        doc = content.get('doc', None)
        arg_pairs = content.get('arg_pairs', None)
        doc_text = doc.get('text', '')
        if task is None:
            return {
                'error': "Please select a task. \
                    (relation_extraction, named_entity_recognition, coreference_resolution)"
            }
        elif task not in self.service_manager.keys():
            return {
                    'error': f"The requested tasks are currently unsupported. \
                        (relation_extraction, named_entity_recognition, coreference_resolution) \
                             is currently supported on this service. \
                            But got ({task})"
                }
        elif doc is None:
            return {
                'error': "There is no document."
            }
        elif task == "relation_extraction":
            if arg_pairs is None:
                return {
                    'error': "You have to pass argument pairs. But got Null argument."
                }
            else:
                if doc_text == '':
                    return {
                        'error': "Empty document string."
                    }
                doc_dict = self.convert_dict_for_re(doc_text, arg_pairs)
                result = self.service_manager[task]['service'](task, doc_dict)
                return result
        elif task == "coreference_resolution":
            if arg_pairs is None:
                return {
                    'error': "You have to pass argument pairs. But got Null argument."
                }
            else:
                if doc_text == '':
                    return {
                        'error': "Empty document string."
                    }
                doc_dict = self.convert_dict_for_cr(doc_text, arg_pairs)
                result = self.service_manager[task]['service'](task, doc_dict)
                return result
        else:
            if doc_text == '':
                return {
                    'error': "Empty document string."
                }

            result = self.service_manager[task]['service'](task, doc_text)
            return result


    @staticmethod
    def convert_dict_for_re(text, arg_pairs):
        re_dict = list()
        for arg in arg_pairs:
            re_dict.append({
                "sentence": text,
                "subject_entity": {
                    "word": text[arg[0][0]:arg[0][1]+1],
                    "start_idx": arg[0][0],
                    "end_idx": arg[0][1]
                },
                "object_entity": {
                    "word": text[arg[1][0]:arg[1][1]+1],
                    "start_idx": arg[1][0],
                    "end_idx": arg[1][1]
                }
            })
        return re_dict

    @staticmethod
    def re_preproc_for_classification_with_idx(
            x,
            with_feature_key=True,
            sep=' '):
        # mark span using start index of the entity
        def _mark_span(text, span_str, span_idx, mark):
            pattern_tmpl = r'^((?:[\S\s]){N})(W)'
            pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
            pattern = pattern_tmpl.replace('W', span_str)
            return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

        # '*' for subejct entity '#' for object entity.

        text = x["sentence"]
        text = _mark_span(text, x['subject_entity']['word'],
                        x['subject_entity']['start_idx'], '*')

        sbj_st, sbj_end, sbj_form = x['subject_entity']['start_idx'], x['subject_entity']['end_idx'], x['subject_entity']['word']
        obj_st, obj_end, obj_form = x['object_entity']['start_idx'], x['object_entity']['end_idx'], x['object_entity']['word']
        sbj_end += 2
        obj_end += 2
        if sbj_st < obj_st:
            obj_st += 2
            obj_end += 2
        else:
            sbj_st += 2
            sbj_end += 2

        # Compensate for 2 added "words" added in previous step.
        span2_index = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
        text = _mark_span(text, x['object_entity']['word'], span2_index, '#')

        strs_to_join = []
        if with_feature_key:
            strs_to_join.append('{}:'.format('text'))
        strs_to_join.append(text)

        ex = {}

        offset = len(sep.join(strs_to_join[:-1] +['']))
        sbj_st+=offset
        sbj_end+=offset
        obj_st+=offset
        obj_end+=offset

        ex['subject_entity'] = {
            "start_idx": sbj_st,
            "end_idx": sbj_end,
            "word": x['subject_entity']['word'],
        }
        ex['object_entity'] = {
            "start_idx": obj_st,
            "end_idx": obj_end,
            "word": x['object_entity']['word'],
        }

        joined = sep.join(strs_to_join)
        ex['inputs'] = joined

        return ex

    @staticmethod
    def tokenize_re_with_tk_idx(x, tokenizer, input_key='inputs'):
        ret = {}

        inputs = x[input_key]
        ret[f'{input_key}_pretokenized'] = inputs
        input_hf = tokenizer(inputs)
        input_ids = input_hf.input_ids

        subject_entity = x['subject_entity']
        object_entity = x['object_entity']

        subject_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                subject_entity['start_idx'], 
                subject_entity['end_idx']
                )
            ]
        subject_tk_idx = [x for x in subject_tk_idx if x is not None]
        subject_tk_idx = sorted(set(subject_tk_idx))
        subject_start = subject_tk_idx[0]
        subject_end = subject_tk_idx[-1]

        object_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                object_entity['start_idx'], 
                object_entity['end_idx']
                )
            ]
        object_tk_idx = [x for x in object_tk_idx if x is not None]
        object_tk_idx = sorted(set(object_tk_idx))
        object_start = object_tk_idx[0]
        object_end = object_tk_idx[-1]

        entity_token_idx = np.array([[subject_start, subject_end], [object_start, object_end]])
        ret['entity_token_idx'] = np.expand_dims(entity_token_idx, axis=0)
        ret['inputs'] = tokenizer(inputs, return_tensors='pt').input_ids
        return ret

    def get_doc_pos(self, doc_text, inputs, pos, offset, tag_len=None):
        if tag_len is None:
            doc_pos = inputs.token_to_chars(pos)[0]
            if doc_pos+offset > len(doc_text):
                return len(doc_text)
            if doc_text[doc_pos+offset] == " ":
                return doc_pos + offset + 1
            else:
                return doc_pos + offset
        else:
            return inputs.token_to_chars(pos+tag_len)[0] + offset

    @staticmethod
    def convert_dict_for_cr(text, arg_pairs):
        re_dict = list()
        for arg in arg_pairs:
            re_dict.append({
                "sentence": text,
                "subject_entity": {
                    "word": text[arg[0][0]:arg[0][1]+1],
                    "start_idx": arg[0][0],
                    "end_idx": arg[0][1]
                }
            })
        return re_dict

    @staticmethod
    def cr_preproc_for_classification_with_idx(
            x,
            with_feature_key=True,
            sep=' '):
        # mark span using start index of the entity
        def _mark_span(text, span_str, span_idx, mark):
            pattern_tmpl = r'^((?:[\S\s]){N})(W)'
            pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
            pattern = pattern_tmpl.replace('W', span_str)
            return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

        # '*' for subejct entity.
        text = x["sentence"]
        text = _mark_span(text, x['subject_entity']['word'],
                        x['subject_entity']['start_idx'], '*')

        strs_to_join = []
        if with_feature_key:
            strs_to_join.append('{}:'.format('text'))
        strs_to_join.append(text)

        joined = sep.join(strs_to_join)

        return joined

    @torch.no_grad()
    def extract(self, task, doc_dict):
        result = list()
        for doc in doc_dict:
            preproc_doc = self.re_preproc_for_classification_with_idx(doc)
            inputs = self.tokenize_re_with_tk_idx(preproc_doc, self.tokenizer)
            outputs = self.service_manager[task]['model'](input_ids=inputs['inputs'], entity_token_idx=inputs['entity_token_idx'])
            label = torch.argmax(outputs['logits'], 1).numpy()[0]
            result.append({
                "subject": preproc_doc['subject_entity']['word'],
                "relation": _KLUE_RE_RELATIONS[label],
                "object": preproc_doc['object_entity']['word']
            })

        return {
            "result": result
        }

    @torch.no_grad()
    def recognize(self, task, doc_text):
        inputs = self.tokenizer(doc_text, return_tensors='pt')
        outputs = self.service_manager[task]['model'](input_ids=inputs.input_ids)
        labels = outputs['logits'][0]

        offset = -5
        tag_len = 0
        for pos, label in enumerate(labels):
            if label != _KLUE_NER_IOB2_TAGS.index('O'):
                if _KLUE_NER_IOB2_TAGS[label].split('-')[0] == 'B':
                    begin = pos
                    tag = _KLUE_NER_IOB2_TAGS[label].split('-')[-1]
                    offset += 5
                    tag_len += 1
                elif _KLUE_NER_IOB2_TAGS[label].split('-')[0] == 'I':
                    tag_len += 1
            elif tag_len > 0:
                doc_text = (doc_text[:self.get_doc_pos(doc_text, inputs, begin, offset)] +
                            "({}:".format(tag) +
                            doc_text[self.get_doc_pos(doc_text, inputs, begin, offset):
                            self.get_doc_pos(doc_text, inputs, begin, offset, tag_len)] +
                            ")" + doc_text[self.get_doc_pos(doc_text, inputs, begin, offset, tag_len):])
                tag_len = 0

        return {
            "result": {
                "text": doc_text
            }
        }

    @torch.no_grad()
    def resolve(self, task, doc_dict):
        result = list()
        for doc in doc_dict:
            doc_text = self.cr_preproc_for_classification_with_idx(doc)
            inputs = self.tokenizer(doc_text, return_tensors='pt')
            outputs = self.service_manager[task]['model'](input_ids=inputs.input_ids)
            labels = outputs['logits'][0]
            offset = -3
            tag_len = 0
            for pos, label in enumerate(labels):
                if label != _DEFAULT_SPAN_TAGS.index('O'):
                    if _DEFAULT_SPAN_TAGS[label] == 'B':
                        begin = pos
                        offset += 3
                        tag_len += 1
                    elif _DEFAULT_SPAN_TAGS[label] == 'I':
                        tag_len += 1
                elif tag_len > 0:
                    doc_text = (doc_text[:self.get_doc_pos(doc_text, inputs, begin, offset)] +
                                "(#" +
                                doc_text[self.get_doc_pos(doc_text, inputs, begin, offset):
                                self.get_doc_pos(doc_text, inputs, begin, offset, tag_len)] +
                                ")" + doc_text[self.get_doc_pos(doc_text, inputs, begin, offset, tag_len):])
                    tag_len = 0

            result.append({
                "text": doc_text
            })

        return {
            "result": result
        }


if __name__ == "__main__":

    test_model = MainService()

    # Relation Extraction
    example_re = [{
        "task": "relation_extraction",
        "doc":{
            "text":"제2총군은 태평양 전쟁 말기에 일본 본토에 상륙하려는 연합군에게 대항하기 위해 설립된 일본 제국 육군의 총군이었다."
        },
        "arg_pairs":[
            [
                [0,3],
                [48,55]
            ]
        ]
    },
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
    },
    {
        "task": "relation_extraction",
        "doc":{
            "text":"이재용 삼성전자 부회장의 회장 취임 시점에 재계의 관심이 집중되고 있다."
        },
        "arg_pairs":[
            [
                [0,2],
                [5,8]
            ]
        ]
    },
    {
        "task": "relation_extraction",
        "doc":{
            "text":"지난달 31일 서울 광진구에서 현빈과 손예진이 결혼시을 올렸으며 현재 허니문 여행을 즐기고 있다."
        },
        "arg_pairs":[
            [
                [17,18],
                [21,23]
            ]
        ]
    },
    {
        "task": "relation_extraction",
        "doc":{
            "text":"삼성전자는 KDDI, NTT 도코모 등 통신사를 통해 어제(21일) 갤럭시S22 시리즈를 일본에 공식 출시했고, 출시에 앞서 이달 7일부터 20일까지 2주간 사전 판매를 진행했다."
        },
        "arg_pairs":[
            [
                [0,3],
                [38,43]
            ]
        ]
    }]
    print("# Relation Extraction #")
    for i in range(len(example_re)):
        print("Text {}: {}".format(i+1, example_re[i]["doc"]["text"]))
        re_ret = test_model.do_service(example_re[i])
        print(re_ret)

    # Named Entity Recognition
    example_ner = [{
        "task": "named_entity_recognition",
        "doc":{
            "text":"대전시는 올해 지역산업 맞춤형 일자리 창출 지원 사업에 국비 33억원을 확보했다고 30일 발표했다."
        }
    },
    {
        "task": "named_entity_recognition",
        "doc":{
            "text":"우리나라 공정거래위원회의 조사 절차가 미국, 유럽에 비해 강제성이 지나치게 높다며, 기업 등 피심인의 권리를 보장하는 법적 장치를 마련해야 한다는 주장이 제시됐다."
        }
    },
    {
        "task": "named_entity_recognition",
        "doc":{
            "text":"검거 직후 변호사 선임을 요구하며 진술을 거부하던 ‘계곡 살인’ 사건 피의자 이은해(31)씨가 구속 후 진행된 검찰 조사에서 국선변호인의 도움을 거부한 것으로 확인됐다."
        }
    },
    {
        "task": "named_entity_recognition",
        "doc":{
            "text":"손예진 부부는 지난달 31일 서울 광진구 워커힐 호텔 애스톤 하우스에서 결혼식을 올렸다."
        }
    }]
    print("\n# Named Entity Recognition #")
    for i in range(len(example_ner)):
        print("Text {}: {}".format(i+1, example_ner[i]["doc"]["text"]))
        ner_ret = test_model.do_service(example_ner[i])
        print(ner_ret)
    
    # Coreference Resolution
    example_cr = [{
        "task": "coreference_resolution",
        "doc":{
            "text":"식약처가 철저하게 관리하고 있는 ‘화장품 안전기준’에 따르면 THB는 염모제에 사용이 허용되는 합법적인 성분이다. 모다모다가 THB를 발색 샴푸에 처음 사용한 것도 아니다. 사실 THB는 염색약의 색을 조절하는 용도로 1950년 대에도 유럽에서 사용되어왔던 전통적인 성분이다. 1961년 미국 염모제 특허에도 THB가 등장한다. 실제로 THB는 국제화장품성분사전에 등재되어 널리 활용되고 있는 범용 염모제 성분이다."
        },
        "arg_pairs":[
            [
                [34,36]
            ]
        ]
    },
    {
        "task": "coreference_resolution",
        "doc":{
            "text":"김인철 부총리 겸 교육부 장관 후보자가 배우자가 서울예술고 강사로 근무한 기간에 같은 학교 비상임이사로 재직한 것으로 확인됐다. 당시 김 후보자는 기간제 교사 임용 등 학교 주요안건을 결정하는 과정에 참여해 ‘이해충돌’ 아니냐는 지적이 나온다. 25일 강민정 더불어민주당 의원이 서울시교육청으로부터 받은 자료에 따르면 김 후보자의 배우자 이모씨(60)는 2003년부터 2020년 말까지 2018년을 제외하고 서울예술고에서 실기 ‘성악’ 강의를 했다. 2003년 3월부터 2010년 2월까지 수업시수는 ‘자료없음’으로 제출했고, 2010년 3월부터 2020년 12월말까지는 주1~2시간 수업했다. 이씨가 서울예술고에서 받은 급여는 연간 95만원에서 200만원 정도로 파악됐다."
        },
        "arg_pairs":[
            [
                [0,2]
            ],
            [
                [178,187]
            ]
        ]
    },
    {
        "task": "coreference_resolution",
        "doc":{
            "text":"오리온 “1승만 더”…챔프전 눈앞 프로농구 PO 모비스전 2연승 지난 6일 프로농구 4강 플레이오프 미디어데이에서 오리온의 추일승 감독은 “유재학 감독은 이제 식상하다. 시청자들이 유 감독이 나오면 채널을 돌린다. 양보할 때가 됐다”며 모비스의 심기를 건드렸다. 지난해까지 챔피언결정전 3연패를 이끌며 선수 때와 마찬가지로 지도자로서 이미 입지를 단단히 굳힌 유재학 감독은 여유롭게 “나도 그러길 바란다”고 응수했다. 나이도 같고 선수 시절 동기로 나란히 한 팀에서 활약했던 두 감독의 웃음을 띤 미묘한 신경전이었다. 하지만 대다수의 전문가들은 유 감독의 노련미와 모비스의 탄탄한 조직력에 더 무게를 뒀다. 판을 열어보니 결과는 정반대였다. 오리온의 2연승이다. 고양 오리온이 10일 울산 동천체육관에서 열린 2015~2016 프로농구 4강 플레이오프(5전3선승) 2차전에서 외국인 듀오 조 잭슨(25점)과 애런 헤인즈(18점·8튄공)의 43점 합작 맹활약을 앞세워 울산 모비스를 62-59로 제압했다. 이로써 1·2차전을 모두 잡은 오리온은 챔피언결정전 진출에 단 1승만 남겼다. 13년 만이다. 이날 모비스는 팀의 중심인 양동근(8점·6도움·6튄공)도 평소만큼의 활약을 보여주지 못하며 반격에 실패했다."
        },
        "arg_pairs":[
            [
                [0,2]
            ],
            [
                [101,104]
            ]
        ]
    }]
    print("\n# Coreference Resolution #")
    for i in range(len(example_cr)):
        print("Text {}: {}".format(i+1, example_cr[i]["doc"]["text"]))
        cr_ret = test_model.do_service(example_cr[i])
        print(cr_ret)
