from datasets import load_dataset
import torch
import re
import json
import copy
import numpy as np
import functools
import random
import os
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from utils.meta import _KLUE_NER_TAGS, _KLUE_NER_IOB2_TAGS
from utils.download import download_file_from_google_drive

_DEFAULT_SPAN_TAGS = ['O', 'B', 'I']

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

class KlueREDataset(object):
    def __init__(self, tokenizer, split="train") -> None:
        self.tokenizer = tokenizer
        self.split = split
        self.data = load_dataset("klue", "re")[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        item = self.data[index]
        sample = self.re_preproc_for_classification_with_idx(item)
        tk_idxs = self.tokenize_re_with_tk_idx(sample)

        return {
            #"item": item,
            "id": sample["id"],
            "input_ids": tk_idxs["input_ids"],
            "attention_mask": tk_idxs["attention_mask"],
            "labels": torch.tensor(sample["targets"]),
            "subject_token_idx": tk_idxs["subject_token_idx"],
            "object_token_idx": tk_idxs["object_token_idx"]
        }

    def re_preproc_for_classification_with_idx(
        self,
        x,
        benchmark_name="klue_re",
        label_names=None,
        no_label_idx=0,
        with_feature_key=False,
        sep=' '):
        # mark span using start index of the entity
        def _mark_span(text, span_str, span_idx, mark):
            pattern_tmpl = r'^((?:[\S\s]){N})(W)'
            pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
            pattern = pattern_tmpl.replace('W', span_str)
            return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

        def _mark_span_sub(text, start_idx, end_idx, mark):
            end_idx += 2
            text = text[:start_idx] + mark + text[start_idx:]
            text = text[:end_idx] + mark + text[end_idx:]
            return text

        # '*' for subejct entity '#' for object entity
        text = x["sentence"]

        text = _mark_span_sub(text,
                              x['subject_entity']['start_idx'],
                              x['subject_entity']['end_idx'],
                              '*')

        sbj_st, sbj_end = x['subject_entity']['start_idx'], x['subject_entity']['end_idx']
        obj_st, obj_end = x['object_entity']['start_idx'], x['object_entity']['end_idx']
        sbj_end += 3
        obj_end += 3
        if sbj_st < obj_st:
            obj_st += 2
            obj_end += 2
        else:
            sbj_st += 2
            sbj_end += 2

        # Compensate for 2 added "words" added in previous step
        span2st = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
        span2et = x['object_entity']['end_idx'] + 2 * (1 if x['subject_entity']['end_idx'] < x['object_entity']['end_idx'] else 0)
        text = _mark_span_sub(text, span2st, span2et, '#')

        strs_to_join = []
        if with_feature_key:
            strs_to_join.append('{}:'.format('text'))
        strs_to_join.append(text)

        ex = {}

        if label_names is not None:
            # put the name of benchmark if the model is generative
            strs_to_join.insert(0, benchmark_name)
            ex['targets'] = label_names[x['label']] if x['label'] >= 0 else '<unk>'
        else:
            ex['targets'] = x['label'] if x['label'] >= 0 else no_label_idx

        offset = len(sep.join(strs_to_join[:-1] +['']))
        sbj_st+=offset
        sbj_end+=offset
        obj_st+=offset
        obj_end+=offset

        ex['subject_entity'] = {
            "type": x['subject_entity']['type'],
            "start_idx": sbj_st,
            "end_idx": sbj_end,
            "word": x['subject_entity']['word'],
        }
        ex['object_entity'] = {
            "type": x['object_entity']['type'],
            "start_idx": obj_st,
            "end_idx": obj_end,
            "word": x['object_entity']['word'],
        }

        joined = sep.join(strs_to_join)
        ex['inputs'] = joined
        ex['id'] = x['guid']

        return ex

    def tokenize_re_with_tk_idx(self, x, input_key='inputs'):
        ret = {}

        inputs = x[input_key]
        ret[f'{input_key}_pretokenized'] = inputs
        input_hf = self.tokenizer(inputs, padding=True, truncation='longest_first', return_tensors='pt')
        input_ids = input_hf.input_ids
        attention_mask = input_hf.attention_mask

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

        subject_token_idx = torch.zeros_like(input_ids)
        object_token_idx = torch.zeros_like(input_ids)
        subject_token_idx[0, subject_start:subject_end] = 1
        object_token_idx[0, object_start:object_end] = 1

        ret['subject_token_idx'] = subject_token_idx
        ret['object_token_idx'] = object_token_idx
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask

        return ret

    def get_collate_fn(self):
        def collate_fn(batch, pad_id):
            if len(batch) == 0:
                return None

            collated_batch = {
                "input_ids": collate_tokens([ex["input_ids"] for ex in batch], pad_id),
                "attention_mask": collate_tokens([ex["attention_mask"] for ex in batch], 0),
                "labels" : default_collate([ex["labels"] for ex in batch]),
                "subject_token_idx": collate_tokens([ex["subject_token_idx"] for ex in batch], 0),
                "object_token_idx": collate_tokens([ex["object_token_idx"] for ex in batch], 0),
            }

            return collated_batch

        return functools.partial(collate_fn, pad_id=self.tokenizer.pad_token_id)

class KlueNERDataset(object):
    def __init__(self, tokenizer, split="train") -> None:
        self.tokenizer = tokenizer
        self.split = split
        self.data = load_dataset("klue", "ner")[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        item = self.preproc_labels(self.data[index], _KLUE_NER_IOB2_TAGS)
        sample = self.tokenize_and_preproc_iob24klue(
            item,
            tags=_KLUE_NER_TAGS,
            iob2_tags=_KLUE_NER_IOB2_TAGS)

        return {
            "id": str(index),
            "sentence": item["sentence"],
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": torch.tensor(sample["targets"]).to(torch.long),
        }

    def preproc_labels(self, x, ibo2_tags):
        ret = []
        tag_len = 0
        for idx, tag_idx in enumerate(x["ner_tags"]):
            if ibo2_tags[tag_idx] != "O":
                if ibo2_tags[tag_idx].split("-")[0] == "B":
                    begin = idx
                    tag = ibo2_tags[tag_idx].split("-")[-1]
                    tag_len += 1
                elif ibo2_tags[tag_idx].split("-")[0] == "I":
                    tag_len += 1
            elif tag_len > 0:
                ret.append({
                    "begin": begin,
                    "end": begin + tag_len,
                    "label" : tag
                })
                tag_len = 0

        x["NE"] = ret
        return x
                
    def tokenize_and_preproc_iob24klue(
            self,
            x,
            tags=None,
            iob2_tags=None,
            tag_label='NE',
            input_key='inputs',
            info4klue=True):

        ret = {}

        inputs = "".join(x["tokens"])
        ret[f'{input_key}_pretokenized'] = inputs
        input_hf = self.tokenizer(inputs, padding=True, truncation='longest_first', return_tensors='pt')
        input_ids = input_hf.input_ids
        attention_mask = input_hf.attention_mask

        if info4klue:
            ret['klue_metric'] = {}
            ret['klue_metric']['char_to_token'] = [
                input_hf.char_to_token(pos) for pos in range(len(inputs))]

        ret["input_ids"] = input_ids
        ret["attention_mask"] = attention_mask

        if tags and iob2_tags:
            outside_label = iob2_tags.index('O')
            tag_labels = x[tag_label]
            labels = np.ones_like(input_ids[0], dtype=np.int32) * outside_label

            if info4klue:
                ret['klue_metric']['char_tag'] = np.ones_like(
                    ret['klue_metric']['char_to_token'], dtype=np.int32) * outside_label

            for tgl in tag_labels:
                label_txt = tgl['label']
                if label_txt != 'O':

                    if info4klue:
                        for idx, pos_idx in enumerate(list(range(tgl['begin'], tgl['end']))):
                            if idx == 0:
                                ret['klue_metric']['char_tag'][pos_idx] = iob2_tags.index(
                                    'B-'+label_txt)
                            else:
                                try:
                                    ret['klue_metric']['char_tag'][pos_idx] = iob2_tags.index(
                                    'I-'+label_txt)
                                except:
                                    ret['klue_metric']['char_tag'][len
                                    (ret['klue_metric']['char_tag'])-1] = iob2_tags.index(
                                    'I-'+label_txt)

                    pos_list = [input_hf.char_to_token(
                        pos) for pos in range(tgl['begin'], tgl['end'])]
                    #pos_list = copy.deepcopy(char_to_token[begin:end])

                    # there is  None position in the case consecutive white spaces.
                    pos_list = [x for x in pos_list if x is not None]
                    token_set = set(pos_list)
                    token_set_order = sorted(list(token_set))

                    for iter_idx, tk_idx in enumerate(token_set_order):
                        if iter_idx == 0:
                            labels[tk_idx] = iob2_tags.index('B-'+label_txt)
                        else:
                            labels[tk_idx] = iob2_tags.index('I-'+label_txt)

            ret['targets'] = labels

        return ret


    def get_collate_fn(self):
        def collate_fn(batch, pad_id):
            if len(batch) == 0:
                return None

            collated_batch = {
                "input_ids": collate_tokens([ex["input_ids"] for ex in batch], pad_id),
                "attention_mask": collate_tokens([ex["attention_mask"] for ex in batch], 0),
                "labels" : collate_tokens([ex["labels"] for ex in batch], -1)
            }

            return collated_batch

        return functools.partial(collate_fn, pad_id=self.tokenizer.pad_token_id)


NIKL_CR_DATASETS = ["data/NIKL_CR/NXCR1902103160.json", "data/NIKL_CR/SXCR1902103160.json"]
NIKL_CR_GOOGLE_DRIVE_ID = "1VBcsKmaofd2PRTdeQkhQJQB4yrNB5Lns"

class NIKLCRDataset(object):
    def __init__(self, tokenizer, split="train") -> None:
        self.tokenizer = tokenizer
        for file_name in NIKL_CR_DATASETS:
            if not os.path.isfile(file_name):
                download_file_from_google_drive(NIKL_CR_GOOGLE_DRIVE_ID, "./data.tar")
            with open (file_name, "r", encoding="utf-8-sig") as json_file:
                json_data = json.load(json_file)
            try:
                keys = list(obj.keys())
                ret = self.parsing_cr(json_data)
                for k in keys:
                    obj[k] += ret[k]
            except:
                obj = self.parsing_cr(json_data)

        data = self.create_doc_span(obj)
        data = self.create_cr_example(data)

        random.shuffle(data)
        sp = int(len(data)/10)
        if split == 'train':
            self.data = data[sp:]
        else:
            self.data = data[:sp]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        item = self.data[index]

        return {
            "id": item["id"],
            "inputs_pretokenized": item["inputs_pretokenized"],
            "form": item["form"],
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "entity_token_idx": item["entity_token_idx"],
            "labels": torch.tensor(item["targets"]).to(torch.long),
        }

    def parsing_cr(self, x):
        ret = {
            'id': [],
            'inputs': [],
            'CR': [],
        }

        for doc in x['document']:
            _id = doc['id']
            _sentence = doc['sentence']
            sent_dict = {}
            _offset = 0
            sents = []
            for v in _sentence:
                sents.append(v['form'])

                sent_dict[v["id"]] = {
                    "form": v["form"],
                    "len": len(v["form"]),
                    "offset": _offset,
                }
                _offset += len(v["form"]) + 1         

            _cr = doc['CR']

            text = ' '.join(sents)
            cr = {'mention': []}
            for _ments in _cr:
                # cr = {'mention': []}
                ments = {
                    'form': [],
                    'begin': [],
                    'end': []
                }
                for _ment in _ments["mention"]:
                    _sentid = _ment['sentence_id']
                    ments['form'].append(_ment['form'])
                    ments['begin'].append(_ment['begin'] + sent_dict[_sentid]['offset'])
                    ments['end'].append(_ment['end'] + sent_dict[_sentid]['offset'])
                cr['mention'].append(ments)
                
            ret['id'].append(_id)
            ret['inputs'].append(text)
            ret['CR'].append(cr)


        return ret

    def create_doc_span(self,
                    x,  
                    span_length=448, 
                    doc_stride=128, 
                    input_key='inputs'):

        ret = {
            'id': [],
            'span_ch_start': [],
            'span_ch_end': [],
            input_key: [],
            'CR': []
        }

        batch_len = len(x[input_key])

        for ex_idx in tqdm(range(batch_len)):
            inputs = x[input_key][ex_idx]
            input_hf = self.tokenizer(inputs, add_special_tokens=False)
            input_ids = input_hf.input_ids

            doc_spans = []
            start_offset = 0
            while start_offset < len(input_ids):
                length = len(input_ids) - start_offset
                if length > span_length:
                    length = span_length
                doc_spans.append((start_offset, length))
                if start_offset + length == len(input_ids):
                    break
                start_offset += min(length, doc_stride)

            for doc_idx, doc_span in enumerate(doc_spans):
                doc_tk_st, doc_tk_ed = doc_span[0], doc_span[0] + doc_span[1] - 1

                doc_ch_st = input_hf.token_to_chars(doc_tk_st).start
                doc_ch_ed = input_hf.token_to_chars(doc_tk_ed).end
                
                ret['id'].append('{}:{}'.format(x['id'][ex_idx], doc_idx))
                ret['span_ch_start'].append(doc_ch_st)
                ret['span_ch_end'].append(doc_ch_ed)
                ret[input_key].append(inputs[doc_ch_st:doc_ch_ed])
                ret['CR'].append(copy.deepcopy(x['CR'][ex_idx]))

        return ret

    def mention_marking(slef, text, mention, mention_cands, mention_marker='*'):
        def _mark_span(text, span_st, span_ed, mark):
            return text[:span_st] + mark + text[span_st:span_ed] + mark + text[span_ed:]
        offset_st = mention[0]
        offset_ed = mention[1]
        text = _mark_span(text, offset_st, offset_ed, mention_marker)
        target_span = [offset_st, offset_ed+2]
        mention_cands_new = []
        for x, y, z in mention_cands:
            if x > offset_ed:
                mention_cands_new.append((x+2, y+2, z))
            elif y == offset_ed:
                # char span (include marker)``
                mention_cands_new.append((x, y+2, z))
            else:
                mention_cands_new.append((x, y, z))

        return text, mention_cands_new, target_span

    def create_labels_for_mention(slef, text, mention_cands, tokenizer, iob2_tags=_DEFAULT_SPAN_TAGS):
        input_hf = tokenizer(text, add_special_tokens=False, padding=True, truncation='longest_first', return_tensors='pt')
        input_ids = input_hf.input_ids
        attention_mask = input_hf.attention_mask

        outside_label = iob2_tags.index('O')
        labels = np.ones_like(input_ids[0], dtype=np.int32) * outside_label

        label_txt = []
        for begin, end, form in mention_cands:
            pos_list = [input_hf.char_to_token(
                pos) for pos in range(begin, end)]
            # there is  None position in the case consecutive white spaces.
            pos_list = [x for x in pos_list if x is not None]
            token_set_order = sorted(list(set(pos_list)))
            l_txt = tokenizer.decode([input_ids[0][x] for x in token_set_order])

            label_txt.append(l_txt)

            for iter_idx, tk_idx in enumerate(token_set_order):
                if iter_idx == 0:
                    labels[tk_idx] = iob2_tags.index('B')
                else:
                    labels[tk_idx] = iob2_tags.index('I')
        
        return input_ids, attention_mask, labels, label_txt

    def create_cr_example(self,
                          x, 
                          exclude_src_span=False,
                          iob2_tags=_DEFAULT_SPAN_TAGS,
                          input_key='inputs'):
        
        ret = []

        batch_len = len(x[input_key])

        for ex_idx in tqdm(range(batch_len)):
            inputs = x[input_key][ex_idx]

            offset = 0
            if 'span_ch_start' in x:
                offset = x['span_ch_start'][ex_idx]
            end_offset = len(inputs)
            if 'span_ch_end' in x:
                end_offset = x['span_ch_end'][ex_idx]
            
            for mention in x['CR'][ex_idx]['mention']:
                valid_mensions = [(begin-offset, end-offset, form) 
                    for begin, end, form 
                    in zip(mention['begin'], mention['end'], mention['form'])
                    if offset < begin and end < end_offset]
                
                for mention_idx in range(len(valid_mensions)):
                    mention_tgt = valid_mensions[mention_idx]

                    if exclude_src_span:
                        mention_cands = [valid_mensions[i] for i in range(len(valid_mensions)) if i != mention_idx]
                    else:
                        mention_cands = valid_mensions

                    inputs_aug, mention_cand_aug, target_span = self.mention_marking(inputs, mention_tgt, mention_cands)
                    input_ids_aug, attention_mask_aug, labels_aug, label_txt = self.create_labels_for_mention(inputs_aug, mention_cand_aug, self.tokenizer)

                    ret.append({
                        'id': '{}:{}'.format(x['id'][ex_idx], mention_idx),
                        'input_ids': input_ids_aug,
                        'attention_mask': attention_mask_aug,
                        f'{input_key}_pretokenized': inputs_aug,
                        'input_key': input_ids_aug,
                        'form': label_txt,
                        'entity_token_idx': torch.tensor(target_span),
                        'targets': labels_aug
                    })

        return ret

    def get_collate_fn(self):
        def collate_fn(batch, pad_id):
            if len(batch) == 0:
                return None

            collated_batch = {
                "input_ids": collate_tokens([ex["input_ids"] for ex in batch], pad_id),
                "attention_mask": collate_tokens([ex["attention_mask"] for ex in batch], 0),
                "labels" : collate_tokens([ex["labels"] for ex in batch], -1),
                #"entity_token_idx" : default_collate([ex["entity_token_idx"] for ex in batch])
            }

            return collated_batch

        return functools.partial(collate_fn, pad_id=self.tokenizer.pad_token_id)

if __name__ == "__main__":

    from transformers import AutoTokenizer

    tokenizer_name = "KETI-AIR/ke-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = NIKLCRDataset(tokenizer, split="train")

    print("dataset len: {}".format(len(train_ds)))

    for idx, ex in zip(range(50), train_ds):
        for k, v in ex.items():
            print("{}: {}".format(k, v))
        print("\n\n")

