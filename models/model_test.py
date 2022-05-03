from sklearn.semi_supervised import LabelSpreading
from models import T5EncoderForCoreferenceResolutionFirstSubmeanObjmeanWithCRF, T5EncoderForSequenceClassificationFirstSubmeanObjmean, T5EncoderForEntityRecognitionWithCRF
from transformers import AutoTokenizer
import torch

pre_trained_model = "KETI-AIR/ke-t5-base"

model = T5EncoderForCoreferenceResolutionFirstSubmeanObjmeanWithCRF.from_pretrained(
    pre_trained_model, num_labels=3)

tokenize = AutoTokenizer.from_pretrained(pre_trained_model)

text = ["8년전 지구에 낙하한 혜성이 태양계 외부의 먼 외계에서 온 행성간 혜성인 것으로 연구자들이 판단하고 있다고 미국 우주사령부(USSC)가 최근 보고서에서 밝힌 것으로 미 CNN이 13일(현지시간) 보도했다.",
        "비록 실제 능력이나 자신감, 긍정적 태도가 없더라도 그런 척 하며 꾸준히 노력하면 언젠가 그렇게 될 수 있다는 조언이다. "]
input_hf = tokenize(text, add_special_tokens=False, padding=True, truncation='longest_first', return_tensors='pt')

input_ids = input_hf.input_ids
attention_mask = input_hf.attention_mask
label = torch.zeros_like(input_ids)
entity_token_idx = torch.tensor([[3,9],[5,8]])

print("label: {}".format(label))
print("input_hf: {}".format(input_hf))
print("entity_token_idx: {}".format(entity_token_idx))

outputs = model(input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label,
                entity_token_idx=entity_token_idx)

print("outputs: {}".format(outputs))

#################################################################3

# model = T5EncoderForEntityRecognitionWithCRF.from_pretrained(
#     pre_trained_model, num_labels=3)

# tokenize = AutoTokenizer.from_pretrained(pre_trained_model)

# text = "8년전 지구에 낙하한 혜성이 태양계 외부의 먼 외계에서 온 행성간 혜성인 것으로 연구자들이 판단하고 있다고 미국 우주사령부(USSC)가 최근 보고서에서 밝힌 것으로 미 CNN이 13일(현지시간) 보도했다."
# input_hf = tokenize(text, add_special_tokens=False, padding=True, truncation='longest_first', return_tensors='pt')

# input_ids = input_hf.input_ids
# attention_mask = input_hf.attention_mask
# label = torch.zeros_like(input_ids)

# print("label: {}".format(label))
# print("input_ids: {}".format(input_ids))
# print("attention_mask: {}".format(attention_mask))

# outputs = model(input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=label)

# print("outputs: {}".format(outputs))