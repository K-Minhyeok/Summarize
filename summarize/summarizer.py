import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer


## Role 부여 
# > 상대는 뉴스를 통해 시사 정보를 알고 싶어하는 대학생이고, 너는 뉴스를 요약해주는 사람이야
# > 몇 가지 요청사항에 기반해서 뉴스 본문을 주면, 200자 내외로 간단하게 요약해줘. 
# > 세부 요청사항은 아래와 같아.

## 세부 요청사항 프롬프팅
# > 1. 어려운 용어를 쉽게 이해할 수 있는 단어로 바꿔줘. 예제를 줘도 좋아
# > 2. 정리하는 format   # ex) 요약 내용(200자 내외): ~~~ --- 장점(100자 내외): ~~~  --- 단점(100자 내외) : ~~~ --- 이렇게 해줘

model_name = "beomi/Llama-3-Open-Ko-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
streamer = TextStreamer(tokenizer)


system_prompt = """
너는 뉴스를 요약해주는 어시스턴트이다.
요약 대상 독자는 시사 정보를 알고 싶어하는 대학생이다.

요약 시 다음 원칙을 반드시 지켜라:
- 어려운 정치·경제·사회 용어는 쉬운 말로 바꿔 설명한다.
- 필요하면 간단한 예시를 들어 이해를 돕는다.
- 핵심 정보만 간결하게 전달한다.
- 중립적인 관점에서 사실 위주로 서술한다.

출력은 반드시 지정된 형식을 그대로 사용하고,
다른 문장이나 설명을 추가하지 마라.
"""

# 크롤링된 내용을 요약하는 부분
def summarize_content(content: str):
    user_prompt = f"""
    아래 뉴스 기사를 요약해줘.

    요구 사항:
    1. 요약 내용은 200자 내외로 작성해줘.
    2. 아래 형식을 반드시 지켜줘.
    3. 각 항목은 문장 형태로 작성해줘.

    출력 형식:
    요약 내용(200자 내외):
    ---
    장점(100자 내외):
    ---
    단점(100자 내외):

    뉴스 기사:
    {content}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"  
    ).to(model.device)
    print("hit1")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    print("hit3")

    response = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

if __name__ == "__main__":
    print(summarize_content("""치사율이 100%인 아프리카돼지열병이 올해만 벌써 7건이 발생하는 등 전국으로 확산하고 있습니다. 최근 ASF 바이러스는 야생멧돼지에서 나온 유전형과 다른 바이러스여서 사람과 차량 등 인위적 요인에 따른 전파 가능성이 높다는 분석입니다.

이용식 기자가 취재했습니다.

방역요원과 중장비가 돼지농장 안으로 들어갑니다.

이곳에서 키우던 새끼돼지 폐사체에 이어 다른 돼지에서도 아프리카돼지열병이 지난 3일 확진됐습니다.

사육돼지 3천5백 마리는 살 처분 됐습니다.

[이근우/대한한돈협회 보령시지부장 : 지금 뭐 초비상사태죠. 매일 소독을 실시하고 있으며 출입차량을 다 통제하고 있습니다.]

특히 발생농장에서 5km 이내에 있는 국내 최대 돼지사육지역인 홍성의 경우 긴장이 더 높아지고 있습니다.

지난달 16일 강릉을 시작으로 올 들어 발생한 아프리카돼지열병은 벌써 7건, 2019년 이후 모두 62건에 이릅니다.

매년 평균 8건가량 발생한 것에 비하면 확산속도가 빨라진 겁니다.

아프리카돼지열병 발생지역은 그동안 경기, 강원, 경북에 머물렀지만 지난해 말 충남에 이어 최근에는 전남과 전북, 경남까지 전국으로 확산됐습니다.

지난달 강릉, 안성, 영광에서 발생한 아프리카돼지열병 바이러스는 그동안 야생멧돼지에서 나온 유전형과 다른 바이러스로 확인됐습니다.

[이동식/농림축산식품부 방역정책국장 : 아마 사람, 물품, 차량에 의한 인위적인 요소에 의한 발생으로 추정되고 있습니다.]

해외에서 유입됐을 가능성이 큽니다.

정부는 야생멧돼지에 맞춰졌던 방역 초점을 농장단위로 바꿔 오는 8일까지 전국 일제소독에 들어갔고 차량 방역 및 농장 출입통제를 강화해줄 것을 당부했습니다."""))
