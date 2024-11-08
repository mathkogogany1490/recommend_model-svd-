from transformers import MarianMTModel, MarianTokenizer

# 로드한 모델로 번역 테스트
def translate_ko_to_en(text):
    # 저장된 모델과 토크나이저를 로드
    save_directory = "../data/model/"
    tokenizer = MarianTokenizer.from_pretrained(save_directory)
    model = MarianMTModel.from_pretrained(save_directory)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    # 테스트 문장
    korean_text = ("웃긴 영화를 추천해 주세요?")
    translated_text = translate_ko_to_en(korean_text)
    print(f"Translated Text: {translated_text}")