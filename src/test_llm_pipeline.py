import sys
import ollama

def analyze_transcript(transcript_text: str) -> float:
    if not transcript_text.strip():
        return 0.0

    prompt = f"""
    Проанализируй транскрипцию игрового клипа.
    Оцени, насколько этот момент интересен, смешон, драматичен или вирален для TikTok/Shorts.
    
    Транскрипция:
    ---
    {transcript_text}
    ---

    Верни ТОЛЬКО JSON-объект строго по следующей схеме:
    {{
      "score": <float, от 0.0 до 1.0>,
      "reason": "<краткое объяснение на русском языке>"
    }}
    """

    try:
        response = ollama.chat(
            model='clip-judge',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0}
        )
        
        import json
        result = json.loads(response['message']['content'])
        score = float(result.get("score", 0.0))
        print(f"  🧠 Оценка LLM: {score:.2f} (Причина: {result.get('reason', '')})")
        return score

    except Exception as e:
        print(f"  ⚠️ Ошибка при обращении к Ollama: {e}")
        return 0.0

def test_llm_analysis():
    print("--- ТЕСТ ИНТЕЛЛЕКТУАЛЬНОГО АНАЛИЗА КЛИПА ---")
    
    transcript_example_1 = "Oh my god, he's going to do it! The clutch! He got him! Unbelievable!"
    transcript_example_2 = "And he's just walking around the map, looking for enemies. Nothing much happening here."

    print("\n1. Анализ 'СКУЧНОГО' клипа...")
    analyze_transcript(transcript_example_2)
    
    print("\n2. Анализ 'ХАЙЛАЙТ' клипа...")
    analyze_transcript(transcript_example_1)

if __name__ == "__main__":
    test_llm_analysis()
