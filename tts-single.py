import os
from pathlib import Path

from openai import OpenAI

QUESTION_GSM8K= """
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and uses four to bake muffins for her friends. She sells the remaining eggs at the farmers' market for $2 each. How much money does she make per day at the farmers' market?
A. $12
B. $14
C. $18
D. $20
"""
QUESTION_GPQA = """
Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they be clearly resolved?
A. 10^-4 ev
B. 10^-9 ev
C. 10^-8 ev
D. 10^-11 ev
"""

def main() -> None:
    client = OpenAI(
        base_url="https://api.ohmygpt.com/v1",
        api_key="sk-2Nqq2VWF6dcE36A03473T3BlbKFJ3c87A119658845D29Bcc",
    )

    output_path = Path(__file__).resolve().parent / "audio_samples" / "question-gpqa.mp3"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream TTS output directly to disk to avoid holding the full audio in memory.
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd-1106",
        voice="alloy",
        input=QUESTION_GPQA,
    ) as response:
        response.stream_to_file(output_path)

    print(f"Saved synthesized question to {output_path}")


if __name__ == "__main__":
    main()
