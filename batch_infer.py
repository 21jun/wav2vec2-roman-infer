from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import jiwer
import fastwer


librispeech_eval = load_dataset(
    "librispeech_asr", "clean", split="test", ignore_verifications=True)
# librispeech_eval = load_dataset(
#     "patrickvonplaten/librispeech_asr_dummy", "clean")

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self").to("cuda")
# model = Wav2Vec2ForCTC.from_pretrained(
#     "/root/develop/data/checkpoint-57500").to("cuda")

tokenizer = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self")

print(tokenizer)


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


librispeech_eval = librispeech_eval.map(map_to_array)


def map_to_pred(batch):
    input_values = tokenizer(
        batch["speech"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    batch["transcription"] = transcription

    return batch


result = librispeech_eval.map(
    map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

# print(result['text'], result['transcription'])

texts = [x['text'] for x in result]
transcriptions = [x['transcription'] for x in result]


cer = fastwer.score(texts, transcriptions, char_level=True)
wer = fastwer.score(texts, transcriptions)
print(cer, wer)
print(jiwer.wer(texts, transcriptions))

print(texts[0])
print(transcriptions[0])

# for a, b in zip(texts, transcriptions):
#     if a != b:
#         print(a)
#         print(b)
#         print("---------------")
