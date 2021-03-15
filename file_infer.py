import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer
from unicode import join_jamos

# load pretrained model

tokenizer = Wav2Vec2Processor.from_pretrained(
    "/root/develop/wav2vec2-infer/processor_korean_base")
model = Wav2Vec2ForCTC.from_pretrained(
    "/root/develop/wav2vec2-infer/checkpoint-43000")


# load audio
audio_input, _ = sf.read("ko.wav")

# transcribe
input_values = tokenizer(audio_input, return_tensors="pt",
                         sampling_rate=16000).input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

print("TRANSCRIPTION")
joined = join_jamos(transcription)
print(joined)
