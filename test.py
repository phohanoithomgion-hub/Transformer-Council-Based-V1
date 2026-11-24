from dataset import get_tokenizer, StreamingPretrainDataset

tok = get_tokenizer()
ds = StreamingPretrainDataset(tok, max_seq_len=512, config_names=["xerv_pure", "automathtext-v2-ultra"])

total_chars = 0
total_colons = 0
n = 0
for i, sample in enumerate(ds):
    text = tok.decode(sample["input_ids"], skip_special_tokens=True)
    total_chars += len(text)
    total_colons += text.count(":")
    n += 1
    if n >= 200:
        break

print(f"Samples: {n}")
print(f"Total chars: {total_chars}")
print(f"Total colons: {total_colons}")
print(f"Colon ratio: {total_colons / total_chars * 100:.2f}%")
print(f"Expected: < 1.5% nếu data sạch, > 3% nếu có vấn đề")