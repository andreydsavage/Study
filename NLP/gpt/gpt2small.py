from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

model_name_or_path = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

# Заранее токенизируем текст
# text = 'Определение: "Нейронная сеть" - это'
def make_generation(text, temperature, top_k, top_p, max_length ):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    # Пример вероятностного сэмплирвоания с ограничением
    out = model.generate(input_ids,
                     do_sample=True,
                     temperature=1.3,
                     top_k=20,
                     top_p=0.8,
                     max_length=30,
                    )
    # Декодирование токенов
    generated_text = list(map(tokenizer.decode, out))[0]
    return generated_text