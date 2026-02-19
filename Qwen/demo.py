from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "image_qwen.png"},
        {"type": "text", "text": "If I tell you something, you'll make a model like this? Explain"},
]}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=128)
trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
 
print("답변:", answer)


