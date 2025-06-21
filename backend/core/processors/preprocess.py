import tritonclient.http as httpclient
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./shared/tokenizer")

def prepare_data(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True ,return_attention_mask=True)

    input_ids_np = inputs['input_ids'].astype(np.int64)
    attention_mask_np = inputs['attention_mask'].astype(np.int64)

    input_ids_tensor = httpclient.InferInput("input_ids", input_ids_np.shape, "INT64")
    input_ids_tensor.set_data_from_numpy(input_ids_np)

    attention_mask_tensor = httpclient.InferInput("attention_mask", attention_mask_np.shape, "INT64")
    attention_mask_tensor.set_data_from_numpy(attention_mask_np)

    return input_ids_tensor, attention_mask_tensor