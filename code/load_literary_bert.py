from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    model = AutoModel.from_pretrained("severinsimmler/literary-german-bert")
    tokenizer = AutoTokenizer.from_pretrained("severinsimmler/literary-german-bert")
    model.save_pretrained("models/literary-german-bert")
    tokenizer.save_pretrained("models/literary-german-bert")