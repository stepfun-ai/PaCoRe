from pacore.batch_inference.base_exp import Exp


class MyExp(Exp):
    # Model & API
    model_name = "pacore-8b"
    api_base = "http://localhost:8000/v1/chat/completions"

    # Data
    data_path = "playground/sample_questions.jsonl"
    output_dir = "outputs"

    # Generation params
    max_tokens = None # NOTE: this will be generated until model max_len of 131072
    temperature = 1.0
    stream = True

    num_responses_per_round = [4,] # low setting: [4,], medium setting: [16,], high setting: [32, 4,]

    # Concurrency
    max_concurrent = 1024


if __name__ == "__main__":
    exp = MyExp()
    exp.run()
