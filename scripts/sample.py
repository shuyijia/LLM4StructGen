import os
from peft import PeftModel

from llm4structgen.utils import *
from llm4structgen.sampling import conditional_composition, batch_conditional_composition

args = ModelConfig(
    run_name="sample-test",
    model_name="13b",
    batch_size=2,
)

args.model_path = "checkpoints/13b-no-val/checkpoint-410000/"
args.temperature = 0.5
args.top_p = 0.95

model_string = get_model_name(args.model_name)
print(model_string)

model = get_model(args, 0)
tokenizer = get_tokenizer(args)
model.eval()

smart_tokenizer_and_embedding_resize(model, tokenizer)
model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")

composition = "SrTiO3"
save_folder_name = "SrTiO3"
os.makedirs(f"outputs/{save_folder_name}", exist_ok=True)

temperatures = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.5]
top_ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for temperature in temperatures:
    for top_p in top_ps:
        result = conditional_composition(model, tokenizer, composition, temperature, top_p)
        with open(f"outputs/{save_folder_name}/{composition}_{temperature}_{top_p}.cif", "w") as f:
            f.write(result["cif"])
        print(f"Saved {composition}_{temperature}_{top_p}.cif")
