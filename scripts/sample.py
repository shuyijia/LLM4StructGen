import os
from peft import PeftModel

from llm4structgen.utils import *
from llm4structgen.sampling import *

args = ModelConfig(
    run_name="sample-test",
    model_name="7b",
    batch_size=2,
)

args.model_path = "exp/noisy-sample-test/checkpoint-33500/"
args.temperature = 0.9
args.top_p = 0.9

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

result = conditional_composition(model, tokenizer, composition, args.temperature, args.top_p)

with open(f"outputs/{save_folder_name}/test.cif", "w") as f:
    f.write(result["cif"])
