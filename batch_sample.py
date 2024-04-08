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

compositions = ["SrTiO3", "TiO2", "TbCd3", "CsTeAu", "Gd2InGe2", "Li3MnCoO5"]

