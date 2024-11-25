from transformers import LlamaConfig,LlamaForCausalLM,AutoModelForCausalLM
import torch, os
import json
from collections import OrderedDict
from safetensors.torch import load_file
import argparse
import shutil
parser = argparse.ArgumentParser("")
parser.add_argument("--inpath", type=str)
parser.add_argument("--inpath2", type=str,help="base model path")
parser.add_argument("--outpath", type=str)
parser.add_argument("--has_special", type=bool, default=False, help="Has special token")
args = parser.parse_args()

hf_config = LlamaConfig.from_pretrained(args.inpath2)
config_dict = hf_config.to_dict()
config = {
    'dim_model': hf_config.hidden_size,
    'dim_ff': hf_config.intermediate_size,
    'num_layers': hf_config.num_hidden_layers,
    'num_heads': hf_config.num_attention_heads,
    'num_heads_kv': hf_config.num_key_value_heads,
    'dim_head': hf_config.hidden_size // hf_config.num_attention_heads,
    'norm_eps': hf_config.rms_norm_eps,
}
if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)
with open(os.path.join(args.outpath, "config.json"), 'w') as f:
    json.dump(config_dict, f)
    
def mapping_function(bm_key):
    if bm_key == "input_embedding.weight":
        return "model.embed_tokens.weight"

    elif bm_key == "encoder.output_layernorm.weight":
        return "model.norm.weight"

    elif bm_key == "output_projection.weight":
        return "lm_head.weight"

    elif "encoder.layers." in bm_key:
        lnum = bm_key.split(".")[2]
        if "self_att.layernorm_before_attention.weight" in bm_key:
            return f"model.layers.{lnum}.input_layernorm.weight"
        elif "self_att.self_attention.project_q.weight" in bm_key:
            return f"model.layers.{lnum}.self_attn.q_proj.weight"
        elif "self_att.self_attention.project_k.weight" in bm_key:
            return f"model.layers.{lnum}.self_attn.k_proj.weight"
        elif "self_att.self_attention.project_v.weight" in bm_key:
            return f"model.layers.{lnum}.self_attn.v_proj.weight"
        elif "self_att.self_attention.attention_out.weight" in bm_key:
            return f"model.layers.{lnum}.self_attn.o_proj.weight"
        elif "ffn.layernorm_before_ffn.weight" in bm_key:
            return f"model.layers.{lnum}.post_attention_layernorm.weight"
        elif "ffn.ffn.w_in.w_0.weight" in bm_key:
            return f"model.layers.{lnum}.mlp.gate_proj.weight"
        elif "ffn.ffn.w_in.w_1.weight" in bm_key:
            return f"model.layers.{lnum}.mlp.up_proj.weight"
        elif "ffn.ffn.w_out.weight" in bm_key:
            return f"model.layers.{lnum}.mlp.down_proj.weight"
    return None
layernum = config['num_layers']
hf_model_dict = OrderedDict()
ckpt_num = None

if args.has_special:
    for name in os.listdir(args.inpath2):
        if name.startswith("model-") and name.endswith(".safetensors"):
            if int(name[-17:-12]):
                ckpt_num = int(name[-17:-12])
    for i in range(1, ckpt_num + 1):
        part = load_file(os.path.join(args.inpath2, f"model-{i:05d}-of-{ckpt_num:05d}.safetensors"))
        hf_model_dict.update(part)
else:
    for name in os.listdir(args.inpath2):
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            if int(name[-9:-4]):
                ckpt_num = int(name[-9:-4])
    for i in range(1, ckpt_num + 1):
        part = torch.load(os.path.join(args.inpath2, f"pytorch_model-{i:05d}-of-{ckpt_num:05d}.bin"))
        hf_model_dict.update(part)
    
bm_model = torch.load(os.path.join(args.inpath, "pytorch_model.pt"))
for key, value in bm_model.items():
    mapped_key = mapping_function(key)
    if mapped_key in hf_model_dict:
        hf_model_dict[mapped_key] = value

key_to_check = '.self_attn.rotary_emb.inv_freq'

keys_to_delete = [key for key in hf_model_dict if key_to_check in key]

for key in keys_to_delete:
    del hf_model_dict[key]

hf_model = AutoModelForCausalLM.from_pretrained(args.inpath2, torch_dtype=torch.bfloat16, device_map='cuda')
hf_model.load_state_dict(hf_model_dict)
hf_model.save_pretrained(args.outpath)

token_files = [f for f in os.listdir(args.inpath) if "token" in f]
for file in token_files:
    shutil.copy(os.path.join(args.inpath, file), os.path.join(args.outpath, file))

# shutil.rmtree(args.inpath)