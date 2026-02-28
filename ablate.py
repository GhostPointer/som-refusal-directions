import os
import time
import torch
import argparse
from models.load_models import load_model
from utils.ablation_utils import ablate_weights
from utils import extract_direction_prefix
from config import Config
from directions_ablation import generate_and_save_hookfree_completions
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="llama2-7b", type=str, help="llm")
    parser.add_argument("--directions_path", type=str, required=True, help="Path to the .pt file containing direction vector.")
    parser.add_argument("--dir_ids", nargs='+', type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="harmbench_test", help="Name of dataset to run on.")
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--layer', type=int, default=31, help="Layer where directions were extracted (centers the hook window for MoE models)")
    args = parser.parse_args()
 
    t_pipeline = time.time()

    # model and config
    t0 = time.time()
    device = torch.device(args.device)
    model = load_model(args.model_name, device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    model_path = args.model_name
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    directions = torch.load(args.directions_path)
    multi_dirs = directions[args.dir_ids] if args.dir_ids else directions
    aux_name = 'raw_dirs'
    print('Using raw directions')

    # Load direction
    len_dir = len(multi_dirs.shape)
    print(f"→ Loaded {len_dir}D direction vectors from {args.directions_path} with shape {multi_dirs.shape}")

    t0 = time.time()
    for idx, sing_dir in enumerate(multi_dirs):
        dir_label = str(args.dir_ids[idx]) if args.dir_ids else str(idx)
        aux_name += '_' + dir_label
        ablate_weights(model, sing_dir, source_layer=args.layer)
    print(f"  Ablation done in {time.time() - t0:.1f}s")

    # completions name
    aux_name = aux_name+'_'+extract_direction_prefix(args.directions_path)

    # Run generation
    t0 = time.time()
    generate_and_save_hookfree_completions(
        cfg=cfg,
        folder='completions',
        model_base=model,
        dataset_name=args.dataset_name,
        aux_name=aux_name
    )
    print(f"  Generation done in {time.time() - t0:.1f}s")
    print(f"  Total pipeline: {time.time() - t_pipeline:.1f}s")