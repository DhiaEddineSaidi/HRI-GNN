import os
import argparse
import ast
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from hri_gnn.config.const import Configuration, SPLITS
from hri_gnn.datasets import RavenDataset, raven_flat_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.models import graph_builder as gb
from hri_gnn.utils import paths

def precompute_dataset(dataset_name="raven-100", batch_size=32, configs=None, device="cuda"):
    print(f"Precomputing {configs} from dataset {dataset_name} on {device}")
    
    out_path, _ = paths.get_precomputed_dataset_path(dataset_name)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load CNN encoder
    cnn_model = RavenCnnEncoder()
    output_path = paths.get_output_path()
    weights_path = output_path / 'models' / 'raven_cnn_encoder_final.pth'
    if weights_path.exists():
        cnn_model.load_state_dict(torch.load(weights_path))
        print("Loaded CNN weights.")
    else:
        print("WARNING: CNN weights not found. Precomputing with untrained CNN!")
    
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    for config in configs:
        for split in SPLITS:
            print(f"\nProcessing {config} - {split}")
            
            # Use percentage_limit=1.0 to precompute ALL problems
            dataset = RavenDataset(dataset=dataset_name, split=split, percentage_limit=1.0, configs=[config])
            
            if len(dataset) == 0:
                print(f"Skipping empty split: {config} - {split}")
                continue
            
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=raven_flat_collate,
                num_workers=0  # Set to 0 to prevent Windows memory exhaustion
            )
            
            config_out_path = out_path / config
            config_out_path.mkdir(parents=True, exist_ok=True)
            
            for batch in tqdm(dataloader, desc=f"Precomputing {config}_{split}"):
                # Move batch to device
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                
                with torch.inference_mode():
                    hetero_data_dict, meta_dict = gb.build_hetero_data_batch(batch, cnn_model, include_columns=True)
                
                # Group graphs by problem_id
                # hetero_data_dict keys are f"{config_id}_{problem_id}_{raw_batch_id}_c{candidate_id}"
                problems = defaultdict(lambda: {'data_list': [], 'layout_keys': []})
                
                # Ensure ordered candidate IDs
                def sort_key(k):
                    # k is e.g. "(0, 12, 0)_c8"
                    import ast
                    tuple_str, cand_str = k.split('_c')
                    key_tuple = ast.literal_eval(tuple_str)
                    return (key_tuple[0], key_tuple[1], key_tuple[2], int(cand_str))

                for graph_key in sorted(hetero_data_dict.keys(), key=sort_key):
                    tuple_str = graph_key.split('_c')[0] 
                    key_tuple = ast.literal_eval(tuple_str)
                    problem_id = key_tuple[1]
                    
                    data = hetero_data_dict[graph_key]
                    meta = meta_dict[graph_key]
                    
                    # Move data back to CPU before saving to save GPU memory and disk compatibility
                    data = data.cpu()
                    
                    problems[problem_id]['data_list'].append(data)
                    problems[problem_id]['layout_keys'].extend(meta['layout_keys'])
                
                # Save each problem
                for problem_id, content in problems.items():
                    save_path = config_out_path / f"RAVEN_{problem_id}_{split}.pt"
                    torch.save(content, save_path)
                    
                # Explicitly free memory
                del batch
                del hetero_data_dict
                del meta_dict
                del problems
                import gc
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="raven-100")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--configs", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.configs is None:
        args.configs = Configuration.all()
    else:
        for c in args.configs:
            if c not in Configuration.all():
                raise ValueError(f"Invalid config: {c}")
    
    print("Configurations to precompute: ", args.configs)
    precompute_dataset(args.dataset, args.batch_size, args.configs, args.device)
