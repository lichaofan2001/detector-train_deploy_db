'''python loss_landscapev3.py     --weights temp1/best_trim_3_4.pt     --data data/data_0912.yaml     --output results/landscape     --steps 20     --batch-size 8     --max-batches 10     --range 0.5     --hyp data/hyp.scratch.p5.yaml'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import argparse
import sys
import yaml
import logging

# Add YOLOv7 paths
sys.path.insert(0, './')

from detector.models.yolo import Model
from detector.utils.datasets import create_dataloader
from detector.utils.general import colorstr, check_img_size, init_seeds
from detector.utils.torch_utils import select_device
from detector.utils.loss import ComputeLoss, ComputeLossOTA
from detector.models.experimental import attempt_load

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize YOLOv7 Loss Landscape')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Dataset config (.yaml)')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='Hyperparameters file')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--device', default='', help='Device (e.g., 0 or cpu)')
    parser.add_argument('--steps', type=int, default=20, help='Grid resolution')
    parser.add_argument('--range', type=float, default=1.0, help='Perturbation range')
    parser.add_argument('--max-batches', type=int, default=10, help='Max batches for loss evaluation')
    parser.add_argument('--output', type=str, default='loss_landscape', help='Output directory')
    parser.add_argument('--layer', type=str, default=None, help='Specific layer to analyze (e.g., " input 1.conv will search model.1.conv.weight"). If not given, analyze full model.')
    return parser.parse_args()


class YOLOv7LossLandscapeVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = select_device(args.device, batch_size=args.batch_size)
        self.model = None
        self.dataloader = None
        self.criterion = None
        self.original_params = None
        self.data_dict = None
        self.hyp = None

    def load_hypers(self):
        """Load hyperparameters from YAML or use defaults."""
        if self.args.hyp and os.path.isfile(self.args.hyp):
            with open(self.args.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)
        else:
            # Minimal default hypers needed for dataloader + loss
            self.hyp = {
                'box': 0.05, 'cls': 0.5, 'obj': 1.0, 'cls_pw': 1.0, 'obj_pw': 1.0,
                'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
                'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
                'shear': 0.0, 'perspective': 0.0,
                'flipud': 0.0, 'fliplr': 0.5,
                'mosaic': 1.0, 'mixup': 0.0,
                'loss_ota': 1,
            }
        print(colorstr('green', 'Hyperparameters loaded'))

    def load_model_and_criterion(self):
        """Load model and instantiate appropriate loss function."""
        if not os.path.isfile(self.args.weights):
            raise FileNotFoundError(f"Weights file not found: {self.args.weights}")

        self.model = attempt_load(self.args.weights, map_location=self.device).float()
        self.model.to(self.device).eval()

        # Choose loss based on hyperparameter
        if self.hyp.get('loss_ota', 1):
            self.criterion = ComputeLossOTA(self.model)
            print(colorstr('green', 'Using ComputeLossOTA'))
        else:
            self.criterion = ComputeLoss(self.model)
            print(colorstr('green', 'Using standard ComputeLoss'))

        self.original_params = self._get_flat_params()
        print(colorstr('green', f'Model loaded with {len(self.original_params)} parameters'))

    def build_dataloader(self):
        """Build dataloader consistent with training settings."""
        with open(self.args.data, errors='ignore') as f:
            self.data_dict = yaml.safe_load(f)

        train_path = self.data_dict['train']
        if not os.path.exists(train_path):
            train_path = os.path.join(os.path.dirname(self.args.data), train_path)
            if not os.path.exists(train_path):
                raise FileNotFoundError(f'Training data not found: {train_path}')

        gs = max(int(self.model.stride.max()), 32)
        imgsz = check_img_size(self.args.img_size, gs)

        class SimpleOpt:
            def __init__(self, workers):
                self.workers = workers
                self.rect = False
                self.cache_images = False
                self.image_weights = False
                self.quad = False
                self.rank = -1
                self.world_size = 1
                self.single_cls = False
                self.local_rank = -1
                self.task = 'train'
                self.global_rank = -1
                self.ota = True  # Required for OTA loss compatibility

        dataloader, _ = create_dataloader(
            train_path, imgsz, self.args.batch_size, gs,
            opt=SimpleOpt(workers=self.args.workers),
            hyp=self.hyp,
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            workers=self.args.workers,
            image_weights=False,
            quad=False,
            prefix=colorstr('train: ')
        )
        self.dataloader = dataloader
        print(colorstr('green', f'Dataloader created with {len(dataloader)} batches'))
    def _get_target_params_info(self):
        """Return list of (name, param) for target layer(s), robust to layer fusion."""
        all_params = dict(self.model.named_parameters())
        
        if self.args.layer is None:
            # Full model
            return [(name, p) for name, p in all_params.items() if p.numel() > 0]
        else:
            # Normalize layer spec: support "1", "model.1", etc.
            layer_id = str(self.args.layer).lstrip('model.').rstrip('.')
            # Candidate parameter names: weight and bias
            candidates = [f"model.{layer_id}.weight", f"model.{layer_id}.bias"]
            
            found = []
            for name in candidates:
                if name in all_params:
                    found.append((name, all_params[name]))
            
            if not found:
                # Debug: print all param names
                print("\nAvailable parameter names (first 20):")
                for i, name in enumerate(list(all_params.keys())[:20]):
                    print(f"  {name}")
                if len(all_params) > 20:
                    print("  ... (and more)")
                raise ValueError(
                    f"No parameters found for layer '{self.args.layer}'. "
                    f"Tried: {candidates}. See available names above."
                )
            return found
    def _get_flat_params(self):
        """Flatten parameters of target layer(s) or full model."""
        self.target_param_info = self._get_target_params_info()
        params = [p.data.view(-1) for _, p in self.target_param_info] # 将每个参数展平
        if not params:
            raise RuntimeError("No trainable parameters selected.")
        return torch.cat(params) # 拼接完整

    def _set_flat_params(self, flat_params):
        """Assign flattened vector back to target layer(s)."""
        pointer = 0
        for name, p in self.target_param_info:
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p.data))
            pointer += numel
    def evaluate_loss_on_batches(self, model, dataloader, max_batches=10):
        """Evaluate average loss over a subset of batches."""
        total_loss = 0.0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for i, (imgs, targets, _, _) in enumerate(dataloader):
                if i >= max_batches:
                    break
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device)

                preds = model(imgs)

                try:
                    if isinstance(self.criterion, ComputeLossOTA):
                        if isinstance(preds, tuple) and len(preds) == 2:
                            loss, _ = self.criterion(preds[1], targets, imgs)
                        else:
                            loss, _ = self.criterion(preds, targets, imgs)
                    else:
                        loss, _ = self.criterion(preds, targets)
                    total_loss += loss.item() * imgs.size(0)
                    total_samples += imgs.size(0)
                except Exception as e:
                    print(colorstr('red', f'Loss computation failed: {e}'))
                    # Fallback to standard loss
                    try:
                        fallback_criterion = ComputeLoss(model)
                        loss, _ = fallback_criterion(preds, targets)
                        total_loss += loss.item() * imgs.size(0)
                        total_samples += imgs.size(0)
                    except Exception:
                        total_loss += 10.0 * imgs.size(0)
                        total_samples += imgs.size(0)

        return total_loss / total_samples if total_samples > 0 else float('inf')

    def sample_orthonormal_directions(self):
        """Sample two orthonormal random directions in parameter space."""
        def normalize_by_layer(vec):
            pointer = 0
            normalized = []
            normalized_v2 = []
            
            
            for _, p in self.target_param_info:
                if p.numel() == 0:
                    continue
                chunk = vec[pointer:pointer + p.numel()]
                
                
                if p.dim() >= 2:
                    if p.dim() == 4:  # Conv layer
                        out_ch = p.shape[0]
                        reshaped = chunk.view(out_ch, -1)
                        reshaped_p = p.view(out_ch, -1)
                        norm = reshaped.norm(dim=1, keepdim=True)
                        norm_p = reshaped_p.norm(dim=1, keepdim=True)
                        normalized_chunk = reshaped / (norm + 1e-10) * norm_p
                        
                        chunk_v2 = torch.randn_like(normalized_chunk) # init
                        for ii in range(out_ch):
                            chunk_v2[ii] = norm_p[ii] * chunk_v2[ii] - normalized_chunk[ii]*(chunk_v2[ii] @ normalized_chunk[ii])
                            
                            chunk_v2[ii] = chunk_v2[ii]/(1e-10+chunk_v2[ii].norm()) * norm_p[ii]
                        
                        normalized_chunk = normalized_chunk.view(-1)
                        chunk_v2 = chunk_v2.view(-1)
                        
                        #print(chunk_v2)
                        normalized.append(normalized_chunk)
                        normalized_v2.append(chunk_v2)
                        
                    else:
                        norm = chunk.norm()
                        norm_p = p.view(-1).norm()
                        v1 = chunk / (norm + 1e-10) * norm_p
                        v2 = torch.randn_like(v1)
                        v2 = norm_p * v2 - v1 * (v1 @ v2)
                        
                        v2 = v2 / (v2.norm() + 1e-10) * norm_p
                        #print(v2)
                        
                        normalized.append(v1)
                        normalized_v2.append(v2)
                else:  # bias or 1D
                    norm = chunk.norm()
                    
                    norm_p = p.view(-1).norm()
                    v1 = chunk / (norm + 1e-10) * norm_p
                    v2 = torch.randn_like(v1)
                    v2 = norm_p * v2 - v1 * (v1 @ v2)
                    v2 = v2 / (v2.norm() + 1e-10) * norm_p
                    
                    #print(v2)
                    normalized.append(v1)
                    normalized_v2.append(v2)
                    
                pointer += p.numel()
            return torch.cat(normalized), torch.cat(normalized_v2)

        d1, d2 = normalize_by_layer(torch.randn_like(self.original_params))
        
        print(f'Norm: {d1.norm()}, {d2.norm()}')
        d1 = d1 / d1.norm()
        d2 = d2 / d2.norm()
        
        return d1, d2

    def compute_landscape(self, dir1, dir2):
        """Compute loss values over a 2D grid of perturbations."""
        steps = self.args.steps
        r = self.args.range
        x = np.linspace(-r, r, steps)
        y = np.linspace(-r, r, steps)
        X, Y = np.meshgrid(x, y)
        Z = np.full((steps, steps), np.nan)

        for i in tqdm(range(steps), desc='Computing landscape'):
            for j in range(steps):
                new_params = self.original_params + X[i, j] * dir1 + Y[i, j] * dir2
                self._set_flat_params(new_params)
                loss = self.evaluate_loss_on_batches(self.model, self.dataloader, self.args.max_batches)
                
                #print(f'Loss = {loss}, {dir1}, {dir2}')
                Z[i, j] = loss
                if (i * steps + j) % 10 == 0:
                    torch.cuda.empty_cache()

        self._set_flat_params(self.original_params)  # restore original
        return X, Y, Z

    def enhance_visualization(self, Z):
        """Enhance landscape for better visualization."""
        Z_clean = np.nan_to_num(Z, nan=np.median(Z))
        q05, q95 = np.percentile(Z_clean, [5, 95])
        Z_clipped = np.clip(Z_clean, q05, q95)
        Z_smooth = gaussian_filter(Z_clipped, sigma=1.0)
        Z_smooth = np.nan_to_num(Z, nan=np.nanmedian(Z)) 
        z_min, z_max = Z_smooth.min(), Z_smooth.max()
        Z_norm = (Z_smooth - z_min) / (z_max - z_min + 1e-10) if z_max > z_min else Z_smooth
        return Z_norm, Z_smooth

    def plot_landscape(self, X, Y, Z_enhanced, Z_original, save_path):
        """Plot 2D contour, heatmap, and 3D surface."""
        fig = plt.figure(figsize=(20, 6))

        ax1 = fig.add_subplot(131)
        cs = ax1.contour(X, Y, Z_original, levels=15, cmap='viridis')
        ax1.clabel(cs, inline=True, fontsize=8)
        cf = ax1.contourf(X, Y, Z_original, levels=50, cmap='viridis', alpha=0.7)
        ax1.set_title('2D Contour (Original Loss)')
        plt.colorbar(cf, ax=ax1)

        ax2 = fig.add_subplot(132)
        im = ax2.imshow(Z_enhanced, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', cmap='hot', aspect='auto')
        ax2.set_title('2D Heatmap (Enhanced)')
        plt.colorbar(im, ax=ax2)

        ax3 = fig.add_subplot(133, projection='3d')
        surf = ax3.plot_surface(X, Y, Z_original, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.8)
        ax3.set_title('3D Surface (Original Loss)')
        fig.colorbar(surf, ax=ax3, shrink=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_flatness(self, Z):
        """Compute flatness metrics."""
        min_l, max_l = Z.min(), Z.max()
        mean_l, std_l = Z.mean(), Z.std()
        flat_ratio = (max_l - min_l) / (min_l + 1e-10)
        metrics = {
            'min_loss': float(min_l),
            'max_loss': float(max_l),
            'mean_loss': float(mean_l),
            'std_loss': float(std_l),
            'flatness_ratio': float(flat_ratio)
        }
        print(colorstr('blue', '\nFlatness Analysis:'))
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics

    def run(self):
        """Main pipeline."""
        init_seeds(40)
        print("=" * 60)
        print("YOLOv7 Loss Landscape Visualization")
        print("=" * 60)

        self.load_hypers()
        self.load_model_and_criterion()
        self.build_dataloader()

        # Quick test
        test_loss = self.evaluate_loss_on_batches(self.model, self.dataloader, max_batches=1)
        print(colorstr('green', f'Initial loss test passed: {test_loss:.4f}'))

        d1, d2 = self.sample_orthonormal_directions()
        X, Y, Z = self.compute_landscape(d1, d2)

        os.makedirs(self.args.output, exist_ok=True)
        Z_enh, Z_smooth = self.enhance_visualization(Z)

        plot_path = os.path.join(self.args.output, 'loss_landscape.png')
        self.plot_landscape(X, Y, Z_enh, Z_smooth, plot_path)

        flatness = self.analyze_flatness(Z_smooth)

        data_path = os.path.join(self.args.output, 'landscape_data.npz')
        np.savez(data_path, X=X, Y=Y, Z_original=Z, Z_enhanced=Z_enh, Z_smooth=Z_smooth, flatness=flatness)

        print(colorstr('green', f'\nResults saved to: {self.args.output}'))


def main():
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    visualizer = YOLOv7LossLandscapeVisualizer(args)
    visualizer.run()


if __name__ == '__main__':
    main()

