import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal
gdal.UseExceptions()

from metrics import calculate_single_metrics, denormalize_dem
from dataset import RealWorldHDF5Dataset as TestDataset
from networks import ResidualSARDEMGenerator


def visualize_results(lr_dem_raw, sr_dem_raw, hr_dem_raw, vv_sar_norm, vh_sar_norm, sample_id, save_dir):
    save_dir.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    vmin, vmax = np.min(hr_dem_raw), np.max(hr_dem_raw)

    def plot_dem(ax, data, title, is_ground_truth=False):
        im = ax.imshow(data, cmap='terrain', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}\nRange: [{data.min():.1f}, {data.max():.1f}]m')
        ax.axis('off')
        if is_ground_truth: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')

    plot_dem(axes[0], lr_dem_raw, 'LR DEM (Real)')
    plot_dem(axes[1], sr_dem_raw, 'SR DEM')
    plot_dem(axes[2], hr_dem_raw, 'HR DEM (Ground Truth)', is_ground_truth=True)
    axes[3].imshow(vv_sar_norm, cmap='gray');
    axes[3].set_title('VV SAR');
    axes[3].axis('off')
    axes[4].imshow(vh_sar_norm, cmap='gray');
    axes[4].set_title('VH SAR');
    axes[4].axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / f'test_sample_{sample_id:05d}.png', dpi=200, bbox_inches='tight')
    plt.close()


def save_geotiff(sr_dem_raw, original_hr_dem_path, output_dir, data_root):
    output_dir.mkdir(exist_ok=True, parents=True)
    full_original_path = str(Path(data_root) / Path(original_hr_dem_path))
    output_path = str(output_dir / Path(original_hr_dem_path).name)
    try:
        src_ds = gdal.Open(full_original_path, gdal.GA_ReadOnly)
        if src_ds is None:
            print(f"\nGDAL cannot open the source file: {full_original_path}")
            return
        src_geotransform = src_ds.GetGeoTransform()
        src_projection = src_ds.GetProjection()
        src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, xsize=sr_dem_raw.shape[1], ysize=sr_dem_raw.shape[0], bands=1,
                               eType=gdal.GDT_Float32, options=['COMPRESS=LZW'])
        out_ds.SetGeoTransform(src_geotransform)
        out_ds.SetProjection(src_projection)
        out_band = out_ds.GetRasterBand(1)
        if src_nodata is not None:
            out_band.SetNoDataValue(src_nodata)
        out_band.WriteArray(sr_dem_raw)
        out_band.FlushCache();
        out_ds = None;
        src_ds = None
    except Exception as e:
        print(f"\nError saving GeoTIFF using GDAL: {e}\n  Source: {full_original_path}\n  Target: {output_path}")


def test_model(args):
    with open(args.global_stats, 'r') as f:
        global_stats = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dataset = TestDataset(
        h5_file_path=args.test_h5_path,
        global_stats_path=args.global_stats,
        csv_file_path=args.test_dataset_csv
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test dataset size: {len(test_dataset)}")

    model = ResidualSARDEMGenerator(
        sar_input_mode=args.sar_input_mode,
        fusion_scales=args.fusion_scales
    ).to(device)
    checkpoint = torch.load(args.model_weights, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded model weights: {args.model_weights}")

    output_dir = Path(args.output_dir)
    vis_dir, geotiff_dir, log_dir = output_dir / 'visualizations', output_dir / 'Bicubic-copdem', output_dir / 'logs'
    for d in [vis_dir, geotiff_dir, log_dir]: d.mkdir(exist_ok=True, parents=True)

    log_csv = log_dir / 'per_sample_test_results.csv'
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'Original_Filename', 'SR_PSNR', 'SR_SSIM', 'SR_RMSE', 'SR_MAE',
                         'Bicubic_PSNR', 'Bicubic_SSIM', 'Bicubic_RMSE', 'Bicubic_MAE'])

    model.eval()
    total_sr_metrics = {'psnr': 0, 'ssim': 0, 'rmse': 0, 'mae': 0}
    total_bicubic_metrics = {'psnr': 0, 'ssim': 0, 'rmse': 0, 'mae': 0}

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for batch_idx, batch in pbar:
            lr_dem_norm, hr_dem_raw = batch['lr_dem_norm'].to(device), batch['hr_dem_raw']
            vv_sar_norm, vh_sar_norm = batch['vv_sar_norm'].to(device), batch['vh_sar_norm'].to(device)

            sr_dem_norm = model(lr_dem_norm, vv_sar_norm, vh_sar_norm)
            bicubic_dem_norm = F.interpolate(lr_dem_norm, scale_factor=4, mode='bicubic', align_corners=False)

            sr_denorm = denormalize_dem(sr_dem_norm.cpu(), global_stats)
            bicubic_denorm = denormalize_dem(bicubic_dem_norm.cpu(), global_stats)
            lr_denorm = denormalize_dem(lr_dem_norm.cpu(), global_stats)

            for i in range(lr_dem_norm.size(0)):
                current_sample_idx = batch_idx * args.batch_size + i
                if current_sample_idx >= len(test_dataset): continue

                hr_sample = hr_dem_raw[i, 0].numpy()
                sr_sample = sr_denorm[i, 0].numpy()
                bicubic_sample = bicubic_denorm[i, 0].numpy()

                sr_metrics = calculate_single_metrics(sr_sample, hr_sample)
                bicubic_metrics = calculate_single_metrics(bicubic_sample, hr_sample)

                for key in total_sr_metrics:
                    total_sr_metrics[key] += sr_metrics[key]
                    total_bicubic_metrics[key] += bicubic_metrics[key]

                original_dem_path = test_dataset.data_index.iloc[current_sample_idx]['dem_path']

                with open(log_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{current_sample_idx:05d}", Path(original_dem_path).name,
                                     f"{sr_metrics['psnr']:.4f}", f"{sr_metrics['ssim']:.4f}",
                                     f"{sr_metrics['rmse']:.4f}", f"{sr_metrics['mae']:.4f}",
                                     f"{bicubic_metrics['psnr']:.4f}", f"{bicubic_metrics['ssim']:.4f}",
                                     f"{bicubic_metrics['rmse']:.4f}", f"{bicubic_metrics['mae']:.4f}"])

                if args.save_geotiff:
                    save_geotiff(sr_sample, original_dem_path, geotiff_dir, args.test_data_root)

            if batch_idx % args.vis_freq == 0:
                visualize_results(lr_denorm[0, 0].numpy(), sr_denorm[0, 0].numpy(), hr_dem_raw[0, 0].numpy(),
                                  vv_sar_norm[0, 0].cpu().numpy(), vh_sar_norm[0, 0].cpu().numpy(),
                                  batch_idx * args.batch_size, vis_dir)

    num_samples = len(test_dataset)
    avg_sr_metrics = {key: val / num_samples for key, val in total_sr_metrics.items()}
    avg_bicubic_metrics = {key: val / num_samples for key, val in total_bicubic_metrics.items()}
    summary_text = (
        f"--- Test Results Summary ---\n" f"Model: {args.model_weights}\n" f"Datasets: {args.test_h5_path}\n"
        f"SAR input mode: {args.sar_input_mode}\n"
        f"Fusion scales: {args.fusion_scales}\n"
        f"Total Test Samples: {num_samples}\n\n"
        f"Model (SR) Average Metrics:\n"
        f"  - PSNR: {avg_sr_metrics['psnr']:.4f} dB\n" f"  - SSIM: {avg_sr_metrics['ssim']:.4f}\n"
        f"  - RMSE: {avg_sr_metrics['rmse']:.4f} m\n"  f"  - MAE:  {avg_sr_metrics['mae']:.4f} m\n\n"
        f"Baseline (Bicubic) Average Metrics:\n"
        f"  - PSNR: {avg_bicubic_metrics['psnr']:.4f} dB\n" f"  - SSIM: {avg_bicubic_metrics['ssim']:.4f}\n"
        f"  - RMSE: {avg_bicubic_metrics['rmse']:.4f} m\n"  f"  - MAE:  {avg_bicubic_metrics['mae']:.4f} m\n")
    print(summary_text)
    with open(output_dir / 'test_summary.txt', 'w') as f:
        f.write(summary_text)



def main():
    parser = argparse.ArgumentParser(description='SAR Enhanced DEM Super Resolution Testing (Real-World LR from HDF5)')


    parser.add_argument('--test_h5_path', type=str,
                        default=r"Test_data/train_data.h5",
                        help='Path to the test HDF5 file (containing real LR data)')

    parser.add_argument('--test_dataset_csv', type=str,
                        default=r"Test_data/data_index.csv",
                        help='Path to the test dataset index CSV file (for filename lookup)')


    parser.add_argument('--global_stats', type=str,
                        default=r"Train_data/global_stats",
                        help='Global statistics JSON file for normalization')
    parser.add_argument('--sar_input_mode', type=str,
                        choices=['none', 'vv', 'vh', 'both'],
                        default='both',
                        help='SAR input mode used during training')
    parser.add_argument('--fusion_scales', type=str,
                        default='16,32,64',
                        help='Fusion scales used during training, e.g. 16, 32, 64, 16,32, 16,64, 32,64, 16,32,64')
    parser.add_argument('--model_weights', type=str,
                        default=r"checkpoints/best_model.pth",
                        help='Path to pretrained model weights')
    parser.add_argument('--output_dir', type=str,
                        default=r"output_dir",
                        help='Directory to save all results')

    parser.add_argument('--test_data_root', type=str,
                        default=r"Test_data/test_data_tif",
                        help='Root directory of original HR TIF data (needed for GeoTIFF metadata)')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--vis_freq', type=int, default=10, help='Save visualization every N batches.')
    parser.add_argument('--save_geotiff', action='store_true', default=True,
                        help='Save all SR results as GeoTIFF files')
    parser.add_argument('--no-save_geotiff', dest='save_geotiff', action='store_false')

    args = parser.parse_args()

    print("\n--- Parameter configuration ---")
    for arg_name, value in sorted(vars(args).items()):
        print(f"  {arg_name:<20}: {value}")
    print("----------------------------------------\n")

    test_model(args)


if __name__ == "__main__":
    main()
