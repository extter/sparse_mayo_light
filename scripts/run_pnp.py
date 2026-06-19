# Run Plug-and-Play HQS method
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from methods.pnp.main_reco import run_for_angles

ANGLE_CONFIGS = [45, 60, 90, 180]


def main():
    riepilogo = []
    for num_angoli in ANGLE_CONFIGS:
        print(f"\n\n{'#'*60}")
        print(f"#  AVVIO CONFIGURAZIONE: {num_angoli} ANGOLI")
        print(f"{'#'*60}")
        res = run_for_angles(num_angoli)
        riepilogo.append(res)

    print(f"\n\n{'='*60}")
    print("  RIEPILOGO COMPLESSIVO (medie per configurazione)")
    print(f"{'='*60}")
    print(f"  {'Angoli':>7} | {'PSNR FBP':>9} | {'PSNR PnP':>9} | {'SSIM FBP':>9} | {'SSIM PnP':>9}")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for r in riepilogo:
        print(f"  {r['angles']:>7} | {r['psnr_fbp']:>9.2f} | {r['psnr_pnp']:>9.2f} | "
              f"{r['ssim_fbp']:>9.4f} | {r['ssim_pnp']:>9.4f}")


if __name__ == "__main__":
    main()