import cv2, glob, csv, argparse
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(
        description="Estimate rust coverage on metal surfaces from images using color-based detection.",
        epilog="""
EXAMPLES:
  # Basic usage: process all JPGs in images/ folder (percent only)
  python rust_estimate.py

  # With direct scale calibration
  python rust_estimate.py --cm-per-pixel 0.05

  # Using reference measurement (known real distance vs. measured pixels)
  python rust_estimate.py --ref-length-cm 10.0 --ref-length-px 200

SCALE CALIBRATION METHODS (in order of priority):
  1. --cm-per-pixel: Direct scale (cm per pixel). Use this if you know the scale.
  2. --ref-length-cm + --ref-length-px: Measure a known real distance (e.g., 10 cm) 
     in the image and count the pixels. Then pass both values.
  3. Without scale: Run percent-only mode (default). Coverage % will be reported,
     but physical area (cm²) will not be calculated.

OUTPUT:
  - results.csv: Contains image names and coverage percentages (and cm² if scaled)
  - *_annot.jpg: Annotated images with detected rust contours highlighted in green

ALGORITHM:
  - Detects rust by analyzing orange/brown hues in HSV color space (H: 5-25°)
  - Identifies the metal panel using edge detection
  - Calculates coverage as: (rust_pixels / panel_pixels) × 100%
  - Applies morphological operations to reduce noise

NOTES:
  - Images with poor lighting or non-standard rust colors may need tuning
  - The detection is color-based, so it works best on painted steel surfaces
  - For pipes/poles, consider using rust_estimate_cv.py with --prefer-vertical flag
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--glob",
        default="images/*.jpg",
        help="Glob pattern for input images (default: images/*.jpg). Example: photos/*.jpg"
    )
    ap.add_argument(
        "--cm-per-pixel",
        type=float,
        default=None,
        help="Direct scale in cm/pixel. Use this if you know the exact scale of your images."
    )
    ap.add_argument(
        "--ref-length-cm",
        type=float,
        default=None,
        help="Physical length (cm) of a known reference in the image. Must be paired with --ref-length-px."
    )
    ap.add_argument(
        "--ref-length-px",
        type=float,
        default=None,
        help="Measured pixel distance for the reference length. Must be paired with --ref-length-cm."
    )
    return ap.parse_args()

def resolve_cm_per_pixel(args):
    if args.cm_per_pixel is not None:
        return args.cm_per_pixel
    if args.ref_length_cm and args.ref_length_px:
        return args.ref_length_cm / args.ref_length_px
    return None  # ingen fysisk skala ⇒ rapporter bare prosent

def rust_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 50, 20], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def panel_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros(gray.shape, np.uint8)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(m, [big], -1, 255, thickness=cv2.FILLED)
    else:
        m[:] = 255
    return m

def compute_metrics(img, cm_per_px=None):
    rust = rust_mask(img)
    panel = panel_mask(img)
    valid = cv2.bitwise_and(rust, panel)
    rust_px  = int(cv2.countNonZero(valid))
    panel_px = int(cv2.countNonZero(panel))
    coverage = 100.0 * rust_px / max(panel_px, 1)
    rust_cm2 = None
    if cm_per_px is not None:
        rust_cm2 = (rust_px * (cm_per_px ** 2))
    return valid, coverage, rust_cm2

def main():
    args = parse_args()
    cm_per_px = resolve_cm_per_pixel(args)
    rows = []
    for fp in glob.glob(args.glob):
        img = cv2.imread(fp)
        if img is None: 
            continue
        mask, cov, cm2 = compute_metrics(img, cm_per_px)
        ann = img.copy()
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, contours, -1, (0,255,0), 2)
        outname = fp.rsplit(".",1)[0] + "_annot.jpg"
        cv2.imwrite(outname, ann)
        row = [fp.split("/")[-1], f"{cov:.2f}"]
        if cm2 is not None:
            row.append(f"{cm2:.2f}")
        rows.append(row)

    # skriv CSV
    headers = ["image", "coverage_pct"] + (["rust_cm2"] if cm_per_px is not None else [])
    with open("results.csv","w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

if __name__ == "__main__":
    main()
