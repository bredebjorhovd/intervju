
import cv2, glob, csv, argparse, os
import numpy as np

# -------------------------- ArUco (optional) --------------------------
def _get_aruco():
    """Return (aruco_module, dictionary, detector_or_params) or (None,None,None) if unavailable."""
    if not hasattr(cv2, "aruco"):
        return None, None, None
    aruco = cv2.aruco
    DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "ArucoDetector"):  # new API (≈4.7+)
        params = aruco.DetectorParameters()
        det = aruco.ArucoDetector(DICT, params)
        return aruco, DICT, det
    else:  # old API
        params = aruco.DetectorParameters_create()
        return aruco, DICT, params

ARUCO, ARUCO_DICT, ARUCO_DET = _get_aruco()

def aruco_detect_markers(img):
    if ARUCO is None:
        return None, None
    if hasattr(ARUCO, "ArucoDetector"):
        corners, ids, _ = ARUCO_DET.detectMarkers(img)
    else:
        corners, ids, _ = ARUCO.detectMarkers(img, ARUCO_DICT, parameters=ARUCO_DET)
    return corners, ids

def cm_per_pixel_from_aruco(img, marker_size_cm=5.0):
    corners, ids = aruco_detect_markers(img)
    if ids is None or len(corners) == 0:
        return None
    c = corners[0][0]  # (4,2)
    s1 = np.linalg.norm(c[0] - c[1])
    s2 = np.linalg.norm(c[1] - c[2])
    px_per_cm = ((s1 + s2) / 2.0) / marker_size_cm
    if px_per_cm <= 0:
        return None
    return 1.0 / px_per_cm  # cm/px

# -------------------------- CLI --------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Advanced rust coverage estimator using computer vision (HSV + LAB color space, ArUco support).",
        epilog="""
EXAMPLES:
  # Basic usage: process all JPGs, output to 'out/' folder
  python rust_estimate_cv.py

  # With direct scale and output to custom directory
  python rust_estimate_cv.py --cm-per-pixel 0.05 --out-dir results/

  # For vertical objects (pipes, poles) with ArUco marker calibration
  python rust_estimate_cv.py --prefer-vertical --aruco --aruco-size-cm 4.5

  # Using reference measurement
  python rust_estimate_cv.py --glob photos/*.png --ref-length-cm 15.0 --ref-length-px 300

SCALE CALIBRATION METHODS (in order of priority):
  1. --cm-per-pixel: Direct scale (cm per pixel). Fastest and most reliable if you know it.
  2. --ref-length-cm + --ref-length-px: Measure a known real distance in the image.
  3. --aruco: Auto-detect ArUco marker in image for automatic calibration.
     Requires a 4x4_50 ArUco marker visible in the image(s).
     Set --aruco-size-cm to match the physical marker size (default: 5.0 cm).
  4. Without scale: Run percent-only mode. Coverage % will be reported,
     but physical area (cm²) will not be calculated.

OUTPUT:
  - out/results.csv: Image names, coverage percentages, and cm² (if scaled)
  - out/*_annot.jpg: Annotated images showing:
    * Yellow contours: detected metal panel
    * Green contours: detected rust regions
    * Text overlay: rust coverage percentage

PANEL DETECTION:
  - Default: Targets gray/low-saturation painted metal surfaces, biased to image center
  - --prefer-vertical: Optimized for tall objects (pipes, poles, vertical structures)

RUST DETECTION:
  - Uses dual HSV + LAB color thresholding for robust orange/brown/red rust identification
  - More robust to lighting variations than simple HSV alone
  - Applies morphological opening/closing to reduce noise

ADVANTAGES OVER rust_estimate.py:
  - CLAHE preprocessing for better contrast and lighting adaptation
  - Dual color-space detection (HSV + LAB) for improved accuracy
  - ArUco marker support for automatic scale calibration
  - Better handling of vertical objects (pipes/poles)
  - Configurable output directory
  - More robust to diverse lighting conditions

NOTES:
  - ArUco support requires OpenCV 4.7+ with aruco module. Falls back gracefully if unavailable.
  - For best results, ensure the rust and panel are clearly visible and distinct in color.
  - Vertical bias (--prefer-vertical) works best when the main object is roughly centered.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--glob",
        default="images/*.jpg",
        help="Glob pattern for input images (default: images/*.jpg). Examples: images/*.jpg, photos/*.png, data/**/*.jpg"
    )
    ap.add_argument(
        "--cm-per-pixel",
        type=float,
        default=None,
        help="Direct scale in cm/pixel. Use this if you know the exact scale of your camera/lens setup."
    )
    ap.add_argument(
        "--ref-length-cm",
        type=float,
        default=None,
        help="Physical length (cm) of a known reference object in the image. Must be paired with --ref-length-px."
    )
    ap.add_argument(
        "--ref-length-px",
        type=float,
        default=None,
        help="Measured pixel distance for the reference length in the image. Must be paired with --ref-length-cm."
    )
    ap.add_argument(
        "--aruco",
        action="store_true",
        help="Enable ArUco marker detection for automatic scale calibration. Requires a 4x4_50 ArUco marker visible in images."
    )
    ap.add_argument(
        "--aruco-size-cm",
        type=float,
        default=5.0,
        help="Physical width of the ArUco marker in centimeters (default: 5.0). Only used with --aruco."
    )
    ap.add_argument(
        "--prefer-vertical",
        action="store_true",
        help="Optimize panel detection for vertical objects (pipes, poles). Enhances detection of tall structures."
    )
    ap.add_argument(
        "--out-dir",
        default="out",
        help="Output directory for results.csv and annotated images (default: out/). Created if it doesn't exist."
    )
    return ap.parse_args()

# -------------------------- Preprocessing & masks --------------------------
def apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2,A,B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def panel_mask_heuristic(img_bgr, prefer_vertical=False):
    """Create a panel mask for metal/painted surface. Heuristic using low saturation gray-ish regions.
       If prefer_vertical=True, we enhance vertical continuity to select a tall component (pipes/poles)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    gray_like = (S < 60) & (V > 80) & (V < 250)
    gray_like = gray_like.astype(np.uint8) * 255

    if prefer_vertical:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 51))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(gray_like, cv2.MORPH_CLOSE, k, iterations=1)

    # pick largest (and optionally most central) component
    cnts,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h,w), np.uint8)
    if cnts:
        cx_bias = w // 2
        best = None
        best_score = -1
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1000:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"])
            score = float(area)
            # slight center bias
            score *= (1.0 / (1 + abs(cx - cx_bias)))
            # vertical bias if requested
            if prefer_vertical:
                x,y,ww,hh = cv2.boundingRect(c)
                score *= (1 + (hh / max(ww,1)))
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    if not np.any(mask):
        mask[:] = closed  # fallback
    return mask

def rust_mask(img_bgr):
    """Combine HSV and LAB thresholds to robustly pick rust."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)

    # HSV rust: orange/brown hues with some saturation
    rust_hsv = cv2.inRange(hsv, np.array([5, 40, 30]), np.array([30, 255, 255]))
    # LAB rust: high a (red) and b (yellow) channels
    rust_lab = cv2.inRange(np.dstack([a,b]), np.array([135, 130]), np.array([255, 255]))

    comb = cv2.bitwise_and(rust_hsv, rust_lab)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, k, iterations=1)
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k, iterations=1)
    return comb

# -------------------------- Scale helpers --------------------------
def resolve_cm_per_pixel(args, first_img=None):
    # 1) Direct value
    if args.cm_per_pixel and args.cm_per_pixel > 0:
        return args.cm_per_pixel
    # 2) Known reference (cm and measured px)
    if args.ref_length_cm and args.ref_length_px and args.ref_length_px > 0:
        return args.ref_length_cm / args.ref_length_px
    # 3) ArUco if requested
    if args.aruco and first_img is not None and ARUCO is not None:
        s = cm_per_pixel_from_aruco(first_img, marker_size_cm=args.aruco_size_cm)
        return s
    return None  # percent-only

# -------------------------- Core --------------------------
def process_image(img_bgr, panel_vertical=False, cm_per_px=None):
    img2 = apply_clahe_bgr(img_bgr)
    pmask = panel_mask_heuristic(img2, prefer_vertical=panel_vertical)
    rmask = rust_mask(img2)
    valid = cv2.bitwise_and(rmask, pmask)

    rust_px  = int(cv2.countNonZero(valid))
    panel_px = int(cv2.countNonZero(pmask))
    coverage = 100.0 * rust_px / max(panel_px, 1)

    rust_cm2 = None
    if cm_per_px is not None:
        rust_cm2 = rust_px * (cm_per_px ** 2)

    # annotation
    ann = img_bgr.copy()
    cnts,_ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (0,255,255), 2)
    rcnts,_ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, rcnts, -1, (0,255,0), 2)  
    cv2.putText(ann, f"Rust coverage: {coverage:.2f}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(ann, f"Rust coverage: {coverage:.2f}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

    return ann, coverage, rust_cm2

def main():
    args = parse_args()
    fpaths = sorted(glob.glob(args.glob))
    if not fpaths:
        print("No images matched glob:", args.glob)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    first = cv2.imread(fpaths[0])
    cm_per_px = resolve_cm_per_pixel(args, first_img=first)

    rows = []
    for fp in fpaths:
        img = cv2.imread(fp)
        if img is None:
            continue
        ann, cov, cm2 = process_image(img, panel_vertical=args.prefer_vertical, cm_per_px=cm_per_px)
        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(args.out_dir, f"{base}_annot.jpg"), ann)
        row = [os.path.basename(fp), f"{cov:.2f}"]
        if cm_per_px is not None:
            row.append(f"{cm2:.2f}")
        rows.append(row)

    headers = ["image", "coverage_pct"] + (["rust_cm2"] if cm_per_px is not None else [])
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

    if cm_per_px is None:
        print("[INFO] Ran without physical scale -> CSV contains coverage_pct only.")
    else:
        print(f"[INFO] Using cm_per_pixel = {cm_per_px:.6f}")
    print("[OK] Done. Output in:", args.out_dir)

if __name__ == "__main__":
    main()
