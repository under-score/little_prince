#!/usr/bin/env python3
# ============================================================
#  Image Noise Type Analyzer
#  Author: chat GPT 5.0
#  Date:   2025-11-12
#
#  Based on principles from:
#    - Immerkaer J. (1996) *A Fast Noise Estimator for Images.* CVIU, 64(2):300–302
#    - Trussell H.J. & Foi A. (2009) *Noise estimation and removal in digital images.*
#      IEEE Signal Processing Magazine 26(5):20–31
#    - Ponomarenko et al. (2013) *Visual quality metrics for noise-like textures.*
#
#  Purpose:
#    Automatically discriminate true film grain from artificial or scanner noise
#    by analysing the power spectral density (PSD) slope, spectral flatness, and
#    RGB-channel independence in locally uniform regions.
# ============================================================

import cv2, numpy as np, matplotlib.pyplot as plt, os, datetime
from scipy import stats as sps

# ========= USER SETTINGS =========

directory = "" 
IMAGE_LIST = [
    os.path.join(directory,f)
    for f in os.listdir(directory)
    if os.path.isfile(os.path.join(directory, f)) and not f.startswith("B")
]

OUTPUT_DIR   = ""
MODE         = "AUTO"     # "AUTO" or "MANUAL"
MANUAL_ROI   = (600, 400, 192, 192)
window_size  = 15
percentile   = 5
patch_max    = 192
hp_cut       = 0.25
hp_order     = 2
fit_low      = 0.30
fit_high     = 0.90
n_log_bins   = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# ========= CORE FUNCTIONS =========
def butterworth_highpass(shape, cutoff_frac, order=2):
    r,c=shape; cy,cx=r//2,c//2
    Y,X=np.ogrid[:r,:c]; R=np.sqrt((X-cx)**2+(Y-cy)**2)
    nyq=min(shape)/2.0; D0=max(1e-6,cutoff_frac*nyq)
    return (1.0/(1.0+(D0/np.maximum(R,1e-6))**(2*order))).astype(np.float32)

def radial_psd_logbins(PSD, n_bins=40):
    """Compute radial PSD averaged in log-spaced frequency bins; robust to empty inputs."""
    rows, cols = PSD.shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    nyq = min(rows, cols) / 2.0
    f = (R / nyq).ravel()
    p = PSD.ravel()

    # mask out invalid values early
    valid = (f > 0) & np.isfinite(p) & (p > 0)
    if not np.any(valid):
        # nothing usable
        return np.array([]), np.array([])

    f, p = f[valid], p[valid]
    fmin = max(f.min(), 1.0 / min(rows, cols))
    edges = np.logspace(np.log10(fmin), np.log10(1.0), n_bins + 1)
    bins = 0.5 * (edges[:-1] + edges[1:])
    psd_binned = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (f >= edges[i]) & (f < edges[i + 1])
        psd_binned[i] = p[sel].mean() if np.any(sel) else np.nan
    keep = np.isfinite(psd_binned) & (psd_binned > 0)
    return bins[keep], psd_binned[keep]

def spectral_flatness(power):
    p=np.asarray(power,float); p=p[p>0]
    if not len(p): return np.nan
    return float(np.exp(np.mean(np.log(p)))/(np.mean(p)+1e-12))

def detrend_and_window(patch):
    sigma=max(3,min(patch.shape)/16.0); ksz=int(round(sigma*6))|1
    base=cv2.GaussianBlur(patch,(ksz,ksz),sigma,sigma,borderType=cv2.BORDER_REFLECT)
    hp=patch-base; hp-=hp.mean()
    wy,wx=np.hanning(hp.shape[0]),np.hanning(hp.shape[1])
    return hp*np.outer(wy,wx).astype(np.float32)

def analyze_patch(gray,x1,y1,w,h,hp_cut_,fit_low_,fit_high_):
    patch=gray[y1:y1+h,x1:x1+w].astype(np.float32)
    hp_win=detrend_and_window(patch)
    F=np.fft.fftshift(np.fft.fft2(hp_win))
    Hhp=butterworth_highpass(hp_win.shape,hp_cut_,hp_order)
    PSD=(np.abs(F*Hhp)**2).astype(np.float64)
    f_bins,psd_bins=radial_psd_logbins(PSD,n_bins=n_log_bins)
    if len(f_bins) == 0:
        return None  # no usable frequency data
    low,high=fit_low_,fit_high_
    for _ in range(4):
        band=(f_bins>=low)&(f_bins<=high)&np.isfinite(psd_bins)&(psd_bins>0)
        if np.count_nonzero(band)>=10: break
        low=max(0.20,low*0.85); high=min(0.95,high*1.05)
    if np.count_nonzero(band)<10: return None
    xlog,ylog=np.log10(f_bins[band]),np.log10(psd_bins[band])
    slope,intercept,r,_,_=sps.linregress(xlog,ylog)
    sfm=spectral_flatness(psd_bins[band])
    return dict(f_bins=f_bins,psd_bins=psd_bins,band=band,
                slope=slope,r2=r**2,intercept=intercept,
                sfm=sfm,low=low,high=high)

def interpret_noise(slope, r2, flatness, mean_rgb_corr=None, K_knee=None, br_ratio=None):
    """
    Interpret PSD slope, R², spectral flatness, RGB correlation, 
    and new 'minilab exclusion' metrics (K_knee, Blue/Red ratio).
    Returns textual class and hints.
    """

    # --- Thresholds ---
    S_FLAT_SYN  = 0.80
    S_FILM_MIN  = -2.0
    S_FILM_MAX  = -0.8
    S_PRINT_MIN = -3.5
    S_PRINT_MAX = -2.3
    R2_OK       = 0.75
    CORR_LOW    = 0.6
    CORR_HIGH   = 0.9
    K_FILMSCAN  = 1.0   # knee > 1.0 → likely film-scan
    K_DIGITAL   = 0.5   # knee < 0.5 → not film-scan
    BR_DIGITAL  = 1.3   # blue/red HF bias typical for sensors

    hints = []

    # --- Core classification by slope + flatness ---
    if flatness >= S_FLAT_SYN and slope > -1.2:
        cls = "Likely synthetic / white-like noise"
        hints += ["Very high spectral flatness (white-ish).",
                  "Slope near 0 to -1 supports a flat digital spectrum."]

    elif S_FILM_MIN <= slope <= -1.2 and flatness < S_FLAT_SYN:
        cls = "Likely film-like 1/f^α grain"
        hints += ["Slope in −1.2…−2 range fits classic film grain.",
                  "Flatness well below 1.0 (colored 1/f spectrum)."]

    elif S_PRINT_MIN <= slope <= S_PRINT_MAX and flatness < 0.8:
        cls = "Digitized film print (MTF-limited 1/f³ grain)"
        hints += ["Steep slope (−2.3…−3.5) consistent with print-scanner low-pass filtering.",
                  "Represents real film grain blurred by paper and scanner optics."]

    elif slope < S_FILM_MIN and flatness < 0.6:
        cls = "Blur / MTF-dominated region (no visible grain)"
        hints += ["Very steep slope (<−2) and low flatness → optical or scanner blur.",
                  "Little or no residual high-frequency texture."]

    else:
        cls = "Indeterminate / mixed characteristics"
        hints += ["Metrics don’t align clearly; consider another ROI or adjust fit bands."]

    # --- Fit quality ---
    if r2 < R2_OK:
        hints += ["Power-law fit only moderate (R² < 0.75)."]

    # --- RGB correlation ---
    if mean_rgb_corr is not None:
        if mean_rgb_corr >= CORR_HIGH:
            hints += [f"RGB spectra nearly identical (corr={mean_rgb_corr:.2f}) → digital or scanner-added noise likely."]
        elif mean_rgb_corr <= CORR_LOW:
            hints += [f"RGB spectra largely independent (corr={mean_rgb_corr:.2f}) → color film layers likely."]
        else:
            hints += [f"RGB correlation intermediate (corr={mean_rgb_corr:.2f})."]

    # --- Knee test (film-scan vs digital) ---
    if K_knee is not None:
        if K_knee >= K_FILMSCAN:
            hints += [f"Two-band PSD knee K={K_knee:.2f} → possible film-scan minilab print."]
        elif K_knee < K_DIGITAL:
            hints += [f"No pronounced PSD knee (K={K_knee:.2f}) → not film-scan, likely digital print/re-photograph."]

    # --- Blue/Red high-frequency ratio ---
    if br_ratio is not None and np.isfinite(br_ratio):
        if br_ratio >= BR_DIGITAL:
            hints += [f"Blue/Red HF RMS={br_ratio:.2f} ≥ {BR_DIGITAL} → sensor-type noise bias (digital)."]
        else:
            hints += [f"Blue/Red HF RMS={br_ratio:.2f} → balanced HF energy (film more likely)."]

    # --- Final override: film-scan exclusion ---
    if (K_knee is not None and K_knee < K_DIGITAL) and \
       (mean_rgb_corr is not None and mean_rgb_corr >= 0.90) and \
       (br_ratio is not None and br_ratio >= BR_DIGITAL):
        cls = "Likely digital print / re-photograph (minilab film-scan excluded)"
        hints += ["Knee <0.5, RGB corr ≥0.9, Blue/Red HF ≥1.3 → minilab film-scan excluded."]

    elif (K_knee is not None and K_knee >= 1.0) and \
         (mean_rgb_corr is not None and mean_rgb_corr <= 0.85):
        cls = "Possible film-scan minilab print"
        hints += ["Two-slope PSD with moderate RGB independence → hybrid film-scan print."]

    return cls, hints

# ========= MAIN LOOP =========
results=[]
for image_path in IMAGE_LIST:
    fname=os.path.basename(image_path)
    img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Cannot read {fname}")
        continue
    H,W=img.shape[:2]; is_color=len(img.shape)==3 and img.shape[2]>=3
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if is_color else img

    # --- ROI SELECTION (simple & safe) ---
    if MODE.upper() == "MANUAL":
        x1, y1, w_, h_ = MANUAL_ROI
    else:
        f32 = gray.astype(np.float32)
        mean = cv2.blur(f32, (window_size, window_size))
        mean_sq = cv2.blur(f32**2, (window_size, window_size))
        local_var = mean_sq - mean**2
    
        thresh = np.percentile(local_var, percentile)
        mask = (local_var < thresh).astype(np.uint8) * 255
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
        if num > 1:
            i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            cx, cy = map(int, centroids[i])
            half = patch_max // 2
            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
        else:
            print(f"⚠️ {fname}: No low-variance zone found; using center.")
            x1 = max(W // 2 - patch_max // 2, 0)
            y1 = max(H // 2 - patch_max // 2, 0)
    
    w_ = min(patch_max, W - x1)
    h_ = min(patch_max, H - y1)
    res = analyze_patch(gray, x1, y1, w_, h_, hp_cut, fit_low, fit_high)

    if res is None:
        print(f"⚠️ {fname}: ROI too uniform — using larger or looser patch.")
        # try expanding the patch slightly or relaxing filter
        half = min(W, H) // 3
        x1 = max(W // 2 - half, 0)
        y1 = max(H // 2 - half, 0)
        w_ = min(2 * half, W - x1)
        h_ = min(2 * half, H - y1)
        res = analyze_patch(gray, x1, y1, w_, h_, hp_cut * 0.5, fit_low * 0.8, fit_high)
        if res is None:
            print(f"⚠️ {fname}: Still no usable data — skipping image.")
            continue

    slope,r2,flatness=res["slope"],res["r2"],res["sfm"]
    low,high=res["low"],res["high"]

    # optional RGB analysis
    mean_rgb_corr=None
    if is_color:
        spectra=[]
        for ci in range(3):
            ch=img[y1:y1+h_,x1:x1+w_,ci].astype(np.float32)
            rch=analyze_patch(ch,0,0,w_,h_,hp_cut,fit_low,fit_high)
            if rch: spectra.append(np.log10(rch["psd_bins"][rch["band"]]))
        if len(spectra)==3:
            c_rg=np.corrcoef(spectra[1],spectra[2])[0,1]
            c_rb=np.corrcoef(spectra[2],spectra[0])[0,1]
            c_gb=np.corrcoef(spectra[1],spectra[0])[0,1]
            mean_rgb_corr=np.nanmean([c_rg,c_rb,c_gb])

    # Interpretation
    cls,hints=interpret_noise(slope,r2,flatness,mean_rgb_corr)
    print(f"\n📄 {fname}")
    print(f"Overall slope={slope:.2f}, R²={r2:.3f}, flatness={flatness:.3f}")
    print(f"→ {cls}")
    for h in hints: print("  -",h)

    # --- Save PSD diagnostic plot for this image ---
    plt.figure(figsize=(14, 5))
    
    # Left: image with ROI
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_color else img, cmap="gray")
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), w_, h_, edgecolor="r", facecolor="none", lw=2)
    )
    plt.axis("off")
    plt.title(f"{fname}\nROI (red)  slope={slope:.2f}")
    
    # Right: PSD plot
    f_bins, psd_bins, band = res["f_bins"], res["psd_bins"], res["band"]
    plt.subplot(1, 2, 2)
    plt.loglog(f_bins, psd_bins, label="PSD")
    fb = (f_bins >= low) & (f_bins <= high)
    if np.any(fb):
        plt.fill_between(
            f_bins[fb],
            psd_bins[fb],
            np.min(psd_bins[fb]) * 0.9,
            alpha=0.2,
            step="mid",
            label=f"fit [{low:.2f}-{high:.2f}]",
        )
    x_fit = np.logspace(
        np.log10(max(low, f_bins.min())), np.log10(min(high, f_bins.max())), 50
    )
    y_fit = 10 ** (res["intercept"]) * (x_fit ** (slope))
    plt.loglog(x_fit, y_fit, "--", label=f"slope={slope:.2f}, R²={r2:.3f}")
    plt.xlabel("Normalized freq (Nyquist=1)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    
    # --- Save the plot instead of showing ---
    out_plot = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_psd.png")
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Saved plot → {out_plot}")
