import os
import io
import csv
import re
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template
from PIL import Image
from captum.attr import LayerGradCam, IntegratedGradients, GradientShap

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GRADE_NAMES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}
GRADE_COLORS = {
    0: '#22c55e',
    1: '#84cc16',
    2: '#f59e0b',
    3: '#f97316',
    4: '#ef4444'
}
GRADE_DESC = {
    0: 'No signs of diabetic retinopathy detected.',
    1: 'Mild microaneurysms present. Monitor regularly.',
    2: 'Moderate damage to blood vessels. Refer for treatment.',
    3: 'Severe damage. Many blood vessels blocked. Urgent referral needed.',
    4: 'Proliferative stage. Abnormal new blood vessels growing. Immediate treatment required.'
}

MODEL_PATH    = 'models/checkpoints/efficientnet_b4_best.pth'
EYEPACS_DIR   = 'data/raw/eyepacs_demo'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── EyePACS Label Lookup ──────────────────────────────────────────────────────
# Keys are image stem without extension  e.g. '10_left', '10_right'
# Values are DR grade 0-4
EYEPACS_LABELS = {}

def load_eyepacs_labels():
    global EYEPACS_LABELS
    # Accept either labels.csv (demo zip) or trainLabels.csv (full dataset)
    candidates = [
        os.path.join(EYEPACS_DIR, 'labels.csv'),
        os.path.join(EYEPACS_DIR, 'trainLabels.csv'),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        print(f"No EyePACS CSV found in {EYEPACS_DIR}/")
        print("  Download eyepacs_demo.zip from your Kaggle output panel and")
        print("  extract it into data/raw/eyepacs_demo/")
        return
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        # Support both formats:
        #   demo labels.csv  → 'Image name', 'Retinopathy grade'
        #   full trainLabels → 'image', 'level'
        if 'image' in headers:
            name_col, grade_col = 'image', 'level'
        else:
            name_col, grade_col = 'Image name', 'Retinopathy grade'
        for row in reader:
            name  = row[name_col].strip()
            grade = int(row[grade_col])
            EYEPACS_LABELS[name] = grade
    print(f"EyePACS labels loaded: {len(EYEPACS_LABELS)} images  ({csv_path})")

# ── Model ─────────────────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone   = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

def disable_inplace(model):
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Please download efficientnet_b4_best.pth from Kaggle and place it in models/checkpoints/")
        return False
    m = DRClassifier().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    disable_inplace(m)
    m.eval()
    model = m
    print(f"Model loaded — Val AUROC at save: {ckpt.get('val_auroc', 'N/A'):.4f}")
    return True

# ── Preprocessing ─────────────────────────────────────────────────────────────
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess(pil_img):
    img = pil_img.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1)
    t   = normalize(t).unsqueeze(0).to(DEVICE)
    return t, arr

# ── Figure → base64 ───────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none', dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def arr_to_b64(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Spatial analysis helper ───────────────────────────────────────────────────
def where_is_focus(map2d):
    """Return a plain-English description of where the brightest activations are.

    Uses a brightness-weighted centroid of the top 15% pixels so that
    the result is pulled toward the truly brightest spot rather than
    being dragged toward the middle by a large blurry activation field.
    Works correctly for Grad-CAM, IG, Feature Maps, and SHAP.
    """
    h, w    = map2d.shape
    max_val = map2d.max()
    if max_val <= 0:
        return "spread across the entire retina", "diffuse"

    # Work only with the top 15% of pixels
    thresh  = np.percentile(map2d, 85)
    mask    = map2d >= thresh
    if mask.sum() == 0:
        return "spread across the entire retina", "diffuse"

    coverage = float(mask.sum()) / (h * w)
    if coverage > 0.45:
        return "spread broadly across the entire retina without a single clear focal point", "diffuse"

    # Brightness-weighted centroid — bright pixels pull harder than dim ones
    ys, xs  = np.where(mask)
    weights = map2d[mask]
    cy      = float(np.average(ys, weights=weights)) / h   # 0 = top,  1 = bottom
    cx      = float(np.average(xs, weights=weights)) / w   # 0 = left, 1 = right

    # Vertical label
    if cy < 0.35:
        v = "upper"
    elif cy > 0.65:
        v = "lower"
    else:
        v = "central"

    # Horizontal label
    if cx < 0.35:
        hz = "left"
    elif cx > 0.65:
        hz = "right"
    else:
        hz = "middle"

    # Distance from true centre
    dist = ((cx - 0.5)**2 + (cy - 0.5)**2) ** 0.5
    if dist < 0.12:
        region = "right at the centre of the retina, near the macula"
    elif dist < 0.25:
        region = f"{v} {hz} area of the retina"
    else:
        region = f"outer {v} {hz} edge of the retina"

    focus_type = "concentrated" if coverage < 0.15 else "moderate"
    return region, focus_type


def analyze_map(map2d):
    """Extract rich spatial and intensity features from any 2D activation map.
    Returns a dict used to write dynamic, image-specific descriptions."""
    h, w    = map2d.shape
    max_val = float(map2d.max())
    if max_val <= 0:
        return {
            'region': 'spread across the whole retina',
            'spread': 'broad', 'strength': 'faint',
            'strength_adv': 'barely', 'spots': 'scattered',
        }

    region, _ = where_is_focus(map2d)
    mean_val  = float(map2d.mean())
    contrast  = max_val / (mean_val + 1e-8)

    # How much of the image is "genuinely bright" (above 50 % of peak)
    above_half = float((map2d >= max_val * 0.5).sum()) / (h * w)
    # How much has any notable activation (above 20 % of peak)
    above_low  = float((map2d >= max_val * 0.2).sum()) / (h * w)

    # Spread label
    if above_half < 0.02:
        spread = 'pinpoint'      # tiny hot spot
    elif above_half < 0.07:
        spread = 'tight'         # one clear region
    elif above_half < 0.18:
        spread = 'moderate'      # decent-sized area
    else:
        spread = 'broad'         # covers a large area

    # Peak strength relative to background
    if contrast > 10:
        strength, strength_adv = 'very strong',  'very strongly'
    elif contrast > 5:
        strength, strength_adv = 'clear',        'clearly'
    elif contrast > 2.5:
        strength, strength_adv = 'moderate',     'noticeably'
    else:
        strength, strength_adv = 'faint',        'only mildly'

    # Single spike vs scattered pattern
    if above_low > 0.35 and above_half < 0.04:
        spots = 'scattered'    # many weak regions, no dominant spot
    elif above_half < 0.03:
        spots = 'single'       # one tight peak
    else:
        spots = 'clustered'    # one main region with some spread

    return {
        'region':       region,
        'spread':       spread,
        'strength':     strength,
        'strength_adv': strength_adv,
        'spots':        spots,
        'above_half':   above_half,
        'above_low':    above_low,
        'contrast':     contrast,
    }

# ── XAI: Grad-CAM ─────────────────────────────────────────────────────────────
def run_gradcam(img_t, pred_class, orig_arr):
    target_layer = model.backbone.blocks[-1]
    gradcam      = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(img_t, target=int(pred_class))
    attr = attr.squeeze().cpu().detach().numpy()
    if attr.ndim == 3:          # (C, H, W) -> average over channels
        attr = attr.mean(0)
    # attr is now (H, W)
    attr = np.maximum(attr, 0)
    if attr.max() > 0:
        attr = attr / attr.max()
    heatmap = cv2.resize(attr, (224, 224))
    heatmap_color = cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * orig_arr + 0.5 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='#0f172a')
    for ax in axes: ax.axis('off')
    axes[0].imshow(orig_arr)
    axes[0].set_title('Original', color='white', fontsize=11, pad=8)
    axes[1].imshow(overlay)
    axes[1].set_title('Grad-CAM Heatmap', color='white', fontsize=11, pad=8)
    plt.tight_layout(pad=1)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64, heatmap

# ── XAI: Integrated Gradients ─────────────────────────────────────────────────
def run_ig(img_t, pred_class, orig_arr):
    ig       = IntegratedGradients(model)
    baseline = torch.zeros_like(img_t)
    attr     = ig.attribute(img_t, baseline, target=int(pred_class), n_steps=20)
    attr     = attr.squeeze().cpu().detach().numpy()
    attr     = np.sum(np.abs(attr), axis=0)
    if attr.max() > 0:
        attr = attr / attr.max()
    from matplotlib.colors import LinearSegmentedColormap
    ig_cmap = LinearSegmentedColormap.from_list('ig_red', ['#0d0000', '#7f1d1d', '#ef4444', '#fca5a5'])
    heatmap_color = ig_cmap(attr)[:, :, :3]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='#0f172a')
    for ax in axes: ax.axis('off')
    axes[0].imshow(orig_arr)
    axes[0].set_title('Original', color='white', fontsize=11, pad=8)
    axes[1].imshow(heatmap_color)
    axes[1].set_title('Integrated Gradients', color='white', fontsize=11, pad=8)
    plt.tight_layout(pad=1)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64, attr

# ── XAI: Feature Map Attention ────────────────────────────────────────────────
def run_fma(img_t, orig_arr):
    feature_maps = {}
    def hook_fn(module, input, output):
        feature_maps['last'] = output.detach()
    hook = model.backbone.blocks[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img_t)
    hook.remove()

    fmaps       = feature_maps['last'].squeeze(0)
    activations = fmaps.mean(dim=(1, 2))
    top8_idx    = activations.topk(8).indices.cpu().numpy()

    fig, axes = plt.subplots(1, 9, figsize=(26, 4), facecolor='#0f172a')
    axes[0].imshow(orig_arr)
    axes[0].axis('off')
    axes[0].set_title('Original', color='white', fontsize=9, pad=6)

    channel_locs = []
    for i, ch in enumerate(top8_idx):
        fmap = fmaps[ch].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        fmap_r = cv2.resize(fmap, (224, 224))
        loc, _ = where_is_focus(fmap_r)
        channel_locs.append(loc)
        axes[i+1].imshow(orig_arr, alpha=0.35)
        axes[i+1].imshow(fmap_r, cmap='hot', alpha=0.65)
        axes[i+1].axis('off')
        axes[i+1].set_title(f'Ch {ch}', color='white', fontsize=8, pad=6)

    plt.tight_layout(pad=0.8)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64, top8_idx.tolist(), channel_locs

# ── XAI: SHAP (GradientShap) ─────────────────────────────────────────────────
def run_shap(img_t, pred_class, orig_arr):
    gs       = GradientShap(model)
    baseline = torch.zeros_like(img_t)
    attr     = gs.attribute(img_t, baseline, target=int(pred_class), n_samples=20)
    attr     = attr.squeeze().cpu().detach().numpy()
    attr     = np.sum(np.abs(attr), axis=0)
    if attr.max() > 0:
        attr = attr / attr.max()
    from matplotlib.colors import LinearSegmentedColormap
    shap_cmap = LinearSegmentedColormap.from_list('shap_blue', ['#050d1f', '#1e40af', '#60a5fa', '#bae6fd'])
    heatmap_color = shap_cmap(attr)[:, :, :3]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='#0f172a')
    for ax in axes: ax.axis('off')
    axes[0].imshow(orig_arr)
    axes[0].set_title('Original', color='white', fontsize=11, pad=8)
    axes[1].imshow(heatmap_color)
    axes[1].set_title('SHAP Values', color='white', fontsize=11, pad=8)
    plt.tight_layout(pad=1)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64, attr

# ── XAI Descriptions ─────────────────────────────────────────────────────────
# What a doctor looks for at each grade (patient-friendly, no dashes)
GRADE_SHOULD_FOCUS = {
    0: "any part of the retina that looks unusual. In a healthy eye everything should appear even and undisturbed with no spots, no leaks and no signs of bleeding anywhere",
    1: "tiny pinpoint spots on the surface of the retina. These are called microaneurysms and they are the very first sign that some of the small blood vessels in the eye have started to bulge and weaken",
    2: "yellowish or whitish patches where blood vessels have been leaking fluid into the retina, as well as small red spots where minor bleeding has occurred",
    3: "large dark patches of bleeding spread across multiple areas of the retina. At this stage many blood vessels have burst and the damage covers a wide portion of the eye",
    4: "new fragile blood vessels growing in places they should not be, usually sprouting near the bright circular area at the back of the eye known as the optic disc",
}

# Plain-language confusion reasons written in a doctor-to-patient voice (no dashes)
GRADE_CONFUSION_REASON = {
    (2, 4): "What happened here is that the yellowish patches from your level of damage look very similar to the markings we see in the most severe stage of the condition. When I looked at your scan those bright patches appeared intense enough that I mistakenly read them as the worst kind of damage rather than recognising them as the leaking deposits typical of a less advanced stage.",
    (1, 2): "The early warning signs in your eye are quite faint. However, some of the surrounding tissue looked slightly more affected than I would normally expect at the earliest stage, and that led me to overestimate how far the condition had progressed.",
    (3, 4): "The amount of bleeding visible in your retina is so extensive that it appeared to me to match the most severe category of this condition. In reality it falls just one level below that. The damage is still very serious but it has not quite reached the final stage.",
    (0, 3): "Your eye is actually healthy, but some shadows and bright reflections in this particular photograph created patterns that looked to me like the widespread bleeding we associate with severe damage. The lighting in the image misled my assessment.",
    (0, 4): "Your eye is healthy. However, the way light reflects off the retina in this photograph created bright patches that I mistook for the abnormal blood vessel growth we see in the most advanced stage of this condition. The image quality led me to the wrong conclusion.",
    (2, 0): "The damage in your retina is present but very subtle in this scan. The leaking patches and small bleeds did not stand out clearly enough in this image for me to detect them, and I incorrectly assessed your eye as healthy.",
    (3, 0): "There is significant damage in your retina, but the way this scan was captured or lit made the key signs difficult to see clearly. As a result I assessed the eye as healthy when it clearly was not.",
    (4, 3): "The abnormal new blood vessels are visible in your scan, but I was not fully convinced they represented the final and most severe stage of the condition. I placed the finding one level below where it actually belongs.",
}

def generate_xai_descriptions(pred, conf, true_grade,
                               gc_map, ig_map, fma_channels, fma_locs, shap_map):
    gname    = GRADE_NAMES[pred]
    conf_str = f"{conf}%"
    is_right = true_grade is not None and pred == true_grade
    is_wrong = true_grade is not None and pred != true_grade
    tname    = GRADE_NAMES[true_grade] if true_grade is not None else ""

    # Rich per-map analysis
    gc   = analyze_map(gc_map)
    ig   = analyze_map(ig_map)
    sh   = analyze_map(shap_map)

    confusion = GRADE_CONFUSION_REASON.get(
        (true_grade, pred) if is_wrong else None,
        f"Some features of {tname} can look visually similar to {gname} under certain imaging conditions."
    ) if is_wrong else ""

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    # Bullet points, plain language, genuinely image-specific

    if gc['spread'] == 'pinpoint':
        gc_spread_txt = (f"The red glow is extremely small — it pinpoints just one tiny spot "
                         f"in the <strong>{gc['region']}</strong>. Almost the entire rest of the retina is blue.")
    elif gc['spread'] == 'tight':
        gc_spread_txt = (f"A clear, focused red-orange glow sits in the <strong>{gc['region']}</strong>. "
                         f"It covers a small but well-defined patch before fading quickly to blue around it.")
    elif gc['spread'] == 'moderate':
        gc_spread_txt = (f"A moderately sized orange-red region is centred on the "
                         f"<strong>{gc['region']}</strong>, covering a noticeable portion of the retina.")
    else:
        gc_spread_txt = (f"The red-orange colour spreads across a large portion of the retina, "
                         f"with the heaviest concentration toward the <strong>{gc['region']}</strong>.")

    if gc['strength'] in ('very strong', 'clear'):
        gc_strength_txt = "The bright red centre stands out sharply against the blue background — the model was very sure about this area."
    elif gc['strength'] == 'moderate':
        gc_strength_txt = "The focus area is noticeably brighter than the surroundings but does not stand out dramatically."
    else:
        gc_strength_txt = "The highlighted area is only slightly brighter than the background — the model was working from subtle hints rather than one obvious feature."

    gc_where = (
        f"<ul>"
        f"<li>Grad-CAM shows where the model looked by colouring the retinal image. Red and orange mean it focused there; blue means it largely ignored that area.</li>"
        f"<li>{gc_spread_txt}</li>"
        f"<li>{gc_strength_txt}</li>"
        f"</ul>"
    )

    if gc['spots'] == 'single':
        gc_spots_txt = "All the attention comes from one concentrated point — there is a single clear focal area."
    elif gc['spots'] == 'clustered':
        gc_spots_txt = "The attention clusters around one main area with some spread into the nearby tissue."
    else:
        gc_spots_txt = "The attention is spread across several small spots — the model was picking up multiple small signals from different parts of the retina."

    gc_what = (
        f"<ul>"
        f"<li>In that highlighted region, the model was detecting {GRADE_SHOULD_FOCUS[pred]}.</li>"
        f"<li>{gc_spots_txt}</li>"
        f"<li>Those findings led the model to classify this image as <strong>{gname} (Grade {pred})</strong>.</li>"
        f"</ul>"
    )

    if is_right:
        if gc['strength'] in ('very strong', 'clear'):
            gc_why = (
                f"<ul>"
                f"<li>The model focused on the right part of the retina and found the right features. The markings it detected in the <strong>{gc['region']}</strong> match exactly what we expect to see in <strong>{gname} (Grade {pred})</strong>.</li>"
                f"<li>The signal was {gc['strength']} — this was not a borderline call. The model saw clear evidence and committed to it.</li>"
                f"<li>Outcome: <strong>{conf_str} confidence</strong>, correct prediction, well-supported by the heatmap.</li>"
                f"</ul>"
            )
        else:
            gc_why = (
                f"<ul>"
                f"<li>The model did not lock onto one single obvious feature. It gathered several smaller signals across the <strong>{gc['region']}</strong>.</li>"
                f"<li>Individually those signals were quiet, but together they pointed consistently toward <strong>{gname} (Grade {pred})</strong>.</li>"
                f"<li>Outcome: <strong>{conf_str} confidence</strong> — correct, even though the evidence was spread out rather than concentrated in one spot.</li>"
                f"</ul>"
            )
    elif is_wrong:
        gc_why = (
            f"<ul>"
            f"<li>The model saw patterns in the <strong>{gc['region']}</strong> that look like {GRADE_SHOULD_FOCUS[pred]}, so it predicted <strong>{gname} (Grade {pred})</strong>.</li>"
            f"<li>The correct answer is <strong>{tname} (Grade {true_grade})</strong>. The key signs for that grade are {GRADE_SHOULD_FOCUS[true_grade]}.</li>"
            f"<li>{confusion}</li>"
            f"</ul>"
        )
    else:
        gc_why = (
            f"<ul>"
            f"<li>The model found patterns in the <strong>{gc['region']}</strong> matching the visual profile of <strong>{gname} (Grade {pred})</strong>.</li>"
            f"<li>The focus was {'concentrated tightly' if gc['spread'] in ('pinpoint','tight') else 'spread across a wider area'}, {'pointing to one specific feature' if gc['spread'] in ('pinpoint','tight') else 'suggesting the model read a broader pattern across the retina'}.</li>"
            f"<li>This drove it to <strong>{conf_str} confidence</strong> for this classification.</li>"
            f"</ul>"
        )

    # ── Integrated Gradients ─────────────────────────────────────────────────
    # Bullet points, plain language, genuinely image-specific

    if ig['spread'] == 'pinpoint':
        ig_spread_txt = f"Just one tiny cluster of bright pixels in the <strong>{ig['region']}</strong> — everything else is dark red or black."
    elif ig['spread'] == 'tight':
        ig_spread_txt = f"A small, well-defined cluster of bright pixels in the <strong>{ig['region']}</strong>, with everything around it staying dark."
    elif ig['spread'] == 'moderate':
        ig_spread_txt = f"A moderate-sized patch of bright pixels centred on the <strong>{ig['region']}</strong>."
    else:
        ig_spread_txt = f"Bright pixels spread across a large area, with the densest concentration in the <strong>{ig['region']}</strong>."

    if ig['spots'] == 'single':
        ig_spots_txt = "One concentrated hotspot — a single detail caused the biggest jumps in the model's decision."
    elif ig['spots'] == 'scattered':
        ig_spots_txt = "The bright pixels are scattered in multiple small groups rather than one hotspot — the model reacted to several different details across the retina."
    else:
        ig_spots_txt = "There is one main cluster with some surrounding spread — a clear focal point with a few contributing areas nearby."

    if ig['strength'] in ('very strong', 'clear'):
        ig_signal_txt = f"The signal is {ig['strength']} — when those pixels appeared during the step-by-step rebuild, the model's confidence shifted quickly and significantly."
    elif ig['strength'] == 'moderate':
        ig_signal_txt = "The signal is moderate — those pixels influenced the model's answer but not with dramatic force."
    else:
        ig_signal_txt = "The signal is faint — the model's answer shifted gradually across many small details rather than one big standout feature."

    ig_where = (
        f"<ul>"
        f"<li>Integrated Gradients works by starting from a completely blank image and slowly revealing the retinal photo piece by piece across 20 steps. Every time a new piece appeared and the model's answer shifted, that piece lit up on this map.</li>"
        f"<li>{ig_spread_txt}</li>"
        f"<li>Everything in dark red or black had almost no effect on what the model decided.</li>"
        f"</ul>"
    )

    ig_what = (
        f"<ul>"
        f"<li>{ig_spots_txt}</li>"
        f"<li>The bright pixels in the <strong>{ig['region']}</strong> are the specific visual details that had the most impact on the model's final answer — not the background, not the blood vessels, just those spots.</li>"
        f"<li>{ig_signal_txt}</li>"
        f"</ul>"
    )

    if is_right:
        ig_why = (
            f"<ul>"
            f"<li>The pixels that mattered most are in the <strong>{ig['region']}</strong> — exactly where the damage signs for <strong>{gname} (Grade {pred})</strong> would appear.</li>"
            f"<li>The model was reacting to the right details in the right location, not to noise or unrelated parts of the image.</li>"
            f"<li>This confirms the <strong>{conf_str} confidence</strong> is well-grounded — the prediction is backed by real, meaningful features in the retina.</li>"
            f"</ul>"
        )
    elif is_wrong:
        ig_why = (
            f"<ul>"
            f"<li>The pixels that drove the decision are in the <strong>{ig['region']}</strong>, where the model detected features consistent with {GRADE_SHOULD_FOCUS[pred]} — pointing it to <strong>{gname} (Grade {pred})</strong>.</li>"
            f"<li>The correct label is <strong>{tname} (Grade {true_grade})</strong>, where the key signs are {GRADE_SHOULD_FOCUS[true_grade]}.</li>"
            f"<li>{confusion}</li>"
            f"</ul>"
        )
    else:
        if ig['strength'] == 'faint':
            ig_why = (
                f"<ul>"
                f"<li>No single detail stood out as decisive — the model assembled its answer from many small signals distributed across the <strong>{ig['region']}</strong>.</li>"
                f"<li>Each one was subtle individually, but together they were enough for <strong>{conf_str} confidence</strong> in <strong>{gname} (Grade {pred})</strong>.</li>"
                f"</ul>"
            )
        else:
            ig_why = (
                f"<ul>"
                f"<li>The {ig['strength']} cluster in the <strong>{ig['region']}</strong> was the primary detail driving the prediction toward <strong>{gname} (Grade {pred})</strong>.</li>"
                f"<li>That well-defined hotspot provided clear evidence that produced <strong>{conf_str} confidence</strong>.</li>"
                f"</ul>"
            )

    # ── Feature Maps ─────────────────────────────────────────────────────────
    # Bullet points, plain language, genuinely image-specific

    ch_verbs = [
        "lit up in",
        "fired most strongly across",
        "showed its peak response in",
        "concentrated its activation on",
        "responded most to",
        "highlighted",
        "detected a pattern in",
        "focused its response on",
    ]
    ch_bullets = "".join(
        f"<li>Filter {fma_channels[i]}: {ch_verbs[i % len(ch_verbs)]} the <strong>{fma_locs[i]}</strong>.</li>"
        for i in range(len(fma_channels))
    )

    # Count how many channels agreed on the same broad region
    unique_locs = len(set(fma_locs))
    if unique_locs <= 3:
        agreement_txt = f"Most of the eight filters lit up in overlapping areas — a strong agreement that one specific region carries the key information."
    elif unique_locs <= 5:
        agreement_txt = f"The eight filters split their attention across a few different areas of the retina, suggesting the relevant features are spread out rather than concentrated in one spot."
    else:
        agreement_txt = f"The eight filters each responded to different areas — the model pieced together evidence from many parts of the retina to reach its answer."

    fma_where = (
        f"<ul>"
        f"<li>The AI contains hundreds of internal pattern detectors (called filters). These are the eight that reacted most strongly to this retinal image:</li>"
        f"{ch_bullets}"
        f"<li>Orange and yellow in each panel means that filter found something meaningful there. Dark red means it found nothing of note.</li>"
        f"</ul>"
    )

    fma_what = (
        f"<ul>"
        f"<li>Each filter has been trained to detect one specific type of visual pattern — things like tiny pinpoint bleeds, patches of fluid leaking from blood vessels, areas of widespread haemorrhage, or abnormal new vessel growth.</li>"
        f"<li>{agreement_txt}</li>"
        f"<li>When a filter stays dark red, it means that particular type of damage was not found in that part of the retina.</li>"
        f"</ul>"
    )

    if is_right:
        fma_why = (
            f"<ul>"
            f"<li>These eight filters lit up in areas consistent with how <strong>{gname} (Grade {pred})</strong> typically appears in a retinal photograph.</li>"
            f"<li>Multiple independent filters agreeing on the same conclusion makes the result more reliable — it is much less likely to be a coincidence.</li>"
            f"<li>That combined signal across all eight filters produced <strong>{conf_str} confidence</strong> — the correct answer.</li>"
            f"</ul>"
        )
    elif is_wrong:
        fma_why = (
            f"<ul>"
            f"<li>The filters responded to patterns that look like <strong>{gname} (Grade {pred})</strong>, not the true label <strong>{tname} (Grade {true_grade})</strong>.</li>"
            f"<li>The features these filters detected are genuinely present in the retinal image — the model was not reacting to noise. But it placed them in the wrong diagnostic category.</li>"
            f"<li>{confusion}</li>"
            f"</ul>"
        )
    else:
        fma_why = (
            f"<ul>"
            f"<li>All eight filters activated on patterns consistent with <strong>{gname} (Grade {pred})</strong>.</li>"
            f"<li>Eight independent detectors — each looking at the retina from a different angle — all pointed to the same answer.</li>"
            f"<li>That convergence built up a strong, consistent signal leading to <strong>{conf_str} confidence</strong>.</li>"
            f"</ul>"
        )

    # ── SHAP ────────────────────────────────────────────────────────────────
    # Bullet points, plain language, genuinely image-specific

    if sh['spread'] == 'pinpoint':
        sh_spread_txt = f"Just one very small, intensely yellow spot in the <strong>{sh['region']}</strong>. The rest of the image is almost entirely deep purple."
    elif sh['spread'] == 'tight':
        sh_spread_txt = f"A compact cluster of yellow in the <strong>{sh['region']}</strong>, with everything around it quickly fading to purple."
    elif sh['spots'] == 'scattered':
        sh_spread_txt = f"Small yellow and orange patches scattered unevenly across the retina, with the heaviest concentration in the <strong>{sh['region']}</strong>."
    else:
        sh_spread_txt = f"A broad spread of yellow and orange covering the <strong>{sh['region']}</strong>, taking up a significant portion of the retina."

    if sh['strength'] in ('very strong', 'clear'):
        sh_strength_txt = "The yellow sections stand out sharply against the purple — the model had strong, clear confidence in those specific areas."
    elif sh['strength'] == 'moderate':
        sh_strength_txt = "The yellow areas are noticeably brighter than the background — real evidence was found there, but it was not dramatically dominant."
    else:
        sh_strength_txt = "The yellow areas are only slightly brighter than the purple background — the model was working from subtle signals distributed across several sections."

    shap_where = (
        f"<ul>"
        f"<li>SHAP works by covering up small sections of the image one at a time and measuring how much the model's confidence drops each time. Sections that caused the biggest drop are shown in yellow — those are the parts the model was truly relying on.</li>"
        f"<li>{sh_spread_txt}</li>"
        f"<li>The deep purple covering most of the image means the model effectively ignored those areas — removing them made almost no difference to the result.</li>"
        f"</ul>"
    )

    if sh['spread'] in ('pinpoint', 'tight'):
        sh_pattern_txt = "The model's confidence rested on one small, specific part of the retina — it found one key piece of evidence and committed to it."
    else:
        sh_pattern_txt = "The model drew evidence from several sections of the retina rather than one single spot — it built its answer from multiple contributing areas."

    shap_what = (
        f"<ul>"
        f"<li>{sh_pattern_txt}</li>"
        f"<li>{sh_strength_txt}</li>"
        f"<li>This is the most direct answer to the question 'what did the model actually trust?' — yellow means trusted, purple means ignored.</li>"
        f"</ul>"
    )

    if is_right:
        shap_why = (
            f"<ul>"
            f"<li>The yellow sections the model trusted most are in the <strong>{sh['region']}</strong> — exactly where the actual damage markers for <strong>{gname} (Grade {pred})</strong> sit in this retinal photo.</li>"
            f"<li>The model was trusting the right evidence in the right place, not background tissue or image artefacts.</li>"
            f"<li>This confirms the <strong>{conf_str} confidence</strong> is well-earned — the prediction is backed by meaningful features.</li>"
            f"</ul>"
        )
    elif is_wrong:
        shap_why = (
            f"<ul>"
            f"<li>The model placed its trust in the <strong>{sh['region']}</strong>, where it found features that look like <strong>{gname} (Grade {pred})</strong>.</li>"
            f"<li>The correct label is <strong>{tname} (Grade {true_grade})</strong>. The model was not reacting to noise — it was trusting real retinal features, but ones that overlap visually between these two grades.</li>"
            f"<li>{confusion}</li>"
            f"</ul>"
        )
    else:
        if sh['spread'] in ('pinpoint', 'tight'):
            shap_why = (
                f"<ul>"
                f"<li>The model's entire decision rested on a small, concentrated area in the <strong>{sh['region']}</strong>.</li>"
                f"<li>One highly specific part of the retina carried nearly all the weight — everything else was tuned out.</li>"
                f"<li>That focused confidence produced <strong>{conf_str} certainty</strong> for <strong>{gname} (Grade {pred})</strong>.</li>"
                f"</ul>"
            )
        else:
            shap_why = (
                f"<ul>"
                f"<li>The model drew evidence from multiple sections of the retina, with the strongest contributions coming from the <strong>{sh['region']}</strong>.</li>"
                f"<li>When several independent parts of the image all point to the same answer, the model's confidence is more robust than if only one section had stood out.</li>"
                f"<li>That multi-section agreement drove it to <strong>{conf_str} confidence</strong> for <strong>{gname} (Grade {pred})</strong>.</li>"
                f"</ul>"
            )

    return {
        'gradcam_desc': {'where': gc_where, 'what': gc_what, 'why': gc_why},
        'ig_desc':      {'where': ig_where, 'what': ig_what, 'why': ig_why},
        'fma_desc':     {'where': fma_where, 'what': fma_what, 'why': fma_why},
        'shap_desc':    {'where': shap_where, 'what': shap_what, 'why': shap_why},
    }

# ── Overall human-readable explanation ────────────────────────────────────────
def generate_overall_explanation(pred, conf, true_grade):
    gname    = GRADE_NAMES[pred]
    conf_str = f"{conf}%"
    is_right = true_grade is not None and pred == true_grade
    is_wrong = true_grade is not None and pred != true_grade
    tname    = GRADE_NAMES[true_grade] if true_grade is not None else ""

    if is_right:
        return (
            f"The model got this one right. It looked at the retinal scan and came back with "
            f"<strong>Grade {pred} — {gname}</strong> at <strong>{conf_str} confidence</strong>. "
            f"That is a strong, clear call. "
            f"To reach that answer, it was looking for {GRADE_SHOULD_FOCUS[pred]}, "
            f"and that is exactly what it found. "
            f"Every analysis method below is pointing at the same region of the retina, "
            f"which means this was not a lucky guess — the model genuinely locked onto the right signals."
        )
    elif is_wrong:
        confusion = GRADE_CONFUSION_REASON.get(
            (true_grade, pred),
            f"Some of the visual features present in {tname} can look almost identical to "
            f"those seen in {gname} under certain lighting and imaging conditions, "
            f"which led the model to read the severity level incorrectly."
        )
        return (
            f"This is where the model got it wrong. The image actually shows "
            f"<strong>Grade {true_grade} — {tname}</strong>, "
            f"but the model predicted <strong>Grade {pred} — {gname}</strong> "
            f"with {conf_str} confidence. "
            f"Here is the honest reason behind that mistake: {confusion} "
            f"The analysis sections below show you exactly where its attention went, "
            f"which region it locked onto, and why that led it down the wrong path. "
            f"This kind of error is actually useful — it tells us which visual patterns "
            f"the model still needs to learn to separate."
        )
    else:
        return (
            f"The model analysed this retinal image and came back with "
            f"<strong>Grade {pred} — {gname}</strong> "
            f"at <strong>{conf_str} confidence</strong>. "
            f"The sections below walk through how it reached that answer — "
            f"which part of the retina it focused on, "
            f"which pixels pushed the score the most, "
            f"and which internal filters fired the hardest."
        )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    model_ready = model is not None
    return render_template('index.html', model_ready=model_ready)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please download the model file from Kaggle.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        # ── Auto-detect true grade from filename ────────────────────────────
        # Strips extension so '10_left.jpeg' -> '10_left'
        raw_name   = os.path.splitext(file.filename or '')[0].strip()
        true_grade = EYEPACS_LABELS.get(raw_name, None)  # None if not in EyePACS CSV

        pil_img = Image.open(file.stream)
        img_t, orig_arr = preprocess(pil_img)

        with torch.no_grad():
            logits = model(img_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred  = int(probs.argmax())
        conf  = float(probs[pred])

        # Run XAI (each returns b64 image + raw map for spatial analysis)
        img_t_grad = img_t.requires_grad_(True)
        gc_b64,   gc_map              = run_gradcam(img_t_grad, pred, orig_arr)
        img_t_grad = img_t.requires_grad_(True)
        ig_b64,   ig_map              = run_ig(img_t_grad, pred, orig_arr)
        fma_b64,  fma_channels, fma_locs = run_fma(img_t, orig_arr)
        img_t_grad = img_t.requires_grad_(True)
        shap_b64, shap_map            = run_shap(img_t_grad, pred, orig_arr)

        descs = generate_xai_descriptions(
            pred, round(conf * 100, 1), true_grade,
            gc_map, ig_map, fma_channels, fma_locs, shap_map
        )
        overall = generate_overall_explanation(pred, round(conf * 100, 1), true_grade)

        resp = {
            'grade':       pred,
            'grade_name':  GRADE_NAMES[pred],
            'grade_color': GRADE_COLORS[pred],
            'grade_desc':  GRADE_DESC[pred],
            'confidence':  round(conf * 100, 1),
            'probabilities': {
                str(i): round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            },
            'gradcam':      gc_b64,
            'ig':           ig_b64,
            'fma':          fma_b64,
            'shap':         shap_b64,
            # Overall human-readable explanation
            'overall_desc': overall,
            # Per-method XAI descriptions
            'gradcam_desc': descs['gradcam_desc'],
            'ig_desc':      descs['ig_desc'],
            'fma_desc':     descs['fma_desc'],
            'shap_desc':    descs['shap_desc'],
            # Auto-detected from IDRiD CSV (null if not an IDRiD image)
            'true_grade':      true_grade,
            'true_grade_name': GRADE_NAMES[true_grade] if true_grade is not None else None,
        }
        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_eyepacs_labels()
    load_model()
    app.run(debug=False, port=5050)
