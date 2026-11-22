import numpy as np
import cv2
import trimesh
from PIL import Image
from skimage import measure
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

FIG_H_MM = 150.0
MAX_W_MM = 78.0
BASE_H_MM = 8.0
BASE_D_MM = 60.0
HALF_MM = 14.0
ROUND_R_MM = 7.0
FRONT_MM = 3.5
BACK_MM = 2.5
STEP = 0.3

UP = '/mnt/vibefs/uploads/8f4825b9-af11-474a-a2b7-a72c1159c6ad/'
SRC = {
    'front': UP + 'yurMleIY-karate front.png',
    'back': UP + 'mhiN1DX6-karate back.png',
}


def find_base_top(mask):
    w = mask.sum(axis=1).astype(np.float64)
    kernel = np.ones(5) / 5.0
    ws = np.convolve(w, kernel, mode='same')
    h = mask.shape[0]
    bottom = np.where(w > 0)[0].max()
    zone_start = max(6, bottom - int(h * 0.15))
    wmax_zone = ws[zone_start:bottom + 1].max()
    for r in range(zone_start, bottom + 1):
        if ws[r] > 1.4 * np.median(ws[r - 6:r]) and ws[r] > 0.5 * wmax_zone:
            return r
    return zone_start


views = {}
for tag in ['front', 'back']:
    img = np.array(Image.open(SRC[tag]).convert('RGB'))
    alpha = np.array(Image.open(f'/tmp/karate_{tag}_cut.png'))[:, :, 3] > 100
    alpha = binary_fill_holes(alpha)
    lab, n = label(alpha)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        alpha = lab == sizes.argmax()
    bt = find_base_top(alpha)
    fig = alpha[:bt, :]
    ys, xs = np.where(fig)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    depth = np.load(f'/tmp/karate_{tag}_depth.npy')
    views[tag] = dict(img=img[y0:y1 + 1, x0:x1 + 1].astype(np.float32),
                      mask=fig[y0:y1 + 1, x0:x1 + 1],
                      depth=depth[y0:y1 + 1, x0:x1 + 1].astype(np.float32))
    views[tag]['hpx'] = y1 - y0 + 1
    views[tag]['wpx'] = x1 - x0 + 1

s_front = FIG_H_MM / views['front']['hpx']
mm_per_px = MAX_W_MM / views['front']['wpx']
FIG_H_MM = views['front']['hpx'] * mm_per_px
print('width-capped: figure %.1f x %.1f mm' % (MAX_W_MM, FIG_H_MM))

grid_h = int(round(FIG_H_MM / STEP)) + 2
canvas_w = int(round(MAX_W_MM / STEP)) + 2


def to_canvas(tag, flip):
    v = views[tag]
    s_own = FIG_H_MM / v['hpx']
    out_h = grid_h
    out_w = max(int(round(v['wpx'] * s_own / STEP)), 1)
    m = v['mask']
    d = v['depth']
    im = v['img']
    if flip:
        m = m[:, ::-1]
        d = d[:, ::-1]
        im = im[:, ::-1]
    m_r = cv2.resize((m * 255).astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_AREA) > 127
    m_r = binary_fill_holes(m_r)
    d_r = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_AREA)
    i_r = cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_AREA)
    rows_bot = m_r[int(0.67 * out_h):]
    cols_bot = np.where(rows_bot.any(axis=0))[0]
    cx = (cols_bot[0] + cols_bot[-1]) / 2.0
    off = int(round(canvas_w / 2 - cx))
    m_c = np.zeros((grid_h, canvas_w), bool)
    d_c = np.zeros((grid_h, canvas_w), np.float32)
    i_c = np.zeros((grid_h, canvas_w, 3), np.float32)
    x0 = max(0, off)
    x1 = min(canvas_w, off + out_w)
    sx0 = x0 - off
    sx1 = sx0 + (x1 - x0)
    m_c[:, x0:x1] = m_r[:, sx0:sx1]
    d_c[:, x0:x1] = d_r[:, sx0:sx1]
    i_c[:, x0:x1] = i_r[:, sx0:sx1]
    return dict(mask=m_c, depth=d_c, img=i_c)


iou_best = None
for flip in [False, True]:
    b = to_canvas('back', flip)
    f = to_canvas('front', False)
    inter = (b['mask'] & f['mask']).sum()
    union = (b['mask'] | f['mask']).sum()
    iou = inter / max(union, 1)
    print('flip=%s IoU %.3f' % (flip, iou))
    if iou_best is None or iou > iou_best[0]:
        iou_best = (iou, flip, f, b)

iou, do_flip, front_p, back_p = iou_best
print('back flip chosen:', do_flip, 'IoU %.3f' % iou)

M = front_p['mask'] | back_p['mask']
M = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_CLOSE,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
M = binary_fill_holes(M)
lab, n = label(M)
if n > 1:
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    M = lab == sizes.argmax()
M[0, :] = False
M[-1, :] = False
M[:, 0] = False
M[:, -1, ] = False
M = binary_fill_holes(M)
print('union px', M.sum(), 'IoU final %.3f' % ((front_p['mask'] & back_p['mask']).sum() / M.sum()))

dist = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5) * STEP
s = np.clip(dist / ROUND_R_MM, 0, 1)
s = s * s * (3 - 2 * s)
half = HALF_MM * s
print('half depth max %.1f mm' % half.max())


def relief_field(p, amp):
    m = p['mask'] & M
    m_soft = cv2.GaussianBlur((m & M).astype(np.float32), (0, 0), 1.5)
    m_soft = np.clip(m_soft * 1.6, 0, 1)
    depth_c = p['depth']
    vals = depth_c[m]
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    d_n = np.clip((depth_c - lo) / max(hi - lo, 1e-6), 0, 1)
    d_blur = cv2.GaussianBlur(d_n.astype(np.float32), (0, 0), 3.0)
    lum = p['img'].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
    lo2, hi2 = np.percentile(lum[m], 3), np.percentile(lum[m], 97)
    l_n = np.clip((lum - lo2) / max(hi2 - lo2, 1e-6), 0, 1)
    l_detail = l_n - cv2.GaussianBlur(l_n, (0, 0), 7.0)
    l_detail = np.clip(l_detail / max(np.abs(l_detail[m]).max(), 1e-6), -1, 1)
    l_detail = cv2.GaussianBlur(l_detail, (0, 0), 1.3)
    sh = d_blur * 0.55 + 0.30 * cv2.GaussianBlur(l_n, (0, 0), 2.0) - 0.25 * l_detail
    s_lo, s_hi = np.percentile(sh[m], 2), np.percentile(sh[m], 98)
    sh = np.clip((sh - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1)
    sh = (sh - 0.5) * 2.0
    return sh * m_soft * amp


front_mm = relief_field(front_p, FRONT_MM)
back_mm = relief_field(back_p, BACK_MM)
print('relief front max %.2f  back %.2f' % (front_mm.max(), back_mm.max()))

nz = int(np.ceil((2 * HALF_MM + FRONT_MM + BACK_MM + 3) / STEP)) + 2
z_vals = (np.arange(nz) - (nz * STEP) / 2.0) * STEP
lo_z = (-half - back_mm)[..., None] - 0.8
hi_z = (half + front_mm)[..., None] + 0.8
solid = M[..., None] & (z_vals[None, None, :] < hi_z) & (z_vals[None, None, :] > lo_z)
vol = np.transpose(solid, (2, 0, 1))
vol[0] = False
vol[-1] = False
vol[:, 0, :] = False
vol[:, -1, :] = False
vol[:, :, 0] = False
vol[:, :, -1] = False

field = gaussian_filter(vol.astype(np.float32), 1.0)
verts, faces, normals, _ = measure.marching_cubes(field, level=0.5)
verts = verts * STEP
verts[:, 1] = (grid_h * STEP) - verts[:, 1]
verts = verts[:, [2, 1, 0]]
verts[:, 1] -= verts[:, 1].min()
verts[:, 2] -= verts[:, 2].min()

figure = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
figure.update_faces(figure.nondegenerate_faces())
figure.fix_normals()
trimesh.smoothing.filter_laplacian(figure, lamb=0.35, iterations=8)
figure.fix_normals()
print('figure watertight', figure.is_watertight, 'faces', len(figure.faces), 'bounds', np.round(figure.bounds, 1))

cz = (M.astype(np.float32) * np.arange(canvas_w)[None, :]).sum() / max(M.sum(), 1) * STEP - 0.0
base = trimesh.creation.cylinder(radius=BASE_D_MM / 2, height=BASE_H_MM, sections=96)
base.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
base.apply_translation([cz, BASE_H_MM / 2 - 0.5, figure.bounds[:, 0].min() + figure.extents[2] / 2])

union = trimesh.boolean.union([figure, base], engine='manifold')
if not union.is_watertight:
    print('union not watertight -> concatenate fallback')
    union = figure + base

union.apply_translation([-union.bounds[0][0], -union.bounds[0][1], -union.bounds[0][2]])
union.export('/workspace/karate_figure_v2.stl')
print('EXPORT watertight:', union.is_watertight)
print('EXPORT faces', len(union.faces), 'bounds', np.round(union.bounds, 1))
print('volume cm3 %.1f' % (union.volume / 1000))
Image.fromarray(((half + front_mm).clip(0, None) / (HALF_MM + FRONT_MM) * 255).astype('uint8')).save(
    '/workspace/preview/karate_v2_front_field.png')
dec = union.simplify_quadric_decimation(0.35)
trimesh.smoothing.filter_laplacian(dec, lamb=0.3, iterations=5)
dec.export('/workspace/preview/karate_v2_preview.ply')
print('done')
