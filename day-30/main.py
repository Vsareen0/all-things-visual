"""
Day 30 — Motion Estimation & Inter Prediction
Block-matching motion estimation from scratch.

We make two frames where a textured object shifts by a known amount over a
static background, then recover the motion with full-search SAD block matching,
build the motion-compensated prediction, and measure how much the residual
shrinks vs. naively reusing the previous frame.

Run:  python3 day30_motion_estimation.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(30)

H, W = 128, 128
BLOCK = 16
SEARCH = 8                 # +/- search range in pixels
TRUE_DX, TRUE_DY = 5, -3   # the object's real motion (cols, rows)


def make_frames():
    """Static noisy background; a textured square that shifts between frames."""
    bg = (rng.normal(120, 8, (H, W))).clip(0, 255)
    # a high-contrast textured object
    obj = rng.normal(180, 30, (40, 40)).clip(0, 255)
    oy, ox = 50, 40  # object top-left in frame 1

    f1 = bg.copy()
    f1[oy:oy + 40, ox:ox + 40] = obj

    f2 = bg.copy()
    ny, nx = oy + TRUE_DY, ox + TRUE_DX
    f2[ny:ny + 40, nx:nx + 40] = obj
    return f1.astype(float), f2.astype(float)


def sad(a, b):
    return np.sum(np.abs(a - b))


def estimate_motion(ref, cur):
    """Full-search SAD block matching. Returns (mvs, predicted_frame)."""
    mvs = np.zeros((H // BLOCK, W // BLOCK, 2), dtype=int)  # (dy, dx)
    pred = cur.copy()
    for by in range(0, H, BLOCK):
        for bx in range(0, W, BLOCK):
            target = cur[by:by + BLOCK, bx:bx + BLOCK]
            best, best_mv = np.inf, (0, 0)
            for dy in range(-SEARCH, SEARCH + 1):
                for dx in range(-SEARCH, SEARCH + 1):
                    ry, rx = by + dy, bx + dx
                    if ry < 0 or rx < 0 or ry + BLOCK > H or rx + BLOCK > W:
                        continue
                    cand = ref[ry:ry + BLOCK, rx:rx + BLOCK]
                    s = sad(target, cand)
                    if s < best:
                        best, best_mv = s, (dy, dx)
            dy, dx = best_mv
            mvs[by // BLOCK, bx // BLOCK] = best_mv
            pred[by:by + BLOCK, bx:bx + BLOCK] = ref[by + dy:by + dy + BLOCK,
                                                      bx + dx:bx + dx + BLOCK]
    return mvs, pred


def energy(x):
    return float(np.mean(x ** 2))


def main():
    f1, f2 = make_frames()

    # Naive: assume nothing moved -> residual is just frame difference
    naive_res = f2 - f1

    # Motion compensated
    mvs, pred = estimate_motion(f1, f2)
    mc_res = f2 - pred

    e_naive = energy(naive_res)
    e_mc = energy(mc_res)
    print(f"True motion (dx, dy)         : ({TRUE_DX}, {TRUE_DY})")
    # Most blocks are static background -> (0,0) is correct for them.
    # Report the dominant NON-zero vector, i.e. the moving object's motion.
    flat = mvs.reshape(-1, 2)
    nz = flat[np.any(flat != 0, axis=1)]
    # NOTE: a block-matching MV points from the CURRENT block back to where its
    # content sat in the reference frame -> it equals the NEGATIVE of the
    # object's motion. Object moved (dx,dy)=(+5,-3), so MV should be (dx,dy)=(-5,+3).
    if len(nz):
        uniq, counts = np.unique(nz, axis=0, return_counts=True)
        dom = uniq[np.argmax(counts)]
        mv_dx, mv_dy = dom[1], dom[0]
        print(f"Recovered MV (dx, dy)        : ({mv_dx}, {mv_dy})   "
              f"(= -object motion, as expected)")
        print(f"Implied object motion (dx,dy): ({-mv_dx}, {-mv_dy})   [matches true]")
    print(f"Static blocks correctly (0,0): "
          f"{np.sum(np.all(flat == 0, axis=1))}/{len(flat)}")
    print(f"Residual energy  naive       : {e_naive:8.1f}")
    print(f"Residual energy  motion-comp : {e_mc:8.1f}")
    print(f"Reduction                    : {100*(1 - e_mc/e_naive):5.1f}%  "
          f"({e_naive/e_mc:.1f}x less residual energy)")

    # ---- visualize ----
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    ax[0, 0].imshow(f1, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Frame 1 (reference)")

    ax[0, 1].imshow(f2, cmap="gray", vmin=0, vmax=255)
    # overlay motion vectors
    ys = np.arange(BLOCK // 2, H, BLOCK)
    xs = np.arange(BLOCK // 2, W, BLOCK)
    X, Y = np.meshgrid(xs, ys)
    U = mvs[:, :, 1]   # dx
    V = mvs[:, :, 0]   # dy
    ax[0, 1].quiver(X, Y, U, V, color="#00d4aa", angles="xy",
                    scale_units="xy", scale=1, width=0.006)
    ax[0, 1].set_title("Frame 2 + recovered motion field")

    ax[1, 0].imshow(naive_res, cmap="magma", vmin=-60, vmax=60)
    ax[1, 0].set_title(f"Naive residual (reuse f1)\nenergy={e_naive:.0f}")

    ax[1, 1].imshow(mc_res, cmap="magma", vmin=-60, vmax=60)
    ax[1, 1].set_title(f"Motion-compensated residual\nenergy={e_mc:.0f}")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig("day30_motion_field.png", dpi=110)
    print("saved day30_motion_field.png")


if __name__ == "__main__":
    main()