import typing
import sys
sys.modules["typing.io"] = typing

import torch
import gradio as gr
from PIL import Image, ImageFilter
import sympy
import re
import numpy as np
import plotly.graph_objects as go

from latex2sympy2 import latex2sympy
from src.model import SwinMathModel, get_masks
from src.tokenizer import MathTokenizer
from src.dataset import ResizeAndPadSquare


torch.set_default_dtype(torch.float32)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = MathTokenizer("vocab.json")
TRANSFORM = ResizeAndPadSquare(target_size=256)

MODEL = SwinMathModel(vocab_size=TOKENIZER.vocab_size).to(DEVICE)
MODEL.load_state_dict(torch.load("checkpoints/swin_math_epoch_11.pth", map_location=DEVICE))
MODEL.eval()


def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    arr  = np.array(gray)

    mask = arr < 240
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size and cols.size:
        pad = 12
        r0, r1 = max(0, rows[0] - pad), min(arr.shape[0], rows[-1] + pad)
        c0, c1 = max(0, cols[0] - pad), min(arr.shape[1], cols[-1] + pad)
        gray = gray.crop((c0, r0, c1, r1))
        arr  = np.array(gray)

    gray = gray.filter(ImageFilter.SHARPEN)
    arr  = np.array(gray)

    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 > p2:
        arr = np.clip((arr.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    arr = np.where(arr < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def clean_latex(raw: str) -> str:
    s = re.sub(r'\s+', ' ', raw)
    s = re.sub(r'([a-zA-Z0-9])\^([0-9]+)', r'\1^{\2}', s)
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\mathrm\{([a-zA-Z]+)\}', r'\1', s)
    return s.strip()


def format_solution(sol):
    exact = sympy.latex(sol)
    try:
        if sol.is_real:
            return f"{exact} \\; (\\approx {float(sol.evalf()):.4f})"
    except Exception:
        pass
    return exact


def solve_expression(expr, var):
    core = expr.lhs - expr.rhs if isinstance(expr, sympy.Equality) else expr
    try:
        sols = sympy.solve(core, var)
        if sols:
            return sols
    except Exception:
        pass
    try:
        r = sympy.roots(core, var)
        if r:
            return list(r.keys())
    except Exception:
        pass
    try:
        return [sympy.sympify(r) for r in sympy.nroots(sympy.Poly(core, var))]
    except Exception:
        pass
    return []


def _safe_scalar(f, x):
    try:
        v = float(f(x))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def safe_eval(f, x_vals):
    with np.errstate(all="ignore"):
        try:
            y_vals = f(x_vals).astype(float)
        except Exception:
            y_vals = np.array([_safe_scalar(f, x) for x in x_vals], dtype=float)

    y_vals[~np.isfinite(y_vals)] = np.nan

    dy       = np.abs(np.diff(np.where(np.isnan(y_vals), 0, y_vals)))
    finite_dy = dy[np.isfinite(dy) & (dy > 0)]
    if len(finite_dy):
        threshold = np.percentile(finite_dy, 99) * 5
        for j in np.where(dy > threshold)[0]:
            y_vals[j] = np.nan
            if j + 1 < len(y_vals):
                y_vals[j + 1] = np.nan

    return y_vals


def plot_explicit(expr, var, solutions=None):
    try:
        f      = sympy.lambdify(var, expr, modules=["numpy"])
        x_vals = np.linspace(-20, 20, 4000)
        y_vals = safe_eval(f, x_vals)

        finite = y_vals[np.isfinite(y_vals)]
        if len(finite):
            q1, q99 = np.nanpercentile(finite, 1), np.nanpercentile(finite, 99)
            pad     = max((q99 - q1) * 0.3, 1)
            y_range = [q1 - pad, q99 + pad]
        else:
            y_range = [-10, 10]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines", name="f(x)",
            line=dict(width=2, color="#00b4d8"), connectgaps=False,
        ))
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1))

        if solutions:
            roots = [float(s.evalf()) for s in solutions
                     if np.isfinite(float(s.evalf() if s.is_real else complex(s.evalf()).real))]
            roots = [r for r in roots if np.isfinite(r)]
            if roots:
                fig.add_trace(go.Scatter(
                    x=roots, y=[0] * len(roots), mode="markers", name="Roots",
                    marker=dict(size=10, color="#ff6b6b"),
                ))

        fig.update_layout(
            title=dict(text=f"f(x) = ${sympy.latex(expr)}$", font=dict(size=14)),
            template="plotly_dark", height=450,
            margin=dict(l=40, r=20, t=50, b=40),
            hovermode="x unified",
            xaxis=dict(range=[-10, 10], rangeslider=dict(visible=True, thickness=0.04),
                       showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(range=y_range, showgrid=True,
                       gridcolor="rgba(255,255,255,0.1)", fixedrange=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig
    except Exception:
        return None


def plot_implicit(expr, x_sym, y_sym):
    try:
        core = expr.lhs - expr.rhs if isinstance(expr, sympy.Equality) else expr
        f    = sympy.lambdify((x_sym, y_sym), core, modules=["numpy"])

        x_vals = np.linspace(-10, 10, 600)
        y_vals = np.linspace(-10, 10, 600)
        X, Y   = np.meshgrid(x_vals, y_vals)

        with np.errstate(all="ignore"):
            Z = f(X, Y).astype(float)
            Z = np.where(np.isfinite(Z), Z, np.nan)

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=x_vals, y=y_vals, z=Z,
            contours=dict(start=0, end=0, size=1, coloring="lines"),
            line=dict(width=2, color="#00b4d8"),
            showscale=False,
        ))
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig.update_layout(
            title=dict(text=f"Implicit: ${sympy.latex(expr)}$", font=dict(size=14)),
            template="plotly_dark", height=450,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title=str(x_sym)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                       title=str(y_sym), scaleanchor="x", scaleratio=1),
        )
        return fig
    except Exception:
        return None


def predict(image):
    if image is None:
        return "", "", "", None

    if isinstance(image, dict):
        image = image.get("composite") or image.get("layers", [None])[0]
    if image is None:
        return "", "", "", None

    image      = preprocess_image(image)
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    sos_idx = TOKENIZER.token_to_id[TOKENIZER.SOS]
    eos_idx = TOKENIZER.token_to_id[TOKENIZER.EOS]
    pad_idx = TOKENIZER.token_to_id[TOKENIZER.PAD]

    target_indices = [sos_idx]
    with torch.no_grad():
        for _ in range(150):
            decoder_input          = torch.tensor([target_indices], dtype=torch.long).to(DEVICE)
            tgt_mask, tgt_pad_mask = get_masks(decoder_input, pad_idx, DEVICE)
            logits                 = MODEL(img_tensor, decoder_input, tgt_mask, tgt_pad_mask)
            next_token             = logits[0, -1, :].argmax().item()
            target_indices.append(next_token)
            if next_token == eos_idx:
                break

    tokens    = [TOKENIZER.id_to_token.get(i, "") for i in target_indices]
    tokens    = [t for t in tokens if t not in (TOKENIZER.SOS, TOKENIZER.EOS, TOKENIZER.PAD)]
    latex_str = " ".join(tokens)
    latex_str = re.sub(r'(?<=\d)\s+(?=\d)', '', latex_str)
    latex_str = re.sub(r'(?<=\d)\s+(?=[a-zA-Z])', '', latex_str)

    plot = None
    try:
        normalized = clean_latex(latex_str)

        if "=" in normalized:
            lhs_str, rhs_str = normalized.split("=", 1)
            lhs_expr = latex2sympy(lhs_str)
            expr     = lhs_expr if not rhs_str.strip() else sympy.Eq(lhs_expr, latex2sympy(rhs_str))
        else:
            expr = latex2sympy(normalized)

        free_vars = list(expr.free_symbols)

        if len(free_vars) == 2:
            x_sym, y_sym = sorted(free_vars, key=lambda s: str(s))
            plot         = plot_implicit(expr, x_sym, y_sym)
            visual       = f"$$\n{normalized}\n$$"
            status       = "Plotted (implicit)"

        elif len(free_vars) == 1:
            var       = free_vars[0]
            solutions = solve_expression(expr, var)
            plot_expr = expr.lhs - expr.rhs if isinstance(expr, sympy.Equality) else expr
            plot      = plot_explicit(plot_expr, var, solutions or None)

            if not solutions:
                visual = f"$$\n{normalized}\n$$"
                status = "Plotted (no closed-form roots)"
            else:
                sol_str = ", \\quad ".join(
                    f"{sympy.latex(var)} = {format_solution(s)}" for s in solutions
                )
                arrow   = "\\implies"
                visual  = f"$$\n{normalized} {arrow} {sol_str}\n$$"
                status  = "Solved"

        else:
            simplified = sympy.simplify(expr)
            try:
                val    = complex(simplified.evalf())
                result = f"{val.real:.6g}" if abs(val.imag) < 1e-10 else sympy.latex(simplified)
                visual = f"$$\n{normalized} = {sympy.latex(simplified)} \\approx {result}\n$$"
                status = "Evaluated"
            except Exception:
                visual = f"$$\n{sympy.latex(simplified)}\n$$"
                status = "Simplified"

    except Exception as e:
        visual = f"$$\n{latex_str}\n$$"
        status = f"CAS Error: {str(e)[:80]}"
        print(f"CAS error: {e}")

    return latex_str, status, visual, plot


css = """
html, body, .gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 8px !important;
}
#sketchpad,
#sketchpad > .wrap,
#sketchpad > div,
#sketchpad .sketchpad-container,
#sketchpad .image-container,
#sketchpad .svelte-1yrgrx2,
#sketchpad [data-testid="sketchpad"] {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    height: 400px !important;
    min-height: 400px !important;
    max-height: 400px !important;
    overflow: visible !important;
}
#sketchpad canvas {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    height: 400px !important;
    min-height: 400px !important;
    display: block !important;
    object-fit: fill !important;
}
#sketchpad .scroll-hide,
#sketchpad .overflow-hidden,
#sketchpad .overflow-auto,
#sketchpad .overflow-scroll {
    overflow: visible !important;
    height: 400px !important;
}
"""

with gr.Blocks(css=css, title="SwinMath: Vision Equation Solver") as interface:
    gr.Markdown("# SwinMath: Vision Equation Solver")

    with gr.Tabs(elem_id="input_tabs"):
        with gr.Tab("Draw"):
            sketchpad = gr.Sketchpad(
                type="pil", label="Draw Equation",
                canvas_size=(2400, 400), elem_id="sketchpad",
            )
            with gr.Row():
                clear_btn      = gr.Button("Clear", variant="secondary")
                draw_submit    = gr.Button("Submit", variant="primary")

        with gr.Tab("Upload"):
            upload        = gr.Image(type="pil", label="Upload a photo of your equation")
            upload_submit = gr.Button("Submit", variant="primary")

    with gr.Row():
        latex_out  = gr.Textbox(label="Raw LaTeX Output")
        status_out = gr.Textbox(label="Calculator Status")

    render_out = gr.Markdown(label="Rendered Equation & Solution")
    plot_out   = gr.Plot(label="Interactive Graph")

    draw_submit.click(predict,   inputs=[sketchpad], outputs=[latex_out, status_out, render_out, plot_out])
    upload_submit.click(predict, inputs=[upload],    outputs=[latex_out, status_out, render_out, plot_out])
    clear_btn.click(fn=lambda: None, inputs=[], outputs=[sketchpad])


if __name__ == "__main__":
    interface.launch()