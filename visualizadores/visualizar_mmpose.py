import json
import math
import os
import sys
import argparse
import webbrowser
import tempfile

KEEP = [0,3,4,5,6,8,9,10,12,13,14,15,16,17,18,19,21,22,23,24]

CONNECTIONS = [
    [3,0],[4,5],[5,6],[4,8],[8,9],[9,10],
    [12,0],[16,0],
    [12,13],[13,14],[14,15],
    [16,17],[17,18],[18,19],
    [21,6],[22,6],[23,10],[24,10]
]

# Nodos que son aproximaciones (no tienen datos reales en human3d)
APPROX_NODES = {15, 19, 21, 22, 23, 24}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualizador 3D de nodos MMPose')
    parser.add_argument('input', help='Ruta al archivo .json de MMPose')
    parser.add_argument('--output', help='Guardar HTML en esta ruta (opcional)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='No normalizar los datos (si ya vienen normalizados)')
    return parser.parse_args()


def normalize(data):
    '''Normaliza los datos centrandolos en SPINEBASE y escalando por
    la distancia SPINEBASE->HEAD del primer frame, igual que normalizer.py.'''

    k0 = type(next(iter(data[0]['nodes'])))(0)
    k3 = type(next(iter(data[0]['nodes'])))(3)

    sb = data[0]['nodes'][k0]
    hd = data[0]['nodes'][k3]
    dist = math.sqrt(
        (sb['x']-hd['x'])**2 +
        (sb['y']-hd['y'])**2 +
        (sb['z']-hd['z'])**2
    )
    print(f"Factor de escala (distancia spinebase->head): {dist:.4f}")

    for frame in data:
        s = frame['nodes'][k0]
        sx, sy, sz = s['x'], s['y'], s['z']
        for node in frame['nodes'].values():
            node['x'] = (node['x'] - sx) / dist
            node['y'] = (node['y'] - sy) / dist
            node['z'] = (node['z'] - sz) / dist

    return data


def to_compact(data):
    '''Convierte los datos al formato compacto [x, y, z] por nodo para el HTML.
    Devuelve (frames, timestamps_ms).'''
    compact = []
    timestamps = []
    for frame in data:
        nodes = {}
        for nid in KEEP:
            key = type(next(iter(frame['nodes'])))(nid)
            n = frame['nodes'].get(key)
            if n:
                nodes[str(nid)] = [round(n['x'], 4), round(n['y'], 4), round(n['z'], 4)]
            else:
                nodes[str(nid)] = [0.0, 0.0, 0.0]
        compact.append(nodes)
        timestamps.append(frame.get('timestamp_ms', 0))
    return compact, timestamps


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>MMPose 3D - {filename}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #eee; font-family: sans-serif; overflow: hidden; }}
#canvas {{ display: block; }}
#controls {{
  position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
  display: flex; align-items: center; gap: 12px; background: rgba(0,0,0,0.6);
  padding: 10px 16px; border-radius: 8px;
}}
#controls label {{ font-size: 13px; color: #aaa; }}
#frame-slider {{ width: 260px; }}
#frame-num {{ font-size: 13px; font-weight: 500; min-width: 80px; }}
button {{
  background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
  color: #eee; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
}}
button:hover {{ background: rgba(255,255,255,0.2); }}
#info {{ position: absolute; top: 12px; left: 12px; font-size: 12px; color: #888; }}
#label {{ position: absolute; top: 12px; right: 12px; font-size: 13px; color: #4af; }}
#legend {{
  position: absolute; bottom: 70px; left: 12px;
  font-size: 11px; color: #aaa; line-height: 1.8;
}}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="info">Arrastra para rotar · Scroll para zoom</div>
<div id="label">{filename} · {n_frames} frames</div>
<div id="legend">
  <span style="color:#4af">&#9632;</span> Nodos reales (H36M)<br>
  <span style="color:#ff6600">&#9632;</span> Nodos aproximados (pies/manos)<br>
  <span style="color:#ff9944">&#9632;</span> SPINEBASE &nbsp;
  <span style="color:#ffdd44">&#9632;</span> HEAD
</div>
<div id="controls">
  <button id="play-btn">&#9654; Play</button>
  <label>Tiempo:</label>
  <input type="range" id="frame-slider" min="0" max="{max_frame}" value="0" step="1">
  <span id="frame-num">0.000s / {duration}s</span>
</div>
<script>
const FRAMES = {frames_json};
const TIMESTAMPS = {timestamps_json};
const KEEP = {keep_json};
const CONNECTIONS = {conn_json};
const APPROX_NODES = new Set({approx_json});

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let W, H;
function resize() {{ W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }}
resize();
window.addEventListener('resize', () => {{ resize(); draw(); }});

let rotX = 0.3, rotY = 0.2, zoom = 1.4;
let isDragging = false, lastMX, lastMY;
let currentFrame = 0, playing = false, animId = null;

canvas.addEventListener('mousedown', e => {{ isDragging = true; lastMX = e.clientX; lastMY = e.clientY; }});
canvas.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  rotY += (e.clientX - lastMX) * 0.01;
  rotX += (e.clientY - lastMY) * 0.01;
  lastMX = e.clientX; lastMY = e.clientY;
  draw();
}});
canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => isDragging = false);
canvas.addEventListener('wheel', e => {{
  zoom *= e.deltaY > 0 ? 0.95 : 1.05;
  zoom = Math.max(0.3, Math.min(6, zoom));
  draw();
}});

function project(x, y, z) {{
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  let y2 = y * cx - z * sx, z2 = y * sx + z * cx;
  let x3 = x * cy + z2 * sy, z3 = -x * sy + z2 * cy;
  const scale = zoom * Math.min(W, H) * 0.45;
  const fov = 2.5, d = fov + z3;
  return {{ px: W/2 + x3/d*scale, py: H/2 - y2/d*scale }};
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const frame = FRAMES[currentFrame];
  const pts = {{}};
  for (const id of KEEP) {{
    const v = frame[String(id)];
    if (v) pts[id] = project(v[0], v[1], v[2]);
  }}

  // Aristas
  for (const [a, b] of CONNECTIONS) {{
    if (!pts[a] || !pts[b]) continue;
    const isApprox = APPROX_NODES.has(a) || APPROX_NODES.has(b);
    ctx.strokeStyle = isApprox ? 'rgba(255,130,0,0.45)' : 'rgba(100,180,255,0.65)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[a].px, pts[a].py);
    ctx.lineTo(pts[b].px, pts[b].py);
    ctx.stroke();
  }}

  // Nodos
  for (const id of KEEP) {{
    const p = pts[id];
    if (!p) continue;
    const isApprox = APPROX_NODES.has(id);
    const r = (id === 0 || id === 3) ? 5 : 3.5;
    const col = id === 0 ? '#ff9944'
               : id === 3 ? '#ffdd44'
               : isApprox ? '#ff6600'
               : '#4af';
    ctx.beginPath();
    ctx.arc(p.px, p.py, r, 0, Math.PI*2);
    ctx.fillStyle = col;
    ctx.fill();
  }}
}}

const slider = document.getElementById('frame-slider');
const frameNum = document.getElementById('frame-num');
const playBtn = document.getElementById('play-btn');

const totalDuration = TIMESTAMPS[TIMESTAMPS.length - 1] / 1000;

function formatTime(ms) {{
  return (ms / 1000).toFixed(3) + 's';
}}

slider.addEventListener('input', () => {{
  currentFrame = parseInt(slider.value);
  frameNum.textContent = formatTime(TIMESTAMPS[currentFrame]) + ' / ' + totalDuration.toFixed(3) + 's';
  draw();
}});

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? '&#9646;&#9646; Pausa' : '&#9654; Play';
  if (playing) animate();
  else clearTimeout(animId);
}});

function animate() {{
  const next = (currentFrame + 1) % FRAMES.length;
  const delay = next > 0
    ? TIMESTAMPS[next] - TIMESTAMPS[next - 1]
    : TIMESTAMPS[1] - TIMESTAMPS[0];
  currentFrame = next;
  slider.value = currentFrame;
  frameNum.textContent = formatTime(TIMESTAMPS[currentFrame]) + ' / ' + totalDuration.toFixed(3) + 's';
  draw();
  if (playing) animId = setTimeout(() => requestAnimationFrame(animate), Math.max(1, delay));
}}

draw();
</script>
</body>
</html>"""


def generate_html(filepath, do_normalize=True):
    with open(filepath, 'r') as f:
        data = json.load(f)

    if do_normalize:
        data = normalize(data)

    compact, timestamps = to_compact(data)
    filename = os.path.basename(filepath)
    duration = f"{timestamps[-1] / 1000:.3f}" if timestamps else "0.000"

    html = HTML_TEMPLATE.format(
        filename=filename,
        n_frames=len(compact),
        max_frame=len(compact) - 1,
        duration=duration,
        frames_json=json.dumps(compact),
        timestamps_json=json.dumps(timestamps),
        keep_json=json.dumps(KEEP),
        conn_json=json.dumps(CONNECTIONS),
        approx_json=json.dumps(list(APPROX_NODES))
    )
    return html


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: no se encontro el archivo {args.input}")
        sys.exit(1)

    html = generate_html(args.input, do_normalize=not args.no_normalize)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(html)
        print(f"HTML guardado en: {args.output}")
    else:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            tmp_path = f.name
        print(f"Abriendo visualizador en el navegador...")
        webbrowser.open(f'file://{tmp_path}')


if __name__ == '__main__':
    main()