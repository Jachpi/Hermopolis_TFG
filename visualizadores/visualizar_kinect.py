import math
import json
import sys
import os
import http.server
import threading
import webbrowser
import argparse

KEEP = [0,3,4,5,6,8,9,10,12,13,14,15,16,17,18,19,21,22,23,24]

CONNECTIONS = [
    [3,0],[4,5],[5,6],[21,6],[22,6],
    [8,9],[9,10],[23,10],[24,10],
    [4,8],
    [12,0],[16,0],
    [12,13],[13,14],[14,15],
    [16,17],[17,18],[18,19]
]

def parse_kinect(filepath):
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    first_ts = int(lines[0].strip().split()[0])
    for f_it, line in enumerate(lines):
        data = line.strip().split()
        if not data:
            continue
        timestamp = int(data[0]) - first_ts
        nodes = {}
        it = 1
        while it <= 73:
            nid = int(it / 3)
            nodes[nid] = {
                'x': float(data[it + 1]),
                'y': float(data[it + 2]),
                'z': float(data[it])
            }
            it += 3
        frames.append({'frame_index': f_it, 'timestamp_ms': timestamp, 'nodes': nodes})
    return frames

def normalize(frames):
    k0, k3 = 0, 3
    sb = frames[0]['nodes'][k0]
    hd = frames[0]['nodes'][k3]
    dist = math.sqrt((sb['x']-hd['x'])**2 + (sb['y']-hd['y'])**2 + (sb['z']-hd['z'])**2)
    for frame in frames:
        s = frame['nodes'][k0]
        sx, sy, sz = s['x'], s['y'], s['z']
        for node in frame['nodes'].values():
            node['x'] = (node['x'] - sx) / dist
            node['y'] = (node['y'] - sy) / dist
            node['z'] = (node['z'] - sz) / dist
    return frames

def to_compact(frames):
    compact = []
    for frame in frames:
        nodes = {}
        for nid in KEEP:
            n = frame['nodes'][nid]
            nodes[str(nid)] = [round(n['x'], 3), round(n['y'], 3), round(n['z'], 3)]
        compact.append(nodes)
    return compact

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Kinect 3D - {filename}</title>
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
#frame-num {{ font-size: 13px; font-weight: 500; min-width: 70px; }}
button {{
  background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
  color: #eee; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
}}
button:hover {{ background: rgba(255,255,255,0.2); }}
#info {{ position: absolute; top: 12px; left: 12px; font-size: 12px; color: #888; }}
#label {{ position: absolute; top: 12px; right: 12px; font-size: 13px; color: #4af; }}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="info">Arrastra para rotar · Scroll para zoom</div>
<div id="label">{filename}</div>
<div id="controls">
  <button id="play-btn">▶ Play</button>
  <label>Frame:</label>
  <input type="range" id="frame-slider" min="0" max="{max_frame}" value="0" step="1">
  <span id="frame-num">0 / {max_frame}</span>
</div>
<script>
const FRAMES = {frames_json};
const KEEP = {keep_json};
const CONNECTIONS = {conn_json};

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
canvas.addEventListener('wheel', e => {{ zoom *= e.deltaY > 0 ? 0.95 : 1.05; zoom = Math.max(0.5, Math.min(5, zoom)); draw(); }});

function project(x, y, z) {{
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  let y2 = y * cx - z * sx, z2 = y * sx + z * cx;
  let x3 = x * cy + z2 * sy, z3 = -x * sy + z2 * cy;
  const scale = zoom * Math.min(W, H) * 0.45;
  const fov = 2.5, d = fov + z3;
  return {{ px: W/2 + x3/d*scale, py: H/2 + y2/d*scale }};
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const frame = FRAMES[currentFrame];
  const pts = {{}};
  for (const id of KEEP) {{
    const [x, y, z] = frame[String(id)];
    pts[id] = project(x, y, z);
  }}
  ctx.strokeStyle = 'rgba(100,180,255,0.6)';
  ctx.lineWidth = 1.5;
  for (const [a, b] of CONNECTIONS) {{
    if (!pts[a] || !pts[b]) continue;
    ctx.beginPath();
    ctx.moveTo(pts[a].px, pts[a].py);
    ctx.lineTo(pts[b].px, pts[b].py);
    ctx.stroke();
  }}
  for (const id of KEEP) {{
    const p = pts[id];
    const r = (id === 0 || id === 3) ? 5 : 3.5;
    const col = id === 0 ? '#ff9944' : id === 3 ? '#ffdd44' : '#4af';
    ctx.beginPath();
    ctx.arc(p.px, p.py, r, 0, Math.PI*2);
    ctx.fillStyle = col;
    ctx.fill();
  }}
}}

const slider = document.getElementById('frame-slider');
const frameNum = document.getElementById('frame-num');
const playBtn = document.getElementById('play-btn');

slider.addEventListener('input', () => {{
  currentFrame = parseInt(slider.value);
  frameNum.textContent = currentFrame + ' / ' + (FRAMES.length - 1);
  draw();
}});

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? '⏸ Pausa' : '▶ Play';
  if (playing) animate();
  else clearTimeout(animId);
}});

function animate() {{
  currentFrame = (currentFrame + 1) % FRAMES.length;
  slider.value = currentFrame;
  frameNum.textContent = currentFrame + ' / ' + (FRAMES.length - 1);
  draw();
  if (playing) animId = setTimeout(() => requestAnimationFrame(animate), 50);
}}

draw();
</script>
</body>
</html>"""

def generate_html(filepath):
    frames = parse_kinect(filepath)
    frames = normalize(frames)
    compact = to_compact(frames)
    filename = os.path.basename(filepath)
    html = HTML_TEMPLATE.format(
        filename=filename,
        max_frame=len(compact) - 1,
        frames_json=json.dumps(compact),
        keep_json=json.dumps(KEEP),
        conn_json=json.dumps(CONNECTIONS)
    )
    return html

def main():
    parser = argparse.ArgumentParser(description='Visualizador 3D de archivos Kinect .txt')
    parser.add_argument('input', help='Ruta al archivo .txt de Kinect')
    parser.add_argument('--output', help='Guardar HTML en esta ruta (opcional)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: no se encontró el archivo {args.input}")
        sys.exit(1)

    html = generate_html(args.input)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(html)
        print(f"HTML guardado en: {args.output}")
    else:
        # Abrir en navegador directamente
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            tmp_path = f.name
        print(f"Abriendo visualizador en el navegador...")
        webbrowser.open(f'file://{tmp_path}')

if __name__ == '__main__':
    main()