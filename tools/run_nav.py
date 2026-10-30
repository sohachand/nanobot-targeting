import json, math, glob, statistics as st

def sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
def add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def mul(a,s): return [a[0]*s, a[1]*s, a[2]*s]
def norm(a): return math.sqrt(a[0]**2+a[1]**2+a[2]**2)

def step(pos, target, speed):
    d = sub(target, pos); dist = norm(d)
    nxt = list(target) if dist < speed else add(pos, mul(d, speed/dist))
    return nxt, dist

def simulate_static(start, target, speed, max_steps=200):
    pos = list(start); traj=[pos]
    for _ in range(max_steps):
        nxt, dist = step(pos, target, speed)
        pos = nxt; traj.append(pos)
        if dist < 1.0: break
    return traj

def simulate_dynamic(start, waypoints, speed, arrive=1.0, max_per=100):
    pos = list(start); traj=[pos]
    for wp in waypoints:
        for _ in range(max_per):
            nxt, dist = step(pos, wp, speed)
            pos = nxt; traj.append(pos)
            if dist < arrive: break
    return traj

def path_length(t):
    return sum(norm(sub(t[i], t[i-1])) for i in range(1, len(t)))

def final_dist(t, pts):
    f = t[-1]
    return min(norm(sub(p, f)) for p in pts)

speed = 1.5
rows = []
for fp in sorted(glob.glob('sim_data_*.json')):
    d = json.load(open(fp))
    pts = [[c[0],c[1],c[2]] for c in sorted(d['centroids'], key=lambda c: c[0])]
    start = [0, 128, 128]
    target = [sum(p[i] for p in pts)/len(pts) for i in range(3)]
    t_static = simulate_static(start, target, speed)
    t_dynamic = simulate_dynamic(start, pts, speed)
    rows.append({
        'patient': d['patient'],
        'slices': len(pts),
        'static_len': path_length(t_static),
        'dynamic_len': path_length(t_dynamic),
        'static_final': final_dist(t_static, pts),
    })

rows.sort(key=lambda r: r['dynamic_len'])
print(f"{'Patient':28s} {'Slices':>6s} {'Static':>8s} {'Dynamic':>8s} {'StatFinal':>10s}")
for r in rows:
    print(f"{r['patient']:28s} {r['slices']:6d} {r['static_len']:8.1f} {r['dynamic_len']:8.1f} {r['static_final']:10.2f}")

print()
print("N patients:", len(rows))
print("Mean static len:", st.mean(r['static_len'] for r in rows))
print("Median static len:", st.median(r['static_len'] for r in rows))
print("Mean dynamic len:", st.mean(r['dynamic_len'] for r in rows))
print("Median dynamic len:", st.median(r['dynamic_len'] for r in rows))
print("Mean static final dist:", st.mean(r['static_final'] for r in rows))
print("Worst static final dist:", max(r['static_final'] for r in rows))
print("Static range:", min(r['static_len'] for r in rows), "-", max(r['static_len'] for r in rows))
print("Dynamic range:", min(r['dynamic_len'] for r in rows), "-", max(r['dynamic_len'] for r in rows))
