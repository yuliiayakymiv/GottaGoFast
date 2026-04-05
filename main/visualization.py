"""
Altitude normalisation

Converts mixed MSL/AGL altitude data to continuous AGL reference frame.
Handles ArduPilot SITL log artefacts where altitude format switches between
MSL (takeoff/landing) and AGL (cruise).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

JUMP_THRESH = 50
AGL_MAX_M   = 100


def _to_agl(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed MSL/AGL altitudes to continuous AGL. Adds 'alt_agl' column."""
    alt = df["alt"].values.astype(float).copy()
    n = len(alt)

    segs = []
    start = 0
    for i in range(1, n):
        if abs(alt[i] - alt[i - 1]) > JUMP_THRESH:
            segs.append((start, i))
            start = i
    segs.append((start, n))

    def is_agl(s, e):
        return float(np.max(alt[s:e])) < AGL_MAX_M

    print(f"[alt] {len(segs)} segment(s):")
    for s, e in segs:
        kind = "AGL" if is_agl(s, e) else "MSL"
        print(f"rows {s:3d}-{e-1:3d} {kind} range {alt[s:e].min():.1f}–{alt[s:e].max():.1f} m")

    agl = alt.copy()
    agl_segs = [(s, e) for s, e in segs if is_agl(s, e)]
    msl_segs = [(s, e) for s, e in segs if not is_agl(s, e)]

    if agl_segs and msl_segs:
        agl_ground = min(alt[s:e].min() for s, e in agl_segs)
        for s, e in msl_segs:
            agl[s:e] = alt[s:e] - (alt[s:e].min() - agl_ground)
        for s, e in agl_segs:
            agl[s:e] = alt[s:e] - agl_ground
    else:
        agl -= agl.min()

    agl = np.clip(agl, 0, None)
    out = df.copy()
    out["alt_agl"] = agl
    print(f"[alt] After fix: {agl.min():.1f}–{agl.max():.1f} m "
          f"start={agl[0]:.1f} end={agl[-1]:.1f}")
    return out


def wgs84_to_enu(df: pd.DataFrame) -> pd.DataFrame:
    """Convert WGS84 to local ENU coordinates. Adds 'east', 'north', 'up' columns."""
    R = 6_371_000
    phi0 = np.radians(df["lat"].iloc[0])
    lam0 = np.radians(df["lon"].iloc[0])
    alt0 = df["alt_agl"].iloc[0]
    out = df.copy()
    out["east"] = R * np.cos(phi0) * (np.radians(df["lon"]) - lam0)
    out["north"] = R * (np.radians(df["lat"]) - phi0)
    out["up"] = df["alt_agl"] - alt0
    return out


def _compute_speed(df: pd.DataFrame) -> np.ndarray:
    """Compute 3D speed in m/s from ENU coordinates."""
    de = df["east"].diff().fillna(0)
    dn = df["north"].diff().fillna(0)
    du = df["up"].diff().fillna(0)
    dt = df["timestamp"].diff().replace(0, np.nan)
    speed = np.sqrt(de**2 + dn**2 + du**2) / dt
    return speed.fillna(0).clip(lower=0).values


def build_3d_figure(
    df_gps: pd.DataFrame,
    color_by: str = "speed",
    title: str = "3D Flight Trajectory",
) -> go.Figure:
    """Create interactive 3D trajectory plot. Color by 'speed' or 'time'."""
    df = wgs84_to_enu(_to_agl(df_gps))

    if color_by == "speed":
        c_values, c_label, cscale = _compute_speed(df), "Speed (m/s)", "Plasma"
    else:
        c_values = (df["timestamp"] - df["timestamp"].iloc[0]).values
        c_label, cscale = "Time (s)", "Viridis"

    east, north, up = df["east"].values, df["north"].values, df["up"].values

    traces = [
        go.Scatter3d(
            x=east, y=north, z=up, mode="markers",
            marker=dict(size=4, color=c_values, colorscale=cscale,
                       showscale=True, colorbar=dict(title=c_label, thickness=14, x=1.0)),
            hovertemplate=f"East: %{{x:.1f}} m<br>North: %{{y:.1f}} m<br>"
                         f"Alt AGL: %{{z:.1f}} m<br>{c_label}: %{{marker.color:.2f}}<extra></extra>",
            name="GPS fix",
        ),
        go.Scatter3d(x=east, y=north, z=up, mode="lines",
                    line=dict(color="rgba(255,255,255,0.3)", width=3),
                    showlegend=False, hoverinfo="skip"),
    ]

    for idx, label, color in [(0, "Start", "lime"), (-1, "End", "red")]:
        traces.append(go.Scatter3d(
            x=[east[idx]], y=[north[idx]], z=[up[idx]],
            mode="markers+text", marker=dict(size=10, color=color, symbol="diamond"),
            text=[label], textposition="top center", name=label,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="East (m)", backgroundcolor="#111", gridcolor="#333"),
            yaxis=dict(title="North (m)", backgroundcolor="#111", gridcolor="#333"),
            zaxis=dict(title="Alt AGL (m)", backgroundcolor="#111", gridcolor="#333"),
            bgcolor="#0d0d0d", aspectmode="data",
        ),
        paper_bgcolor="#0d0d0d", font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def build_altitude_chart(df_gps: pd.DataFrame) -> go.Figure:
    """Create altitude vs time plot."""
    df = _to_agl(df_gps)
    t = df["timestamp"] - df["timestamp"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=df["alt_agl"], mode="lines", fill="tozeroy",
        line=dict(color="#00b4d8", width=2), fillcolor="rgba(0,180,216,0.15)",
        name="Altitude AGL (m)",
        hovertemplate="t=%{x:.1f}s  alt=%{y:.1f} m<extra></extra>",
    ))
    fig.update_layout(
        title="Altitude AGL over time", xaxis_title="Time (s)", yaxis_title="Altitude AGL (m)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_speed_chart(df_gps: pd.DataFrame) -> go.Figure:
    """Create speed vs time plot (km/h)."""
    df = wgs84_to_enu(_to_agl(df_gps))
    t = df["timestamp"] - df["timestamp"].iloc[0]
    spd = _compute_speed(df) * 3.6
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=spd, mode="lines", line=dict(color="#f77f00", width=2),
        name="Speed (km/h)",
        hovertemplate="t=%{x:.1f}s  v=%{y:.1f} km/h<extra></extra>",
    ))
    fig.update_layout(
        title="Horizontal speed over time", xaxis_title="Time (s)", yaxis_title="Speed (km/h)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


if __name__ == "__main__":
    import sys, os, argparse
    sys.path.insert(0, os.path.dirname(__file__))
    from bin_parser import TelemetryParser

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--color", default="speed", choices=["speed", "time"])
    args = ap.parse_args()

    p = TelemetryParser(args.input)
    df_gps, _ = p.parse()
    build_3d_figure(df_gps, color_by=args.color).show()
