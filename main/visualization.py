"""
3D flight trajectory visualization for ArduPilot logs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _to_agl(df_gps: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed MSL/AGL altitudes to continuous AGL."""
    JUMP_THRESH = 50
    AGL_MAX_M = 100

    alt = df_gps["alt"].values.astype(float).copy()
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
    out = df_gps.copy()
    out["alt_agl"] = agl
    return out


def wgs84_to_enu(df_gps: pd.DataFrame) -> pd.DataFrame:
    """Convert WGS84 to local ENU coordinates."""
    R = 6_371_000
    phi0 = np.radians(df_gps["lat"].iloc[0])
    lam0 = np.radians(df_gps["lon"].iloc[0])
    alt0 = df_gps["alt_agl"].iloc[0]
    out = df_gps.copy()
    out["east"] = R * np.cos(phi0) * (np.radians(df_gps["lon"]) - lam0)
    out["north"] = R * (np.radians(df_gps["lat"]) - phi0)
    out["up"] = df_gps["alt_agl"] - alt0
    return out


def compute_speed(df: pd.DataFrame) -> np.ndarray:
    """Compute 3D speed in m/s."""
    de = df["east"].diff().fillna(0)
    dn = df["north"].diff().fillna(0)
    du = df["up"].diff().fillna(0)
    dt = df["timestamp"].diff().fillna(0.01)
    dt = np.maximum(dt, 0.01)
    speed = np.sqrt(de**2 + dn**2 + du**2) / dt
    speed = np.minimum(speed, 50)  # макс 50 м/с
    return speed.fillna(0).values


def build_3d_figure(df_gps: pd.DataFrame, title: str = "3D Flight Trajectory") -> go.Figure:
    """Create interactive 3D trajectory plot."""
    df = wgs84_to_enu(_to_agl(df_gps))

    # Якщо точок забагато — проріджуємо
    if len(df) > 3000:
        step = len(df) // 3000
        df = df.iloc[::step].reset_index(drop=True)

    speed = compute_speed(df)
    east, north, up = df["east"].values, df["north"].values, df["up"].values

    fig = go.Figure()

    # Точки траєкторії (кольоровані за швидкістю)
    fig.add_trace(go.Scatter3d(
        x=east, y=north, z=up,
        mode="markers",
        marker=dict(
            size=2,
            color=speed,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Speed (m/s)"),
        ),
        name="Trajectory",
    ))

    # Лінія траєкторії
    fig.add_trace(go.Scatter3d(
        x=east, y=north, z=up,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=2),
        showlegend=False,
    ))

    # Старт
    fig.add_trace(go.Scatter3d(
        x=[east[0]], y=[north[0]], z=[up[0]],
        mode="markers+text",
        marker=dict(size=8, color="lime", symbol="diamond"),
        text=["Start"], textposition="top center",
        name="Start",
    ))

    # Фініш
    fig.add_trace(go.Scatter3d(
        x=[east[-1]], y=[north[-1]], z=[up[-1]],
        mode="markers+text",
        marker=dict(size=8, color="red", symbol="diamond"),
        text=["End"], textposition="top center",
        name="End",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        height=600,
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(color="white"),
    )

    return fig


def build_altitude_chart(df_gps: pd.DataFrame) -> go.Figure:
    """Create altitude vs time plot."""
    df = _to_agl(df_gps)
    t = df["timestamp"] - df["timestamp"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=df["alt_agl"], mode="lines", fill="tozeroy",
        line=dict(color="#00b4d8", width=2),
        name="Altitude (m)",
    ))
    fig.update_layout(
        title="Altitude over time",
        xaxis_title="Time (s)",
        yaxis_title="Altitude (m)",
        height=400,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111",
        font=dict(color="white"),
    )
    return fig


def build_speed_chart(df_gps: pd.DataFrame) -> go.Figure:
    """Create speed vs time plot (km/h)."""
    df = wgs84_to_enu(_to_agl(df_gps))
    t = df["timestamp"] - df["timestamp"].iloc[0]
    spd = compute_speed(df) * 3.6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=spd, mode="lines",
        line=dict(color="#f77f00", width=2),
        name="Speed (km/h)",
    ))
    fig.update_layout(
        title="Speed over time",
        xaxis_title="Time (s)",
        yaxis_title="Speed (km/h)",
        height=400,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111",
        font=dict(color="white"),
    )
    return fig
