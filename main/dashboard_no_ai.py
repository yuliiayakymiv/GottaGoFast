"""
Interactive dashboard for drone flight analysis.

This module provides a web interface for uploading Ardupilot .bin flight logs,
displaying computed metrics, and visualizing 3D trajectory.
"""

import os
import tempfile
import streamlit as st

from bin_parser import TelemetryParser
from analytics import get_metrics
from visualization import build_3d_figure

st.set_page_config(
    page_title="Аналізатор польотів",
    page_icon="🚁",
    layout="wide"
)

st.title("Аналізатор польотів дронів")
st.markdown("Завантаж файл `.bin` і отримай аналіз польоту")

# File upload widget
uploaded_file = st.file_uploader(
    "Виберіть BIN файл польоту",
    type=['bin']
)

if uploaded_file is not None:
    st.success(f"Файл {uploaded_file.name} завантажено")

    # Save uploaded file to temporary storage
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Parse log and compute metrics
    with st.spinner("Аналізую лог..."):
        try:
            parser = TelemetryParser(tmp_path)
            df_gps, df_imu = parser.parse()
            metrics = get_metrics(df_gps, df_imu)

            # Main metrics (4 columns)
            st.subheader("Метрики польоту")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Тривалість", f"{metrics.get('duration_s', 0):.1f} с")
            with col2:
                st.metric("Дистанція", f"{metrics.get('total_distance_m', 0):.0f} м")
            with col3:
                st.metric(
                    "Макс. швидкість (гориз.)",
                    f"{metrics.get('max_speed_h_kmh', 0):.1f} м/с"
                )
            with col4:
                st.metric(
                    "Макс. набір висоти",
                    f"{metrics.get('max_altitude_gain_m', 0):.0f} м"
                )

            # Detailed metrics (expandable)
            with st.expander("Детальні метрики"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Вертикальна швидкість (макс.)",
                        f"{metrics.get('max_speed_v_ms', 0):.1f} м/с"
                    )
                with col_b:
                    st.metric(
                        "Прискорення (макс.)",
                        f"{metrics.get('max_acceleration_ms2', 0):.1f} м/с²"
                    )
                with col_c:
                    st.metric(
                        "Швидкість з IMU (макс.)",
                        f"{metrics.get('max_speed_imu_ms', 0):.1f} м/с"
                    )

            # 3D Visualization
            st.subheader("3D Траєкторія польоту")
            fig_3d = build_3d_figure(df_gps)
            st.plotly_chart(fig_3d, use_container_width=True)

        except (ValueError, IOError, KeyError) as e:
            st.error(f"Помилка при аналізі: {str(e)}")

    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except PermissionError:
        pass  # File already deleted or still in use

else:
    st.info("Завантаж BIN файл, щоб почати")
