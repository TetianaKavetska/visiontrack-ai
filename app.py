import os
import shutil
import subprocess
import tempfile
import time

import cv2
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(page_title="VisionTrack AI | Аналіз трекінгу", page_icon="🎯", layout="wide")
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 85% 0%, #172b4b 0%, #0b1220 45%, #090f1b 100%); color: #e9f2ff; }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
h1,h2,h3 { color: #f3f7ff !important; letter-spacing: -.025em; }
p, label, .stCaption { color: #c4d2e5; }
[data-testid="stMetric"] { background: #14233a; border: 1px solid #2b4260; padding: 1rem; border-radius: 16px; }
[data-testid="stMetricValue"] { color: #77d3ff; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 8px; background: #121f33; padding: 7px; border-radius: 14px; }
[data-testid="stTabs"] button[role="tab"] { border-radius: 10px; color: #c3d1e4; padding: 10px 14px; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: #24476a; color: white; }
.stButton button[kind="primary"], .stDownloadButton button { border-radius: 12px; border: 1px solid #4389bb; }
.stButton button { border-radius: 12px; }
[data-testid="stFileUploader"], [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
.vt-hero { border: 1px solid #294260; border-radius: 22px; padding: 26px 30px; margin-bottom: 22px; background: linear-gradient(120deg,#142b49,#112037 65%,#182a47); }
.vt-kicker { color: #75d6ff; text-transform: uppercase; font-size: .8rem; letter-spacing: .18em; font-weight: 700; }
.vt-hero h1 { margin: 8px 0 6px; font-size: clamp(2rem, 4vw, 3.25rem); }
.vt-hero p { margin: 0; color: #b7cce4; font-size: 1rem; }
.vt-note { border-left: 3px solid #57b9ee; padding: 8px 14px; background: #12253d; border-radius: 6px; color: #d4e7f9; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="vt-hero">
<div class="vt-kicker">Computer Vision · Research Dashboard</div>
<h1>VisionTrack AI</h1>
<p>Порівняльний аналіз моделей виявлення та трекінгу людей у відеопотоці</p>
</div>
""", unsafe_allow_html=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource
def load_model(name):
    return YOLO(name)


if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "video_summary" not in st.session_state:
    st.session_state.video_summary = None
if "video_preview_supported" not in st.session_state:
    st.session_state.video_preview_supported = False


if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

tab_demo, tab_compare, tab_results, tab_mot = st.tabs(
    ["Демо", "Порівняння на відео", "Результати порівняння", "Повний аналіз MOT17"]
)

with tab_demo:
    col1, col2 = st.columns(2)
    model_name = col1.selectbox("Детектор", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    tracker = col2.selectbox("Трекер", ["bytetrack.yaml", "botsort.yaml"])
    mode = st.radio("Режим обробки", ["Тестовий (частина відео)", "Повний (усе відео)"], horizontal=True)
    max_frames = (
        st.slider("Скільки кадрів обробити", 30, 300, 100)
        if mode.startswith("Тестовий") else None
    )
    if max_frames is None:
        st.caption("Повний режим: усі кадри відео. Для початку обирайте короткі відео.")
    st.caption(f"Пристрій обчислень: {device}")

    uploaded_file = st.file_uploader("Завантажте відео", type=["mp4", "avi", "mov"])

    if uploaded_file is not None and st.button("Запустити"):
        st.session_state.video_result = None
        st.session_state.video_summary = None
        input_path = None
        output_path = None
        converted_path = None
        writer = None
        capture = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                input_path = tmp.name

            capture = cv2.VideoCapture(input_path)
            if not capture.isOpened():
                raise ValueError("Не вдалося відкрити відео. Спробуйте інший файл.")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0 or source_fps > 240:
                source_fps = 25.0
            total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            expected_frames = (
                min(max_frames, total_source_frames) if max_frames is not None and total_source_frames > 0
                else max_frames if max_frames is not None
                else total_source_frames if total_source_frames > 0 else None
            )
            capture.release()
            capture = None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out:
                output_path = out.name

            model = load_model(model_name)
            frame_box = st.empty()
            progress_bar = st.progress(0)
            info = st.empty()
            n = 0
            ids = set()
            start = time.perf_counter()

            for r in model.track(
                input_path,
                classes=[0],
                tracker=tracker,
                device=device,
                stream=True,
                verbose=False,
            ):
                annotated = r.plot()  # OpenCV: BGR
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Не вдалося створити вихідне MP4-відео.")

                writer.write(annotated)
                n += 1
                if r.boxes.id is not None:
                    ids.update(r.boxes.id.int().cpu().tolist())

                frame_box.image(annotated[:, :, ::-1], width="stretch")
                speed_ms = sum(float(r.speed.get(key, 0)) for key in ("preprocess", "inference", "postprocess"))
                model_fps = 1000 / speed_ms if speed_ms > 0 else 0
                elapsed = time.perf_counter() - start
                actual_fps = n / elapsed if elapsed > 0 else 0
                info.write(
                    f"Кадр: {n} | Швидкість моделі: {model_fps:.1f} FPS | "
                    f"Фактична швидкість: {actual_fps:.1f} FPS | "
                    f"Людей у кадрі: {len(r.boxes)} | Унікальних ID: {len(ids)}"
                )
                if expected_frames is not None:
                    progress_bar.progress(min(n / max(expected_frames, 1), 1.0))
                if max_frames is not None and n >= max_frames:
                    break

            if writer is not None:
                writer.release()
                writer = None
            if n == 0:
                raise ValueError("У відео немає кадрів, які можна обробити.")

            elapsed = time.perf_counter() - start
            # Для відтворення MP4 у браузері бажаний кодек H.264.
            # Якщо ffmpeg недоступний, зберігаємо MP4V для завантаження.
            final_path = output_path
            preview_supported = False
            if shutil.which("ffmpeg"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as converted:
                    converted_path = converted.name
                command = [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", output_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    converted_path,
                ]
                conversion = subprocess.run(command, capture_output=True, text=True)
                if conversion.returncode == 0 and os.path.getsize(converted_path) > 0:
                    final_path = converted_path
                    preview_supported = True

            with open(final_path, "rb") as video_file:
                st.session_state.video_result = video_file.read()
            st.session_state.video_preview_supported = preview_supported
            st.session_state.video_summary = (
                f"Обробку завершено! Кадрів: {n} | "
                f"Час обробки: {elapsed:.2f} с | "
                f"Середній FPS: {n / elapsed:.1f} | "
                f"Унікальних ID: {len(ids)}"
            )
            progress_bar.progress(1.0)
        except Exception as exc:
            st.error(f"Помилка обробки: {exc}")
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            for path in (input_path, output_path, converted_path):
                if path and os.path.exists(path):
                    os.unlink(path)

    if st.session_state.video_result is not None:
        st.success(st.session_state.video_summary)
        st.subheader("Результат обробки")
        if st.session_state.video_preview_supported:
            st.video(st.session_state.video_result, format="video/mp4")
        else:
            st.info(
                "Відео збережено, але для відтворення безпосередньо у браузері "
                "потрібен FFmpeg (H.264). Файл можна завантажити нижче."
            )
        st.download_button(
            "Завантажити оброблене відео (MP4)",
            data=st.session_state.video_result,
            file_name="tracked_people.mp4",
            mime="video/mp4",
        )

with tab_compare:
    st.subheader("Порівняння детекторів на одному відео")
    st.write(
        "Усі вибрані моделі обробляють однакові початкові кадри відео "
        "з одним і тим самим трекером. Вимірюємо повний час обробки без "
        "постійного оновлення зображення на сайті."
    )
    compare_file = st.file_uploader(
        "Відео для порівняння", type=["mp4", "avi", "mov"], key="compare_upload"
    )
    compare_models = st.multiselect(
        "Моделі", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    )
    compare_tracker = st.selectbox(
        "Трекер для всіх моделей", ["bytetrack.yaml", "botsort.yaml"],
        key="compare_tracker",
    )
    compare_frames = st.slider(
        "Кадрів для кожної моделі", 30, 300, 100, key="compare_frames"
    )
    st.caption(f"Обчислювальний пристрій: {device}. Порівняння може зайняти кілька хвилин.")

    if st.button("Порівняти моделі", key="run_comparison"):
        if compare_file is None:
            st.warning("Спочатку завантаж відео.")
        elif not compare_models:
            st.warning("Вибери хоча б одну модель.")
        else:
            st.session_state.comparison_result = None
            compare_path = None
            try:
                suffix = os.path.splitext(compare_file.name)[1].lower() or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(compare_file.getvalue())
                    compare_path = tmp.name

                check = cv2.VideoCapture(compare_path)
                try:
                    if not check.isOpened():
                        raise ValueError("Відеофайл не відкривається.")
                    available = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    check.release()
                if available > 0 and available < compare_frames:
                    st.info(f"У відео лише {available} кадрів: обробимо всі доступні.")

                rows = []
                status = st.empty()
                progress = st.progress(0)
                for index, name in enumerate(compare_models):
                    status.write(f"Обробка {name} ({index + 1}/{len(compare_models)})...")
                    # Створюємо окремий об'єкт моделі, щоб не переносити стан
                    # трекера між незалежними експериментами.
                    model = YOLO(name)
                    count = 0
                    unique_ids = set()
                    start = time.perf_counter()
                    for result in model.track(
                        compare_path,
                        classes=[0],
                        tracker=compare_tracker,
                        device=device,
                        stream=True,
                        verbose=False,
                    ):
                        count += 1
                        if result.boxes.id is not None:
                            unique_ids.update(result.boxes.id.int().cpu().tolist())
                        if count >= compare_frames:
                            break
                    elapsed = time.perf_counter() - start
                    if count == 0:
                        raise ValueError("Не вдалося обробити жодного кадру.")
                    rows.append({
                        "detector": name,
                        "tracker": compare_tracker,
                        "frames": count,
                        "time_seconds": round(elapsed, 2),
                        "fps": round(count / elapsed, 2) if elapsed > 0 else 0,
                        "unique_ids": len(unique_ids),
                        "device": device,
                    })
                    progress.progress((index + 1) / len(compare_models))
                st.session_state.comparison_result = pd.DataFrame(rows)
                status.success("Порівняння завершено!")
            except Exception as exc:
                st.error(f"Помилка порівняння: {exc}")
            finally:
                if compare_path and os.path.exists(compare_path):
                    os.unlink(compare_path)

    if st.session_state.comparison_result is not None:
        comparison = st.session_state.comparison_result
        st.dataframe(comparison, width="stretch")
        st.bar_chart(comparison.set_index("detector")["fps"])
        st.download_button(
            "Завантажити результати (CSV)",
            comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name="comparison_results.csv",
            mime="text/csv",
        )
        st.caption(
            "FPS — фактична швидкість обробки в цьому тесті. "
            "Кількість ID не є кількістю різних людей або показником точності. "
            "Для оцінки якості трекінгу (IDF1/MOTA) потрібна розмічена вибірка. "
            "Перший запуск може включати завантаження ваг і прогрів моделі."
        )

with tab_results:
    st.subheader("MOT17-09: точність і швидкість (Colab T4)")
    if os.path.exists("results_mot17.csv"):
        st.dataframe(pd.read_csv("results_mot17.csv"), width="stretch")
    else:
        st.info("Файл results_mot17.csv не знайдено в папці проєкту.")
    if os.path.exists("accuracy_vs_fps.png"):
        st.image("accuracy_vs_fps.png", width="stretch")

    st.subheader("Швидкість на MacBook (власне відео)")
    if os.path.exists("results.csv"):
        st.dataframe(pd.read_csv("results.csv"), width="stretch")
    if os.path.exists("fps_chart.png"):
        st.image("fps_chart.png", width="stretch")


with tab_mot:
    st.subheader("Повний аналіз MOT17: 4 сцени × 3 детектори × 2 трекери")
    st.caption("Результати експериментів Google Colab T4. MOTA та IDF1 показані як частки або відсотки; FPS — кадри за секунду.")
    mot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_mot17_all.csv")
    if not os.path.isfile(mot_path):
        st.warning("Не знайдено results_mot17_all.csv. Поклади CSV у ту саму папку, що й app.py, і онови сторінку.")
    else:
        try:
            mot = pd.read_csv(mot_path)
            required = {"sequence", "detector", "tracker", "fps", "MOTA", "IDF1", "ID_switches", "precision", "recall"}
            missing = required.difference(mot.columns)
            if missing:
                st.error("У CSV бракує стовпців: " + ", ".join(sorted(missing)))
            else:
                numeric_cols = ["fps", "MOTA", "IDF1", "ID_switches", "precision", "recall"]
                for col in numeric_cols:
                    mot[col] = pd.to_numeric(mot[col], errors="coerce")
                if mot[numeric_cols].isna().any().any():
                    st.error("У CSV є пропущені або некоректні числові значення. Перевір файл.")
                else:
                    mot["combination"] = mot["detector"].str.replace(".pt", "", regex=False) + " + " + mot["tracker"].str.replace(".yaml", "", regex=False)
                    scenes = sorted(mot["sequence"].unique().tolist())
                    chosen = st.selectbox("Сцена для аналізу", ["Усі сцени"] + scenes, key="mot_scene")
                    filtered = mot if chosen == "Усі сцени" else mot[mot["sequence"] == chosen]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Експерименти", len(filtered))
                    c2.metric("Детектори", filtered["detector"].nunique())
                    c3.metric("Трекери", filtered["tracker"].nunique())
                    c4.metric("Сцени", filtered["sequence"].nunique())
                    st.dataframe(filtered.drop(columns="combination"), width="stretch", hide_index=True)
                    st.download_button("Завантажити вибрані дані (CSV)", filtered.drop(columns="combination").to_csv(index=False).encode("utf-8-sig"), file_name="mot17_selected.csv", mime="text/csv")

                    summary = (filtered.groupby(["detector", "tracker", "combination"], as_index=False)
                               .agg(fps=("fps", "mean"), MOTA=("MOTA", "mean"), IDF1=("IDF1", "mean"),
                                    precision=("precision", "mean"), recall=("recall", "mean")))
                    summary["MOTA_%"] = summary["MOTA"] * 100
                    summary["IDF1_%"] = summary["IDF1"] * 100
                    st.subheader("Середні показники за вибраними сценами")
                    st.dataframe(summary[["detector", "tracker", "fps", "MOTA_%", "IDF1_%", "precision", "recall"]].round(2), width="stretch", hide_index=True)
                    st.caption("Це прості середні арифметичні за сценами, а не загальні (pooled) метрики MOTChallenge. Для офіційної оцінки потрібна перевірка протоколу та еталонної розмітки.")
                    st.download_button("Завантажити середні результати (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), file_name="mot17_summary.csv", mime="text/csv")

                    st.subheader("Порівняльні графіки")
                    for metric, title in [("fps", "Швидкість (FPS)"), ("MOTA_%", "MOTA (%)"), ("IDF1_%", "IDF1 (%)")]:
                        st.markdown("**" + title + "**")
                        chart_data = summary.pivot(index="detector", columns="tracker", values=metric)
                        chart_data = chart_data.reindex(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]).dropna(how="all")
                        st.bar_chart(chart_data)

                    fastest = summary.loc[summary["fps"].idxmax()]
                    best_idf = summary.loc[summary["IDF1"].idxmax()]
                    best_mota = summary.loc[summary["MOTA"].idxmax()]
                    st.subheader("Автоматичні висновки")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Максимальний FPS", f"{fastest['fps']:.1f}")
                    m2.metric("Найкращий IDF1", f"{best_idf['IDF1']*100:.1f}%")
                    m3.metric("Найкращий MOTA", f"{best_mota['MOTA']*100:.1f}%")
                    st.write(f"Найвища середня швидкість: **{fastest['combination']}** — **{fastest['fps']:.2f} FPS**.")
                    st.write(f"Найвищий середній IDF1: **{best_idf['combination']}** — **{best_idf['IDF1'] * 100:.2f}%**.")
                    st.write(f"Найвищий середній MOTA: **{best_mota['combination']}** — **{best_mota['MOTA'] * 100:.2f}%**.")
                    st.info("Вищий FPS не гарантує кращої точності. Кількість ID switches між різними сценами не слід порівнювати без урахування тривалості та кількості об'єктів.")
        except Exception as exc:
            st.error(f"Не вдалося прочитати результати MOT17: {exc}")
