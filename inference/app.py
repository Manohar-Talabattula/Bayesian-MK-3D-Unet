import json
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import streamlit as st

from inference.predict import (
    load_models,
    predict_case,
    DEFAULT_THRESHOLD,
    DEFAULT_GATE_MARGIN_MM,
    load_lits_kits_models,
    predict_lits_kits_case,
)

st.set_page_config(
    page_title="Bayesian MK-3D-UNet",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading pancreas models...")
def get_models():
    return load_models()

@st.cache_resource(show_spinner="Loading LiTS/KiTS models...")
def get_lits_kits_models():
    return load_lits_kits_models()


st.title("Bayesian MK-3D-UNet")
st.subheader("3D CT Pathology Segmentation")

mode = st.selectbox(
    "Select segmentation workflow",
    [
        "Pancreas (Task07)",
        "Liver/Kidney (LiTS + KiTS19)",
    ],
)

is_lits_kits = mode == "Liver/Kidney (LiTS + KiTS19)"

if is_lits_kits:
    st.write(
        "Upload a 3D abdominal CT scan. The application uses "
        "Stage 1 liver/kidney anatomy localization followed by "
        "the general LiTS/KiTS Stage 2 pathology model."
    )
    default_threshold = 0.65
    default_margin = 15.0
    margin_label = "Liver/Kidney anatomy gate margin (mm)"
else:
    st.write(
        "Upload a 3D abdominal CT scan. The application runs "
        "the Task07 pancreas-guided lesion segmentation pipeline."
    )
    default_threshold = float(DEFAULT_THRESHOLD)
    default_margin = float(DEFAULT_GATE_MARGIN_MM)
    margin_label = "Pancreas gate margin (mm)"

st.warning(
    "Research prototype only. Predictions are not clinically verified "
    "and must not be used for diagnosis or treatment."
)

uploaded_file = st.file_uploader(
    "Upload a 3D CT scan (.nii or .nii.gz)",
    type=["nii", "gz"],
)

col1, col2 = st.columns(2)

with col1:
    threshold = st.slider(
        "Pathology probability threshold",
        min_value=0.10,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
    )

with col2:
    gate_margin = st.slider(
        margin_label,
        min_value=0.0,
        max_value=30.0,
        value=default_margin,
        step=1.0,
    )

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        st.error("Please upload a valid .nii or .nii.gz file.")
        st.stop()

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button(
        "Run CT Segmentation",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Loading models and processing CT scan..."):
                if is_lits_kits:
                    stage1_model, stage2_model = get_lits_kits_models()
                else:
                    stage1_model, stage2_model = get_models()

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    input_path = temp_dir / uploaded_file.name
                    input_path.write_bytes(uploaded_file.getvalue())

                    if is_lits_kits:
                        result = predict_lits_kits_case(
                            image_path=input_path,
                            stage1_model=stage1_model,
                            stage2_model=stage2_model,
                            threshold=threshold,
                            gate_margin_mm=gate_margin,
                        )
                        measurements = result["report"][
                            "final_pathology_measurements"
                        ]
                    else:
                        result = predict_case(
                            image_path=input_path,
                            stage1_model=stage1_model,
                            stage2_model=stage2_model,
                            threshold=threshold,
                            gate_margin_mm=gate_margin,
                        )
                        measurements = result["report"][
                            "final_lesion_measurements"
                        ]

                    mask_path = temp_dir / "predicted_pathology_mask.nii.gz"
                    probability_path = temp_dir / "pathology_probability.nii.gz"

                    nib.save(result["mask_nii"], str(mask_path))
                    nib.save(result["probability_nii"], str(probability_path))

                    st.session_state["prediction_result"] = {
                        "mode": mode,
                        "report": result["report"],
                        "measurements": measurements,
                        "mask_bytes": mask_path.read_bytes(),
                        "probability_bytes": probability_path.read_bytes(),
                    }

            st.success("Prediction completed successfully.")

        except Exception as exc:
            st.session_state.pop("prediction_result", None)
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)


result_data = st.session_state.get("prediction_result")

if result_data is not None:
    report = result_data["report"]
    measurements = result_data["measurements"]
    result_mode = result_data["mode"]

    st.divider()
    st.header("Prediction Results")
    st.caption(f"Workflow: {result_mode}")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Predicted pathology volume",
        f'{measurements["predicted_volume_ml"]:.3f} mL',
    )

    metric2.metric(
        "Predicted pathology voxels",
        f'{measurements["total_predicted_voxels"]:,}',
    )

    metric3.metric(
        "Connected components",
        measurements["components_at_least_10_voxels"],
    )

    if result_mode == "Liver/Kidney (LiTS + KiTS19)":
        st.subheader("Liver/Kidney anatomy localization")

        st.write(
            "Mapped liver/kidney gate voxels:",
            f'{report["stage1"]["mapped_anatomy_gate_voxels"]:,}',
        )

        st.write(
            "Anatomy gate margin:",
            f'{report["anatomy_gate"]["margin_mm"]:.1f} mm',
        )

        st.write(
            "Pathology probability threshold:",
            report["stage2"]["probability_threshold"],
        )

        st.caption(
            "The gate uses Stage 1 classes 2, 3 and 6: "
            "right kidney, left kidney and liver."
        )

    else:
        st.subheader("Pancreas localization")

        st.write(
            "Pancreas voxels mapped to Stage 2 grid:",
            f'{report["stage1"]["mapped_pancreas_voxels"]:,}',
        )

        st.write(
            "Pancreas gate margin:",
            f'{report["stage3"]["gate_margin_mm"]:.1f} mm',
        )

    st.subheader("Pathology components")

    components = measurements["reported_components"]

    if components:
        st.dataframe(
            [
                {
                    "Component": item["component_id"],
                    "Voxels": item["voxel_count"],
                    "Volume (mL)": round(item["volume_ml"], 4),
                    "Dimensions (mm)": [
                        round(x, 2)
                        for x in item["dimensions_grid_axes_mm"]
                    ],
                    "Centroid RAS (mm)": [
                        round(x, 2)
                        for x in item["centroid_ras_mm"]
                    ],
                }
                for item in components
            ],
            use_container_width=True,
        )
    else:
        st.info(
            "No connected components with at least 10 voxels "
            "were detected in the final prediction."
        )

    st.subheader("Download results")

    download1, download2, download3 = st.columns(3)

    with download1:
        st.download_button(
            "Download pathology mask",
            data=result_data["mask_bytes"],
            file_name="predicted_pathology_mask.nii.gz",
            mime="application/gzip",
            use_container_width=True,
        )

    with download2:
        st.download_button(
            "Download probability map",
            data=result_data["probability_bytes"],
            file_name="pathology_probability.nii.gz",
            mime="application/gzip",
            use_container_width=True,
        )

    with download3:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2, allow_nan=False),
            file_name="pathology_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("View complete JSON report"):
        st.json(report)

st.divider()
st.caption(
    "Bayesian MK-3D-UNet | Academic research prototype | "
    "Not validated for clinical use"
)
