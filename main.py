import streamlit as st
import numpy as np
import tempfile
import os
from stl import mesh
import trimesh

# Function to calculate similarity using point distances
def calculate_similarity(reference_mesh, target_mesh):
    ref_points = reference_mesh.vertices
    target_points = target_mesh.vertices
    distances = np.linalg.norm(ref_points[:, None] - target_points, axis=-1)
    return np.mean(np.min(distances, axis=1))

# Function to load STL file and convert to mesh
def load_stl_to_mesh(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name

    stl_mesh = trimesh.load_mesh(temp_filename)
    os.remove(temp_filename)
    
    return stl_mesh

# Streamlit app
def main():
    logo_path = "logo.png"
    st.image(logo_path, use_container_width=True)

    st.title("DR. Janafza AI Cast Editor")

    st.sidebar.header("Upload Existing Patient's STL Files")
    reference_files = st.sidebar.file_uploader("Required Format: STL", type=["stl"], accept_multiple_files=True)

    st.sidebar.header("Upload _qaleb STL Files")
    qaleb_files = st.sidebar.file_uploader("Required Format: ..._qaleb.STL", type=["stl"], accept_multiple_files=True)

    st.sidebar.header("Upload New Patient's STL File")
    new_file = st.sidebar.file_uploader("Required Format: STL", type=["stl"])

    if st.button("Generate"):
        if not reference_files or not new_file or not qaleb_files:
            st.error("Please upload reference STL files, corresponding _qaleb STL files, and a new STL file.")
            return

        st.write("Processing Existing Patient's STL Files...")
        reference_meshes = []
        reference_filenames = []
        qaleb_mapping = {}

        for ref_file in reference_files:
            try:
                reference_meshes.append(load_stl_to_mesh(ref_file))
                reference_filenames.append(ref_file.name)
                qaleb_name = ref_file.name.replace(".stl", "_qaleb.stl")
                qaleb_mapping[ref_file.name] = qaleb_name
            except Exception as e:
                st.error(f"Failed to process {ref_file.name}: {e}")

        qaleb_filenames = {file.name: file for file in qaleb_files}

        st.write("Processing the new patient's STL file...")
        try:
            new_mesh = load_stl_to_mesh(new_file)
        except Exception as e:
            st.error(f"Failed to process new file {new_file.name}: {e}")
            return

        st.write("Generating the new STL file using Artificial Intelligence...")
        similarity_scores = []
        for ref_mesh in reference_meshes:
            score = calculate_similarity(ref_mesh, new_mesh)
            similarity_scores.append(score)

        most_similar_index = np.argmin(similarity_scores)
        most_similar_file = reference_filenames[most_similar_index]
        most_similar_score = similarity_scores[most_similar_index]
        most_similar_qaleb_file = qaleb_mapping.get(most_similar_file)

        if most_similar_qaleb_file not in qaleb_filenames:
            st.error(f"Matching _qaleb file `{most_similar_qaleb_file}` not found. Ensure it is uploaded.")
            return

        qaleb_file = qaleb_filenames[most_similar_qaleb_file]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as temp_qaleb:
            temp_qaleb_filename = "AI Generated STL.stl"
            temp_qaleb.write(qaleb_file.read())
            temp_qaleb_path = temp_qaleb.name

        st.write(f"### Similarity Score: `{most_similar_score:.4f}`")
        st.download_button(label="Download AI Generated File", data=open(temp_qaleb_path, "rb"), file_name=temp_qaleb_filename, mime="application/octet-stream")

        os.remove(temp_qaleb_path)

if __name__ == "__main__":
    main()
