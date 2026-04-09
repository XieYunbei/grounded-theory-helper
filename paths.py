import os
def get_project_paths(username, project_name):
    user_base = os.path.join("users_data", username)
    project_root = os.path.join(user_base, project_name)

    dirs = {
        "user_base": user_base,
        "project_root": project_root,

        # Step 1
        "preprocessed": os.path.join(project_root, "1_preprocessed_data"),

        # Step 2
        "opening_root": os.path.join(project_root, "2_opening_coding"),
        "opening_autosave": os.path.join(project_root, "2_opening_coding", "_autosave"),
        "opening_final": os.path.join(project_root, "2_opening_coding", "_final"),

        # Step 3
        "analysis_root": os.path.join(project_root, "3_analysis_cache"),
        "analysis_autosave": os.path.join(project_root, "3_analysis_cache","_autosave"),
        "analysis_final": os.path.join(project_root, "3_analysis_cache","_final"),

        # Step 4
        "axial_root": os.path.join(project_root, "4_axial_coding"),
        "axial_autosave": os.path.join(project_root, "4_axial_coding","_autosave"),
        "axial_final": os.path.join(project_root, "4_axial_coding","_final"),

        # Step 5
        "theory": os.path.join(project_root, "5_theoretical_model"),
    }

    # ✅ 统一创建
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs