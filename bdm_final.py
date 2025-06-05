
# Đây chỉ là stub. Bạn cần thay thế bằng nội dung từ bdm_final.py của bạn.
def run_pipeline(mat_path):
    import pandas as pd
    print("Running pipeline on", mat_path)
    # Dummy data để test giao diện Streamlit
    df = pd.DataFrame({
        'Feature': ['feat_1', 'feat_2'],
        'Importance': [0.8, 0.2]
    })
    edge_df = pd.DataFrame({
        'Edge': ['0-1', '1-2'],
        'Importance': [0.6, 0.4]
    })
    return None, df, edge_df
