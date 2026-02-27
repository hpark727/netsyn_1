from data_processing import convert_to_df, bin_frames, aggregate_bins, plot_bytes_per_bin, vectorize_capture


CLASS_1_PATH = "/Users/haelpark/Desktop/class_1_data"
CLASS_2_PATH = "/Users/haelpark/Desktop/class_2_data"

for i in range(1, 16):
    path = f"{CLASS_1_PATH}/cap_{i}.pcap"
    df = convert_to_df(path)
    df_binned = bin_frames(df)
    vec = vectorize_capture(df_binned, label = 0, save_dir = "/Users/haelpark/Desktop/c1_embeddings", capture_id=f"embedding_{i}")

for i in range(1, 16):
    path = f"{CLASS_2_PATH}/cap_{i}.pcap"
    df = convert_to_df(path)
    df_binned = bin_frames(df)
    vec = vectorize_capture(df_binned, label = 1, save_dir = "/Users/haelpark/Desktop/c1_embeddings", capture_id=f"embedding_{i}")



