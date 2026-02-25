from data_processing import convert_to_df, bin_frames, aggregate_bins, plot_bytes_per_bin


PCAP_PATH = ["cla1.pcap", "cla2.pcap"]

c1 = convert_to_df(PCAP_PATH[0])
c2 = convert_to_df(PCAP_PATH[1])

print(c2.head)

c1_binned = bin_frames(c1)
c2_binned = bin_frames(c2)

c1_agg = aggregate_bins(c1_binned)
c2_agg = aggregate_bins(c2_binned)

plot_bytes_per_bin(c2_agg["bin_start"], c2_agg["bytes_total"])

