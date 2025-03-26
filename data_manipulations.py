# normalized = (block_rss - min_rss) / (max_rss - min_rss)
def normalize(data, min, max):
    return (data - min) / (max - min)

def denormalize(normalized, min, max):
    return normalized * (max - min) + min