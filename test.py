from scope.compression import CompressionMatrix

cm = CompressionMatrix(
    compressors_names=['bz2', 'zlib', 'zstd', 'rle', 'huffman', 'lz77'],
    compression_metric_names=['ncd', 'nrc'],
    join_string='',
    compression_level=9,
    # min_size_threshold=50,
    qval=8
)

test_samples = {
    0: ['Hola', 'holi', 'dasdasdsads'],
    1: ['Adios', 'adioz']
}

test_sample = 'Holaaaaa'


results = cm(
    samples=test_sample,
    kw_samples=test_samples
)


print(results)