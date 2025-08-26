from scope.compression import CompressionMatrix

cm = CompressionMatrix(
    compressor_names=['bz2', 'zlib'],
    compression_metric_names=['ncd', 'cdm'],
    join_string=' ',
    compression_level=9,
    min_size_threshold=50
)

test_samples = {
    0: ['Hola', 'holi', 'djsahkdjas', 'djsahkdjas'],
    1: ['Adios', 'adioz']
}

test_sample = 'Hola'


results = cm(
    samples=test_sample,
    kw_samples=test_samples
)

print(results)