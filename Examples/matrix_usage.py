from scope.compression import CompressionMatrix

cm = CompressionMatrix(
    compressor_names=['lz77'],
    compression_metric_names=['ncc'],
    join_string=' ',
)

test_samples = {
    0: ['Hola', 'holi', 'djsahkdjas', 'djsahkdjas'],
    1: ['Adios', 'adioz']
}

test_sample = 'ola'


results = cm(
    samples=test_sample,
    kw_samples=test_samples
)

print(results)