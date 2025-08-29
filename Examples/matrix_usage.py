import time
from scope.compression import CompressionMatrix

compressor_names=['gzip']
compression_metric_names=['ncc']


# Setup
cm = CompressionMatrix(
    compressor_names=compressor_names,
    compression_metric_names=compression_metric_names,
    join_string=' ',
)

test_samples = { 
    1: [ 
        "Una obra maestra cinematográfica que combina actuaciones excepcionales con una dirección brillante. Cada escena está cuidadosamente crafteada y la banda sonora es simplemente espectacular.",
        "Increíble película que supera todas las expectativas. Los efectos visuales son impresionantes y la historia te mantiene en el borde del asiento desde el primer minuto hasta el último.",
        "Una experiencia cinematográfica única e inolvidable. El guión es inteligente, los personajes están perfectamente desarrollados y la cinematografía es absolutamente hermosa.",
        "Excelente film que demuestra el poder del cine para emocionar y inspirar. Las actuaciones son soberbias y la narrativa fluye de manera perfecta creando una experiencia mágica.",
        "Brillante adaptación que honra el material original mientras añade elementos frescos. La dirección es magistral y cada actor entrega una performance memorable y convincente." 
    ],
    0: [ 
        "Una completa pérdida de tiempo que no logra conectar con la audiencia. El guión es predecible, las actuaciones son forzadas y la dirección carece de visión clara.",
        "Película decepcionante que desperdicia un gran potencial. Los diálogos son torpes, la trama tiene agujeros enormes y los efectos especiales parecen de bajo presupuesto.",
        "Un desastre cinematográfico que no entiende su propio género. Los personajes son unidimensionales, el ritmo es desesperantemente lento y el final es completamente insatisfactorio.",
        "Film aburrido y mal ejecutado que no cumple ninguna de sus promesas. La actuación principal es terrible, la fotografía es mediocre y la banda sonora es completamente olvidable.",
        "Una producción sin alma que se siente como un producto manufacturado. Carece de originalidad, profundidad emocional y cualquier elemento que la haga memorable o valiosa."
    ] 
}

test_sample = "Fantástica película que combina acción emocionante con momentos de gran profundidad emocional. Los actores entregan performances convincentes y la dirección mantiene un ritmo perfecto throughout."

# Test
print("Testing CompressionMatrix Serie...")

start = time.time()
results = cm(samples=test_sample, kw_samples=test_samples)
end = time.time()

print(f"Time: {end - start:.4f}s")
print("Done!")


# Setup
cm = CompressionMatrix(
    compressor_names=compressor_names,
    compression_metric_names=compression_metric_names,
    join_string=' ',
    n_jobs=-1
)

test_samples = { 
    1: [ 
        "Una obra maestra cinematográfica que combina actuaciones excepcionales con una dirección brillante. Cada escena está cuidadosamente crafteada y la banda sonora es simplemente espectacular.",
        "Increíble película que supera todas las expectativas. Los efectos visuales son impresionantes y la historia te mantiene en el borde del asiento desde el primer minuto hasta el último.",
        "Una experiencia cinematográfica única e inolvidable. El guión es inteligente, los personajes están perfectamente desarrollados y la cinematografía es absolutamente hermosa.",
        "Excelente film que demuestra el poder del cine para emocionar y inspirar. Las actuaciones son soberbias y la narrativa fluye de manera perfecta creando una experiencia mágica.",
        "Brillante adaptación que honra el material original mientras añade elementos frescos. La dirección es magistral y cada actor entrega una performance memorable y convincente." 
    ],
    0: [ 
        "Una completa pérdida de tiempo que no logra conectar con la audiencia. El guión es predecible, las actuaciones son forzadas y la dirección carece de visión clara.",
        "Película decepcionante que desperdicia un gran potencial. Los diálogos son torpes, la trama tiene agujeros enormes y los efectos especiales parecen de bajo presupuesto.",
        "Un desastre cinematográfico que no entiende su propio género. Los personajes son unidimensionales, el ritmo es desesperantemente lento y el final es completamente insatisfactorio.",
        "Film aburrido y mal ejecutado que no cumple ninguna de sus promesas. La actuación principal es terrible, la fotografía es mediocre y la banda sonora es completamente olvidable.",
        "Una producción sin alma que se siente como un producto manufacturado. Carece de originalidad, profundidad emocional y cualquier elemento que la haga memorable o valiosa."
    ] 
}

test_sample = "Fantástica película que combina acción emocionante con momentos de gran profundidad emocional. Los actores entregan performances convincentes y la dirección mantiene un ritmo perfecto throughout."

# Test
print("Testing CompressionMatrix Parallel...")

start = time.time()
results = cm(samples=test_sample, kw_samples=test_samples)
end = time.time()

print(f"Time: {end - start:.4f}s")
print("Done!")