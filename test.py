from cdc.models import EucCDC, CDCEnsabmle, CosCDC, EucCosCDC

from cdc.models import MinMaxCDC


def testModelMinMax(sample, *classes):
    model = MinMaxCDC()
    ensamble = CDCEnsabmle(model=model, compressors=COMPRESSORS,
                           distances=DISTANCES, typ='text')
    
    class_, _, _ = ensamble.predict(sample=sample, classes=list(classes))
    print(class_)
    

def testModelEuc(sample, *classes):    
    model = EucCDC()
    ensamble = CDCEnsabmle(model=model, compressors=COMPRESSORS,
                           distances=DISTANCES, typ='text')
    
    class_, _, _ = ensamble.predict(sample=sample, classes=list(classes))
    print(class_)

def testModelCos(sample, *classes):    
    model = CosCDC()
    ensamble = CDCEnsabmle(model=model, compressors=COMPRESSORS,
                        distances=DISTANCES, typ='text')
        
    class_, _, _ = ensamble.predict(sample=sample, classes=list(classes))
    print(class_)

def testModelEucCosCDC(sample, *classes):    
    model = EucCosCDC()
    ensamble = CDCEnsabmle(model=model, compressors=COMPRESSORS,
                        distances=DISTANCES, typ='text')
        
    class_, _, _ = ensamble.predict(sample=sample, classes=list(classes))
    print(class_)

if __name__ == '__main__':
    
    data1 = ["Hola, ¿cómo te encuentras en este momento?",
             "¡Saludos! ¿Cómo va todo por tu lado?",
             "¿Qué tal estás? Hola desde este lado.",
             "Hola, ¿cómo andas? Espero que todo esté bien."]
    
    data2 = ["Adiós, no me interesa saber cómo estás.",
             "Hola, pero realmente no me importa tu estado de ánimo.",
             "¿Cómo estás? No tengo interés en saberlo.",
             "Saludos, aunque no me importa tu respuesta.",]
    
    data3 = ["¡Buen provecho! ¿Cómo está tu plato?",
             "¿Listo para disfrutar de una deliciosa comida?",
             "Saludos culinarios, ¿qué delicia has probado hoy?",
             "Hola, ¿qué tal una charla sobre tus platillos favoritos?"]
    
    
    COMPRESSORS='__all__'
    DISTANCES='__all__'
    TYP='text_as_array'
    
    classes = [data1, data2, data3]
    
    sample:str = "Que onda!"
    print(f'Expected: 0, {sample}')
    testModelEuc([sample], *classes)
    testModelCos([sample], *classes)
    testModelEucCosCDC([sample], *classes)
    testModelMinMax([sample], *classes)
    
    sample:str = "No me imprtas."
    print(f'Expected: 1, {sample}')
    testModelEuc(sample, *classes)
    testModelCos(sample, *classes)
    testModelEucCosCDC([sample], *classes)
    testModelMinMax([sample], *classes)
    
    sample:str = "Te gusta tu pollo?"
    print(f'Expected: 2, {sample}')
    testModelEuc(sample, *classes)
    testModelCos(sample, *classes)
    testModelEucCosCDC([sample], *classes)
    testModelMinMax([sample], *classes)