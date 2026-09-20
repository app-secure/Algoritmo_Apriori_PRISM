import csv
from itertools import combinations

def cargar_datos(ruta_archivo):
    with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        items = [col for col in lector.fieldnames if col != 'Transaccion_ID']
        transacciones = []
        for fila in lector:
            transaccion = set(item for item in items if fila[item] == '1')
            transacciones.append(transaccion)
    return transacciones, items

def contar_cobertura(itemset, transacciones):
    conteo = 0
    for t in transacciones:
        if itemset.issubset(t):
            conteo += 1
    return conteo

def ejecutar_apriori(ruta_archivo, min_soporte=0.70, min_confianza=0.85):
    transacciones, items = cargar_datos(ruta_archivo)
    total_transacciones = len(transacciones)
    
    cobertura_minima = int(total_transacciones * min_soporte)
    
    print("=" * 80)
    print("ALGORITMO APRIORI - APRENDIZAJE BASADO EN REGLAS DE ASOCIACION")
    print("=" * 80)
    print(f"Total de observaciones (N): {total_transacciones}")
    print(f"Hiperparametro Soporte Minimo: {min_soporte:.2f} ({min_soporte*100:.1f}%)")
    print(f"Hiperparametro Confianza Minima: {min_confianza:.2f} ({min_confianza*100:.1f}%)")
    print(f"FASE 0: Cobertura Minima requerida = {cobertura_minima} transacciones")
    print("=" * 80)
    
    itemsets_frecuentes = {}
    
    frecuentes_k1 = {}
    for item in items:
        cob = contar_cobertura({item}, transacciones)
        if cob >= cobertura_minima:
            frecuentes_k1[frozenset([item])] = cob
            
    itemsets_frecuentes[1] = frecuentes_k1
    
    print("\nFASE 1: FILTRADO DE ITEMSETS FRECUENTES")
    print(f"Nivel K=1: {len(frecuentes_k1)} items cumplen la cobertura minima")
    for it, cob in sorted(frecuentes_k1.items(), key=lambda x: x[1], reverse=True):
        sop = cob / total_transacciones
        print(f"   {set(it)} -> Cobertura: {cob}, Soporte: {sop:.4f}")
        
    k = 2
    while True:
        items_anteriores = list(itemsets_frecuentes[k - 1].keys())
        vocabulario_items = set()
        for it in items_anteriores:
            vocabulario_items.update(it)
            
        candidatos = set()
        for comb in combinations(vocabulario_items, k):
            candidato = frozenset(comb)
            subconjuntos_validos = True
            for sub in combinations(candidato, k - 1):
                if frozenset(sub) not in itemsets_frecuentes[k - 1]:
                    subconjuntos_validos = False
                    break
            if subconjuntos_validos:
                candidatos.add(candidato)
                
        frecuentes_k = {}
        for cand in candidatos:
            cob = contar_cobertura(cand, transacciones)
            if cob >= cobertura_minima:
                frecuentes_k[cand] = cob
                
        if not frecuentes_k:
            break
            
        itemsets_frecuentes[k] = frecuentes_k
        print(f"\nNivel K={k}: {len(frecuentes_k)} itemsets cumplen la cobertura minima")
        for it, cob in sorted(frecuentes_k.items(), key=lambda x: x[1], reverse=True):
            sop = cob / total_transacciones
            print(f"   {set(it)} -> Cobertura: {cob}, Soporte: {sop:.4f}")
            
        k += 1
        
    print("\n" + "=" * 80)
    print("FASE 2: GENERACION Y SELECCION DE LAS MEJORES REGLAS DE ASOCIACION")
    print("=" * 80)
    
    reglas = []
    
    conteo_individual = {}
    for item in items:
        conteo_individual[item] = contar_cobertura({item}, transacciones)
        
    for nivel in range(2, k):
        for itemset, cob_regla in itemsets_frecuentes[nivel].items():
            for tam_consecuente in range(1, nivel):
                for cons in combinations(itemset, tam_consecuente):
                    consecuente = frozenset(cons)
                    antecedente = itemset - consecuente
                    
                    cob_antecedente = contar_cobertura(antecedente, transacciones)
                    if cob_antecedente == 0:
                        continue
                        
                    confianza = cob_regla / cob_antecedente
                    soporte = cob_regla / total_transacciones
                    
                    cob_consecuente = contar_cobertura(consecuente, transacciones)
                    prob_consecuente = cob_consecuente / total_transacciones
                    
                    lift = confianza / prob_consecuente if prob_consecuente > 0 else 0
                    
                    if confianza >= min_confianza:
                        if lift > 1.0:
                            utilidad = "Util (Correlacion positiva)"
                        elif lift == 1.0:
                            utilidad = "Independiente"
                        else:
                            utilidad = "No Util (Correlacion negativa)"
                            
                        reglas.append({
                            'antecedente': set(antecedente),
                            'consecuente': set(consecuente),
                            'cobertura': cob_regla,
                            'cobertura_ant': cob_antecedente,
                            'confianza': confianza,
                            'soporte': soporte,
                            'lift': lift,
                            'utilidad': utilidad
                        })
                        
    reglas_ordenadas = sorted(
        reglas,
        key=lambda r: (r['confianza'], r['cobertura'], r['soporte'], r['lift']),
        reverse=True
    )
    
    print(f"Total de reglas generadas que superan la confianza minima: {len(reglas_ordenadas)}")
    print("\nLISTADO DE LAS MEJORES REGLAS (Ordenadas por Confianza, Cobertura, Soporte y Lift):")
    print("-" * 80)
    
    for i, r in enumerate(reglas_ordenadas, 1):
        ant_str = "{" + ", ".join(sorted(r['antecedente'])) + "}"
        cons_str = "{" + ", ".join(sorted(r['consecuente'])) + "}"
        print(f"Regla {i:2d}: SI {ant_str} ENTONCES {cons_str}")
        print(f"         Confianza: {r['confianza']:.4f} ({r['confianza']*100:.2f}%) | "
              f"Cobertura: {r['cobertura']} | Soporte: {r['soporte']:.4f} | "
              f"Lift: {r['lift']:.4f} -> {r['utilidad']}")
        print("-" * 80)

if __name__ == '__main__':
    ejecutar_apriori('dataset_apriori_supermercado.csv', min_soporte=0.70, min_confianza=0.85)
