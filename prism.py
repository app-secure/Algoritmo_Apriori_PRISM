import csv

def cargar_datos(ruta_archivo):
    with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        atributos = [col for col in lector.fieldnames if col not in ('Proyecto_ID', 'Exito_Proyecto')]
        clase_objetivo = 'Exito_Proyecto'
        filas = list(lector)
    return filas, atributos, clase_objetivo

def evaluar_condiciones(instancias, regla_actual, atributos_disponibles, clase, valor_clase, n_total, prob_clase):
    mejor_condicion = None
    mejores_metricas = (-1, -1, -1, -1)
    
    for attr in atributos_disponibles:
        valores = sorted(list(set(d[attr] for d in instancias)))
        for val in valores:
            candidato = dict(regla_actual)
            candidato[attr] = val
            
            subconjunto = [d for d in instancias if all(d[k] == v for k, v in candidato.items())]
            cob_ant = len(subconjunto)
            if cob_ant == 0:
                continue
                
            aciertos = sum(1 for d in subconjunto if d[clase] == valor_clase)
            confianza = aciertos / cob_ant
            soporte = aciertos / n_total
            lift = confianza / prob_clase if prob_clase > 0 else 0
            
            metricas = (confianza, aciertos, soporte, lift)
            
            if metricas > mejores_metricas:
                mejores_metricas = metricas
                mejor_condicion = (attr, val)
                
    return mejor_condicion, mejores_metricas

def ejecutar_prism(ruta_archivo, min_confianza=1.00):
    datos_completos, atributos, clase_objetivo = cargar_datos(ruta_archivo)
    n_total = len(datos_completos)
    valores_clase = sorted(list(set(d[clase_objetivo] for d in datos_completos)))
    
    print("=" * 85)
    print("ALGORITMO PRISM - RECUBRIMIENTO SECUENCIAL (SEPARATE-AND-CONQUER)")
    print("=" * 85)
    print(f"Total de observaciones (N): {n_total}")
    print(f"Atributos predictivos: {', '.join(atributos)}")
    print(f"Variable de clase: {clase_objetivo} -> Clases: {valores_clase}")
    print("=" * 85)
    
    mejores_reglas_definitivas = []
    
    for val_clase in valores_clase:
        n_clase = sum(1 for d in datos_completos if d[clase_objetivo] == val_clase)
        prob_clase = n_clase / n_total
        
        print(f"\n>>> INDUCCION SECUENCIAL PARA LA CLASE: {clase_objetivo} = '{val_clase}'")
        print(f"    Total instancias de la clase: {n_clase}/{n_total} | Probabilidad marginal P({val_clase}) = {prob_clase:.4f}")
        print("-" * 85)
        
        datos_trabajo = list(datos_completos)
        num_regla = 1
        
        while any(d[clase_objetivo] == val_clase for d in datos_trabajo):
            regla_actual = {}
            atributos_libres = list(atributos)
            iteracion = 1
            casos_restantes_clase = sum(1 for d in datos_trabajo if d[clase_objetivo] == val_clase)
            
            print(f"\n   [Construccion de Regla Maestra {num_regla} - Casos de la clase pendientes por cubrir: {casos_restantes_clase}]")
            
            while atributos_libres:
                mejor_cond, metricas = evaluar_condiciones(
                    datos_trabajo, regla_actual, atributos_libres,
                    clase_objetivo, val_clase, n_total, prob_clase
                )
                
                if not mejor_cond:
                    break
                    
                attr_ganador, val_ganador = mejor_cond
                conf_g, cob_g, sop_g, lift_g = metricas
                
                print(f"      Paso {iteracion}: Se selecciona ({attr_ganador} = '{val_ganador}')")
                print(f"         Confianza: {conf_g:.4f} ({conf_g*100:.2f}%) | Cobertura: {cob_g} | Soporte: {sop_g:.4f} | Lift: {lift_g:.4f}")
                
                regla_actual[attr_ganador] = val_ganador
                atributos_libres.remove(attr_ganador)
                iteracion += 1
                
                if conf_g >= min_confianza:
                    print(f"         -> Se alcanzo pureza absoluta (Confianza = {conf_g:.4f}). Regla completada exitosamente.")
                    break
                    
            todos_ant = [d for d in datos_completos if all(d[k] == v for k, v in regla_actual.items())]
            todos_regla = [d for d in todos_ant if d[clase_objetivo] == val_clase]
            
            cob_global = len(todos_regla)
            cob_ant_global = len(todos_ant)
            conf_global = cob_global / cob_ant_global if cob_ant_global > 0 else 0
            sop_global = cob_global / n_total
            lift_global = conf_global / prob_clase if prob_clase > 0 else 0
            
            if conf_global < min_confianza or lift_global <= 1.0:
                print(f"\n   [Criterio de Parada Global]: Las siguientes reglas caen a confianza < {min_confianza*100:.0f}% o Lift <= 1.0.")
                print(f"   Se detiene el recubrimiento para evitar sobreajuste y reglas no utiles.")
                break
                
            info_regla = {
                'clase': val_clase,
                'antecedente': regla_actual,
                'confianza': conf_global,
                'cobertura': cob_global,
                'soporte': sop_global,
                'lift': lift_global,
                'utilidad': "Util (Correlacion positiva)" if lift_global > 1.0 else "No Util"
            }
            mejores_reglas_definitivas.append(info_regla)
            
            cubiertos_esta_regla = [
                d for d in datos_trabajo 
                if all(d[k] == v for k, v in regla_actual.items()) and d[clase_objetivo] == val_clase
            ]
            
            datos_trabajo = [d for d in datos_trabajo if d not in cubiertos_esta_regla]
            print(f"      -> Se eliminan los {len(cubiertos_esta_regla)} casos resueltos. Procediendo a los casos restantes.")
            num_regla += 1

    print("\n" + "=" * 85)
    print("CONSOLIDACION DEFINITIVA DE LAS MEJORES REGLAS DE CLASIFICACION (PRISM)")
    print("=" * 85)
    print(f"Total de reglas maestras puras obtenidas: {len(mejores_reglas_definitivas)}")
    print("-" * 85)
    
    for i, r in enumerate(mejores_reglas_definitivas, 1):
        ant_str = " Y ".join(f"{k} = '{v}'" for k, v in r['antecedente'].items())
        print(f"Regla {i:2d}: SI {ant_str} ENTONCES {clase_objetivo} = '{r['clase']}'")
        print(f"         Confianza: {r['confianza']:.4f} ({r['confianza']*100:.2f}%) | "
              f"Cobertura: {r['cobertura']} proyectos | Soporte: {r['soporte']:.4f} | "
              f"Lift: {r['lift']:.4f} -> {r['utilidad']}")
        print("-" * 85)

if __name__ == '__main__':
    ejecutar_prism('dataset_prism_proyectos_software.csv', min_confianza=1.00)
