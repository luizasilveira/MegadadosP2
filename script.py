import pyspark
import math
import pandas as pd


def conta_palavras(item):
    url, conteudo = item
    palavras = conteudo.strip().split()
    return [(palavra.lower(), 1) for palavra in palavras]


def conta_documentos(item):
    url, conteudo = item
    palavras = conteudo.strip().split()
    return [(palavra.lower(), 1) for palavra in set(palavras)]


def junta_contagens(nova_contagem, contagem_atual):
    return nova_contagem + contagem_atual


def filtra_doc_freq(item):
    contagem = item[1]
    return (contagem < doc_max) and (contagem >= doc_min)


def computa_idf(item):
    palavra, contagem = item
    idf = math.log10(N_docs / contagem)
    return (palavra, idf)


def computa_freq(item):
    palavra, contagem = item
    freq = math.log10(1 + contagem)
    return (palavra, freq)


def computa_rel(item):
    palavra, contagem = item
    freq, idf = contagem
    relevancia = freq*idf
    return (palavra, relevancia)


if __name__ == '__main__':

    sc = pyspark.SparkContext(appName="Projeto2")
    rdd = sc.sequenceFile("s3://megadados-alunos/web-brasil")

    N_docs = rdd.count()

    # rdds específicos
    rdd_ip = rdd.filter(lambda x: "iphone" in x[1] and not "android" in x[1])
    rdd_an = rdd.filter(lambda x: "android" in x[1] and not "iphone" in x[1])
    rdd_in = rdd.filter(lambda x: "iphone" in x[1] and "android" in x[1])

    # conta palavras
    rdd_iphone = rdd_ip.flatMap(conta_palavras).reduceByKey(junta_contagens)
    rdd_android = rdd_an.flatMap(conta_palavras).reduceByKey(junta_contagens)
    rdd_inter = rdd_in.flatMap(conta_palavras).reduceByKey(junta_contagens)

    # conta documentos
    rdd_iphone_docs = rdd_ip.flatMap(
        conta_documentos).reduceByKey(junta_contagens)
    rdd_android_docs = rdd_an.flatMap(
        conta_documentos).reduceByKey(junta_contagens)
    rdd_inter_docs = rdd_in.flatMap(
        conta_documentos).reduceByKey(junta_contagens)

    doc_min = 10
    doc_max = 0.70 * N_docs

    # filtra documentos
    rdd_iphone_docs_filtrado = rdd_iphone_docs.filter(filtra_doc_freq)
    rdd_android_docs_filtrado = rdd_android_docs.filter(filtra_doc_freq)
    rdd_inter_docs_filtrado = rdd_inter_docs.filter(filtra_doc_freq)

    # computa frequências
    rdd_iphone_freq = rdd_iphone.map(computa_freq)
    rdd_android_freq = rdd_android.map(computa_freq)
    rdd_inter_freq = rdd_inter.map(computa_freq)

    # computa idfs
    rdd_iphone_idf = rdd_iphone_docs_filtrado.map(computa_idf)
    rdd_android_idf = rdd_android_docs_filtrado.map(computa_idf)
    rdd_inter_idf = rdd_inter_docs_filtrado.map(computa_idf)

    # computa relevâncias
    # intersecção
    rdd_inter_join = rdd_inter_freq.join(rdd_inter_idf)
    rdd_inter_rel = rdd_inter_join.map(computa_rel)
    list_inter_rel = rdd_inter_rel.takeOrdered(100, key=lambda x: -x[1])

    # iphone
    rdd_iphone_join = rdd_iphone_freq.join(rdd_iphone_idf)
    rdd_iphone_rel = rdd_iphone_join.map(computa_rel)
    list_iphone_rel = rdd_iphone_rel.takeOrdered(100, key=lambda x: -x[1])

    # android
    rdd_android_join = rdd_android_freq.join(rdd_android_idf)
    rdd_android_rel = rdd_android_join.map(computa_rel)
    list_android_rel = rdd_android_rel.takeOrdered(100, key=lambda x: -x[1])

    # csvs
    inter_df = pd.DataFrame(list_inter_rel, columns=["palavra", "relevancia"])
    inter_csv = inter_df.to_csv("inter.csv", index=False)
    iphone_df = pd.DataFrame(list_iphone_rel, columns=[
                             "palavra", "relevancia"])
    iphone_csv = iphone_df.to_csv("iphone.csv", index=False)
    android_df = pd.DataFrame(list_android_rel, columns=[
                              "palavra", "relevancia"])
    android_csv = android_df.to_csv("android.csv", index=False)
