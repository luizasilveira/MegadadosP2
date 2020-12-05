import pyspark
import math
import pandas as pd


def conta_palavras(item):
    url, conteudo = item
    palavras = conteudo.strip().split()
    palavras_ = [p for p in palavras if p.isalpha()]
    return [(palavra.lower(), 1) for palavra in palavras_]


def conta_documentos(item):
    url, conteudo = item
    palavras = conteudo.strip().split()
    palavras_ = [p for p in palavras if p.isalpha()]
    return [(palavra.lower(), 1) for palavra in set(palavras_)]


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


sc = pyspark.SparkContext(appName="Projeto2")
rdd = sc.sequenceFile("s3://megadados-alunos/web-brasil")

N_docs = rdd.count()

# rdds específicos
rdd_ni = rdd.filter(lambda x: "nike" in x[1] and not "adidas" in x[1])
rdd_ad = rdd.filter(lambda x: "adidas" in x[1] and not "nike" in x[1])
rdd_in = rdd.filter(lambda x: "nike" in x[1] and "adidas" in x[1])

# conta palavras
rdd_nike = rdd_ni.flatMap(conta_palavras).reduceByKey(junta_contagens)
rdd_adidas = rdd_ad.flatMap(conta_palavras).reduceByKey(junta_contagens)
rdd_inter = rdd_in.flatMap(conta_palavras).reduceByKey(junta_contagens)

# conta documentos
rdd_nike_docs = rdd_ni.flatMap(
    conta_documentos).reduceByKey(junta_contagens)
rdd_adidas_docs = rdd_ad.flatMap(
    conta_documentos).reduceByKey(junta_contagens)
rdd_inter_docs = rdd_in.flatMap(
    conta_documentos).reduceByKey(junta_contagens)

doc_min = 15
doc_max = 0.60 * N_docs

# filtra documentos
rdd_nike_docs_filtrado = rdd_nike_docs.filter(filtra_doc_freq)
rdd_adidas_docs_filtrado = rdd_adidas_docs.filter(filtra_doc_freq)
rdd_inter_docs_filtrado = rdd_inter_docs.filter(filtra_doc_freq)

# computa frequências
rdd_nike_freq = rdd_nike.map(computa_freq)
rdd_adidas_freq = rdd_adidas.map(computa_freq)
rdd_inter_freq = rdd_inter.map(computa_freq)

# computa idfs
rdd_nike_idf = rdd_nike_docs_filtrado.map(computa_idf)
rdd_adidas_idf = rdd_adidas_docs_filtrado.map(computa_idf)
rdd_inter_idf = rdd_inter_docs_filtrado.map(computa_idf)

# computa relevâncias
# intersecção
rdd_inter_join = rdd_inter_freq.join(rdd_inter_idf)
rdd_inter_rel = rdd_inter_join.map(computa_rel)
list_inter_rel = rdd_inter_rel.takeOrdered(100, key=lambda x: -x[1])

# nike
rdd_nike_join = rdd_nike_freq.join(rdd_nike_idf)
rdd_nike_rel = rdd_nike_join.map(computa_rel)
list_nike_rel = rdd_nike_rel.takeOrdered(100, key=lambda x: -x[1])

# adidas
rdd_adidas_join = rdd_adidas_freq.join(rdd_adidas_idf)
rdd_adidas_rel = rdd_adidas_join.map(computa_rel)
list_adidas_rel = rdd_adidas_rel.takeOrdered(100, key=lambda x: -x[1])


inter_df = pd.DataFrame(list_inter_rel, columns=["palavra", "relevancia"])
inter_csv = inter_df.to_csv(
    "s3://megadados-alunos/gabriela-luiza/inter.csv", index=False)
nike_df = pd.DataFrame(list_nike_rel, columns=[
    "palavra", "relevancia"])
nike_csv = nike_df.to_csv(
    "s3://megadados-alunos/gabriela-luiza/nike.csv", index=False)
adidas_df = pd.DataFrame(list_adidas_rel, columns=[
    "palavra", "relevancia"])
adidas_csv = adidas_df.to_csv(
    "s3://megadados-alunos/gabriela-luiza/adidas.csv", index=False)
