from modules.utils.connectors import SnowflakeConnector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, lit, col, expr, when, floor, date_sub
from pyspark.sql.window import Window
import modules.gold.fastview.hospitalization_business_rules as hospitalization
import pyspark.sql.functions as F
from pyspark import StorageLevel
import re
from modules.gold.fastview.hospitalization_rules_udf import group_by_days
spark = SparkSession.builder \
    .appName("AppName") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
conn = SnowflakeConnector()


def apply_translate_srvc(table: DataFrame, column_name: str) -> DataFrame:
    """
    Applies a translation service to a specified column in a Spark DataFrame. The function translates
    specific characters to a predefined set of characters, removes certain blacklisted words, and 
    trims and cleans up the resulting text. 
    Args:
        table (DataFrame): The input Spark DataFrame.
        column_name (str): The name of the column to be translated and cleaned. 
    Returns:
        DataFrame: A new DataFrame with the modified column.
    """
    original_chars = '1234567890AÁÃÂBCÇDEÉÊFGHIÍJKLMNÖOÔÕÓÒPQRSTUÜÚWXÝYV‡Æä¢¡£.,()-+/*$?"=%º:;_#|àÀ¹¶µ€“~…  ‚[]{}Øƒª'
    translated_chars = '          AAAABCCDEEEFGHIIJKLMNOOOOOOPQRSTUUUWXYYVCAOOIU                   AA  ACA O          '

    blacklist = ["DO", "DA", "DE", "X", "G", "MG", "ML", "SOL", "INJ", "CX", "CM", "CT", "INC",
                 "AMPOLA", "E", "EM", "OU", "MT", "MTS", "COM", "COMPRIMIDO", "HT", "MM", "POR",
                 "PARA", "FR", "P", "AMP", "CMX", "VD", "AMP."]
    pattern = r'\b(?:' + '|'.join(blacklist) + r')\b'

    col_mod = F.translate(F.upper(col(column_name)),
                          original_chars, translated_chars)
    col_mod = F.regexp_replace(col_mod, pattern, "")
    col_mod = F.regexp_replace(col_mod, r'\s+', ' ')
    col_mod = F.trim(col_mod)

    return table.withColumn(f"{column_name}_MOD", col_mod)


def filter_last_two_years(table: DataFrame) -> DataFrame:
    """
    Filters the input DataFrame to include only the rows from the last two years for each unique 'CD_EMPR'.

    Args:
        table (DataFrame): Input DataFrame containing at least 'CD_EMPR' and 'DT_REFR' columns.

    Returns:
        DataFrame: Filtered DataFrame with rows from the last two years for each 'CD_EMPR'.
    """
    table = table.withColumn("DT_REFR", col("DT_REFR").cast("date"))
    max_dates = table.groupBy("CD_EMPR").agg(
        F.max(col("DT_REFR")).alias("max_DT_REFR"))
    max_dates = max_dates.withColumn(
        "start_date", date_sub(col("max_DT_REFR"), 730))
    result_df = table.join(max_dates, on="CD_EMPR").filter((col("DT_REFR") >= col(
        "start_date")) & (col("DT_REFR") <= col("max_DT_REFR"))).drop("max_DT_REFR", "start_date")
    return result_df

def group_hospitalizations_by_days(df: DataFrame) -> DataFrame:
    """
    Agrupa internações hospitalares por chave de internação e período, com separação
    caso a diferença entre atendimentos seja maior que 10 dias ou o próximo atendimento seja nulo.
    Parâmetros:
        df (DataFrame): DataFrame do Spark com colunas:
            - CHAVE_INTERNACAO_AON
            - CODIGO_BENEFICIARIO
            - DT_ATND
            - NEXT_DAY
            - CD_APLC
            - CD_EMPR_GRPO
    Retorna:
        DataFrame com uma linha por internação detectada.
    """
    # Janela para ordenar por paciente e data
    window_spec = Window.partitionBy("CODIGO_BENEFICIARIO").orderBy("DT_ATND")
    
    # Selecionar e calcular as colunas necessárias
    df = df.select(
        "*",
        F.lag("DT_ATND").over(window_spec).alias("DT_ATND_PREV"),
        when(col("DT_ATND_PREV").isNotNull(), F.datediff(col("DT_ATND"), col("DT_ATND_PREV")))
        .otherwise(lit(0)).alias("DIFF_DIAS"),
        when(
            (col("CHAVE_INTERNACAO_AON").isNotNull()) &
            ((col("DIFF_DIAS") > 10) | col("NEXT_DAY").isNull()),
            1
        ).otherwise(0).alias("is_new_group")
    )
    
    # Criar um ID de grupo com soma acumulada
    df = df.select(
        "*",
        F.sum("is_new_group").over(window_spec).alias("group_id")
    )
    
    # Agrupar por grupo de internação e pegar os campos desejados
    df_grouped = df.groupBy("CODIGO_BENEFICIARIO", "group_id").agg(
        F.first("CD_APLC").alias("CD_APLC"),
        F.first("CD_EMPR_GRPO").alias("CD_EMPR_GRPO"),
        F.first("CHAVE_INTERNACAO_AON").alias("CHAVE_INTERNACAO_AON"),
        F.min("DT_ATND").alias("start_date"),
        F.max("DT_ATND").alias("end_date")
    )
    
    return df_grouped.select(
        col("CD_APLC").alias("UDF_CD_APLC"),
        col("CD_EMPR_GRPO").alias("UDF_CD_EMPR_GRPO"),
        col("CODIGO_BENEFICIARIO").alias("UDF_CODIGO_BENEFICIARIO"),
        col("CHAVE_INTERNACAO_AON").alias("UDF_INTERNACAO_AON"),
        col("start_date").alias("INICIO_INTERNACAO"),
        col("end_date").alias("FIM_INTERNACAO")
    )

def generate_hospitalization_key(table: DataFrame) -> DataFrame:
    """
    Generates a hospitalization key for each record in the given table.

    Args:
        table (DataFrame): Input DataFrame containing hospitalization data.

    Returns:
        DataFrame: A DataFrame with additional columns for hospitalization key, 
                   total hospitalization cost, event name, and start of hospitalization.
    """
    df_key = table.filter(F.coalesce(col("FL_INTD"), lit("")) == 'S')\
                  .select("CD_APLC", "CD_EMPR_GRPO", "CODIGO_BENEFICIARIO", "DT_ATND")\
                  .withColumn("CHAVE_INTERNACAO_AON", F.concat(lit("INT"), col("CD_APLC"), col("CD_EMPR_GRPO"),
                                                               col("CODIGO_BENEFICIARIO"), col("DT_ATND"))).distinct()

    window = Window.partitionBy(
        "CD_APLC", "CD_EMPR_GRPO", "CODIGO_BENEFICIARIO").orderBy("DT_ATND")
    df_key = df_key.withColumn("NEXT_DAY", F.lead("DT_ATND").over(window))
    #hosp_keys = df_key.repartition("CHAVE_INTERNACAO_AON").rdd.mapPartitions(group_by_days).toDF([
    #    "UDF_CD_APLC", "UDF_CD_EMPR_GRPO", "UDF_CODIGO_BENEFICIARIO",
    #    "UDF_INTERNACAO_AON", "INICIO_INTERNACAO", "FIM_INTERNACAO"
    #]).persist()

    hosp_keys = group_hospitalizations_by_days(df_key.repartition("CHAVE_INTERNACAO_AON")).persist()

    join_cond = (
        (table["CD_APLC"] == hosp_keys["UDF_CD_APLC"]) &
        (table["CD_EMPR_GRPO"] == hosp_keys["UDF_CD_EMPR_GRPO"]) &
        (table["CODIGO_BENEFICIARIO"] == hosp_keys["UDF_CODIGO_BENEFICIARIO"]) &
        (table["DT_ATND"] >= hosp_keys["INICIO_INTERNACAO"]) &
        (table["DT_ATND"] <= hosp_keys["FIM_INTERNACAO"])
    )
    df_joined = table.join(F.broadcast(hosp_keys), join_cond, "left")
    window_sum = Window.partitionBy("UDF_INTERNACAO_AON")
    df_joined = df_joined.withColumn(
        "CUSTO_TOTAL_INTERNACAO", F.sum("VL_PAGO_EVNT").over(window_sum))
    df_joined = df_joined.withColumn(
        "CHAVE_INTERNACAO_AON",
        when(col("CUSTO_TOTAL_INTERNACAO") > 500, col("UDF_INTERNACAO_AON"))
    ).withColumn(
        "NOME_EVENTO",
        when(col("CHAVE_INTERNACAO_AON").isNotNull(), lit("INTERNACAO"))
    )

    return df_joined.drop("UDF_CD_APLC", "UDF_CD_EMPR_GRPO", "UDF_CODIGO_BENEFICIARIO", "UDF_INTERNACAO_AON")


def filter_sas_claims(table: DataFrame) -> DataFrame:
    """
    Retrieve SAS claims data for a given carrier from Snowflake, filter it to include only the last two years,
    and return the resulting DataFrame.

    Args:
        carrier (str): The carrier identifier.

    Returns:
        DataFrame: A DataFrame containing the filtered SAS claims data.
    """
    companies_table = conn.get_table_from_snowflake('TABELA_EMPRESAS', 'GOLD').select("CODIGO_OPERADORA", "CODIGO_EMPRESA").distinct()\
        .withColumnRenamed("CODIGO_OPERADORA", "CD_OPRD") \
        .withColumnRenamed("CODIGO_EMPRESA", "CD_EMPR")
    table = table.withColumn("CODIGO_BENEFICIARIO",
                             F.concat(col("CD_TTLR"), col("CD_DPND")))
    return table.join(F.broadcast(companies_table), ["CD_OPRD", "CD_EMPR"], "inner")


def classify_events(
    table: DataFrame,
    classifications: DataFrame,
    code_name: str = "CD_SRVC",
    description_name: str = "DS_SRVC"
) -> DataFrame:
    """
    Classify events in the given table based on the classifications DataFrame.

    Parameters:
    table (DataFrame): The input DataFrame containing events to be classified.
    classifications (DataFrame): The DataFrame containing classification information.
    code_name (str): The column name in the table to join on for code. Default is "CD_SRVC".
    description_name (str): The column name in the table to join on for description. Default is "DS_SRVC".

    Returns:
    DataFrame: The resulting DataFrame with classified events.
    """

    table = apply_translate_srvc(table, description_name)
    classifications = apply_translate_srvc(
        classifications, "DS_SRVC_OPERADORA")

    columns = [col for col in classifications.columns if col != 'CD_SRVC']
    classifications = classifications.withColumn("CD_SRVC", when(col("CD_SRVC") != '0', F.regexp_replace(col("CD_SRVC"), "^0+", "")).otherwise(col("CD_SRVC")))\
                                     .withColumn("priority", when((col("COD_TUSS").isNotNull()) & (col("PROC_PRINC").isNotNull()), 1).otherwise(0))\
                                     .orderBy(F.desc("priority")).dropDuplicates(["CD_SRVC", "DS_SRVC_OPERADORA_MOD"]).drop("priority").persist()

    columns = [col for col in table.columns if col != code_name]
    table = table.withColumn(code_name, when(col(code_name) != '0', F.regexp_replace(
        col(code_name), "^0+", "")).otherwise(col(code_name)))
    join_cond = (
        (table[code_name] == classifications["CD_SRVC"]) &
        (table[f"{description_name}_MOD"] ==
         classifications["DS_SRVC_OPERADORA_MOD"])
    )

    columns = [
        "CD_SRVC",
        "DS_SRVC_OPERADORA",
        "DS_SRVC_OPERADORA_MOD",
        "CATEGORIA",
        "COD_TUSS",
        "PROC_PRINC",
        "COD_CBHPM",
        "DS_SRVC_2",
        "TIPO_INTERNACAO",
        "CRONICO",
        "REGIME_DIARIA",
        "NM_EVNT2",
        "GRUPO"
    ]

    table = table.join(F.broadcast(classifications.select(columns)), join_cond, "left")\
                 .drop(classifications['CD_SRVC'], classifications['DS_SRVC_OPERADORA_MOD'], table[f'{description_name}_MOD'])

    columns = [col for col in table.columns if col not in {
        'NOME_EVENTO', 'CHAVE_INTERNACAO_AON'}]
    table = table.withColumn(
        'NOME_EVENTO',
        F.when(table.NM_EVNT2 == 'CONSULTAELETIVA', F.lit(None))
        .when(table.NM_EVNT2 == 'CONSULTAPS', F.lit(None))
        .when(table.NOME_EVENTO.isNotNull(), table.NOME_EVENTO)
        .otherwise(F.lit(None)))\
        .withColumn('CHAVE_INTERNACAO_AON', F.when(table.NM_EVNT2 == 'CONSULTAELETIVA', F.lit(None))
                    .when(table.NM_EVNT2 == 'CONSULTAPS', F.lit(None))
                    .when(
            F.col('NOME_EVENTO').isNotNull(),
            F.col('CHAVE_INTERNACAO_AON')
        )
            .otherwise(F.lit(None)))

    return table


def define_main_procedures(table: DataFrame) -> DataFrame:
    w_intern = Window.partitionBy(
        "CHAVE_INTERNACAO_AON").orderBy(F.desc("VL_PAGO_EVNT"))
    intern_proc = table.filter("PROC_PRINC = 1 AND NOME_EVENTO = 'INTERNACAO'")\
                       .withColumn("rank", F.row_number().over(w_intern))\
                       .filter("rank = 1")\
                       .select("CHAVE_INTERNACAO_AON", col("COD_TUSS").alias("PROC_INT"), col("TIPO_INTERNACAO"))

    w_terap = Window.partitionBy(
        "CHAVE_TERAPIA_COMPLEXA_AON").orderBy(F.desc("VL_PAGO_EVNT"))
    terap_proc = table.filter("PROC_PRINC = 1 AND NOME_EVENTO = 'TERAPIACOMPLEXA'")\
                      .withColumn("rank", F.row_number().over(w_terap))\
                      .filter("rank = 1")\
                      .select("CHAVE_TERAPIA_COMPLEXA_AON", col("COD_TUSS").alias("PROC_TERP"))

    table = table.drop("TIPO_INTERNACAO").join(intern_proc, "CHAVE_INTERNACAO_AON", "left").join(
        terap_proc, "CHAVE_TERAPIA_COMPLEXA_AON", "left")

    return table.withColumn(
        "PROC_PRINC", when(col("NOME_EVENTO") == "INTERNACAO",
                           when(col("PROC_INT").isNotNull(), col("PROC_INT")).otherwise("Internação clínica"))
        .when(col("NOME_EVENTO") == "TERAPIACOMPLEXA", col("PROC_TERP"))
    ).drop("PROC_INT", "PROC_TERP")


def define_entrance_type(table: DataFrame, beneficiary_id: str = "CODIGO_BENEFICIARIO") -> DataFrame:
    """
    Define the type of entrance for hospitalization in the given table.

    Args:
        table (DataFrame): The input Spark DataFrame containing hospitalization data.
        beneficiary_id (str, optional): The column name for the beneficiary ID. Defaults to "CODIGO_BENEFICIARIO".

    Returns:
        DataFrame: A Spark DataFrame with an additional column 'TIPO_ENTRADA_INTERNACAO' indicating the type of entrance.
    """
    hospitalization_entrance = table \
        .filter("NM_EVNT2 = 'CONSULTAPS'") \
        .select(
            F.col("DT_ATND").alias("INICIO_INTERNACAO"),
            F.col(beneficiary_id),
            F.lit("URGENCIA").alias("TIPO_ENTRADA_INTERNACAO")
        ) \
        .distinct()
    
    hospitalization_by_start_date = table \
        .filter(F.col("CHAVE_INTERNACAO_AON").isNotNull()) \
        .select(beneficiary_id, "INICIO_INTERNACAO", "CHAVE_INTERNACAO_AON") \
        .join(
            hospitalization_entrance,
            [beneficiary_id, "INICIO_INTERNACAO"]
        ) \
        .select("CHAVE_INTERNACAO_AON", "TIPO_ENTRADA_INTERNACAO") \
        .distinct()

    columns = table.columns
    table = table \
        .join(
            hospitalization_by_start_date,
            "CHAVE_INTERNACAO_AON",
            "LEFT"
        ) \
        .select(
            *columns, 
            F.when(
                    F.col('TIPO_ENTRADA_INTERNACAO').isNotNull(),
                    F.col('TIPO_ENTRADA_INTERNACAO')
                ) \
                .when(
                    F.col('CHAVE_INTERNACAO_AON').isNotNull(),
                    F.lit('ELETIVA')
                ) \
                .otherwise(F.lit(None)).alias('TIPO_ENTRADA_INTERNACAO')
        )  

    return table



def fill_complex_therapy(df: DataFrame, beneficiary_id: str = "CODIGO_BENEFICIARIO") -> DataFrame:
    chave_cols = ['CD_APLC', 'CD_EMPR_GRPO', beneficiary_id, 'DT_ATND']
    df_validated = df.filter(col("CHAVE_TERAPIA_COMPLEXA_AON").isNotNull()) \
        .select(*chave_cols,
                col("CHAVE_TERAPIA_COMPLEXA_AON").alias(
                    "nova_chave_terapia"),
                col("GRUPO").alias("novo_grupo"),
                lit("TERAPIACOMPLEXA").alias("novo_nome_evento"),
                col("PROC_PRINC").alias("novo_proc_princ")
                ) \
        .dropDuplicates(chave_cols)

    df_joined = df.join(df_validated, on=chave_cols, how='left')

    df_final = df_joined.withColumn(
        "CHAVE_TERAPIA_COMPLEXA_AON",
        when(
            (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "INTERNACAO", "CONSULTAPS"])) & (
                col("nova_chave_terapia").isNotNull()),
            col("nova_chave_terapia")
        ).otherwise(col("CHAVE_TERAPIA_COMPLEXA_AON")))\
        .withColumn("GRUPO", when(
            (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "INTERNACAO", "CONSULTAPS"])) & (
                col("novo_grupo").isNotNull()),
            col("novo_grupo")
        ).otherwise(col("GRUPO")))\
        .withColumn("NOME_EVENTO", when(
            (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "INTERNACAO", "CONSULTAPS"])) & (
                col("nova_chave_terapia").isNotNull()),
            col("novo_nome_evento")
        ).otherwise(col("NOME_EVENTO")))\
        .withColumn("PROC_PRINC", when(
            (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "INTERNACAO", "CONSULTAPS"])) & (
                col("nova_chave_terapia").isNotNull()),
            col("novo_proc_princ")
        ).otherwise(col("PROC_PRINC"))).drop("nova_chave_terapia","novo_grupo","novo_nome_evento","novo_proc_princ")
    return df_final


def fill_emergency(df: DataFrame, beneficiary_id: str = "CODIGO_BENEFICIARIO") -> DataFrame:
    chave_cols = ['CD_APLC', 'CD_EMPR_GRPO', beneficiary_id, 'DT_ATND']
    df_validated = df.filter(col("CHAVE_PRONTO_SOCORRO_AON").isNotNull()) \
        .select(*chave_cols,
                col("CHAVE_PRONTO_SOCORRO_AON").alias(
                    "NOVA_CHAVE_PRONTO_SOCORRO"),
                lit("ATENDIMENTO P.S").alias("novo_nome_evento")) \
        .dropDuplicates(chave_cols)

    df_joined = df.join(df_validated, on=chave_cols, how='left')

    df_final = df_joined.withColumn("CHAVE_PRONTO_SOCORRO_AON",
                                    when(
                                        (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "TERAPIASIMPLES", "INTERNACAO", "TERAPIACOMPLEXA"])) & (
                                            col("NOVA_CHAVE_PRONTO_SOCORRO").isNotNull()),
                                        col("NOVA_CHAVE_PRONTO_SOCORRO")
                                    ).otherwise(col("CHAVE_PRONTO_SOCORRO_AON")))\
        .withColumn("NOME_EVENTO", when(
            (~col("NOME_EVENTO").isin(["CONSULTAELETIVA", "TERAPIASIMPLES", "INTERNACAO", "TERAPIACOMPLEXA", "CONSULTAPS"])) & (
                col("NOVA_CHAVE_PRONTO_SOCORRO").isNotNull()),
            col("novo_nome_evento")
        ).otherwise(col("NOME_EVENTO"))).drop("NOVA_CHAVE_PRONTO_SOCORRO","novo_nome_evento")
    return df_final


def generate_complex_treatment_key(table: DataFrame, beneficiary_id: str = "CODIGO_BENEFICIARIO") -> DataFrame:
    """
    Generates a complex treatment key for the given table and updates the table with this key.

    Args:
        table (DataFrame): The input DataFrame containing treatment data.
        beneficiary_id (str): The column name for the beneficiary ID. Default is "CODIGO_BENEFICIARIO".

    Returns:
        DataFrame: The updated DataFrame with the complex treatment key and modified columns.
    """
    complex_treatment = table \
        .filter("NM_EVNT2 = 'TERAPIACOMPLEXA' AND NOME_EVENTO IS NULL")\
        .select(
            "CD_APLC",
            "CD_EMPR_GRPO",
            beneficiary_id,
            "DT_ATND",
            "NM_EVNT2",
            F.concat(
                F.lit('TC'),
                F.col("CD_APLC"),
                F.col("CD_EMPR_GRPO"),
                F.col(beneficiary_id),
                F.col("DT_ATND")
            ).alias('CHAVE_TERAPIA_COMPLEXA_AON')
        ) \
        .dropDuplicates()

    table = table \
        .join(
            complex_treatment,
            [
                "CD_APLC",
                "CD_EMPR_GRPO",
                beneficiary_id,
                "DT_ATND",
                "NM_EVNT2"
            ],
            "LEFT"
        ) \
        .withColumn(
            "NOME_EVENTO",
            F.when(
                F.col("CHAVE_TERAPIA_COMPLEXA_AON").isNotNull(),
                F.lit('TERAPIACOMPLEXA')
            )
            .when(
                F.col("NOME_EVENTO").isNotNull(),
                F.col("NOME_EVENTO")
            )
            .otherwise(F.lit(None))
        )\
        .withColumn(
            'CHAVE_INTERNACAO_AON',
            F.when(
                F.col("CHAVE_TERAPIA_COMPLEXA_AON").isNotNull(),
                F.lit(None)
            )
            .otherwise(
                F.col("CHAVE_INTERNACAO_AON")
            )
        )

    return table


def generate_emergency_care_key(table: DataFrame, beneficiary_id: str = "CODIGO_BENEFICIARIO") -> DataFrame:
    """
    Generates a key for emergency care events and updates the table with this key and event names.

    Args:
        table (DataFrame): The input Spark DataFrame containing healthcare event data.
        beneficiary_id (str, optional): The column name for the beneficiary ID. Defaults to "CODIGO_BENEFICIARIO".

    Returns:
        DataFrame: The updated DataFrame with the emergency care key and event names.
    """
    columns = table.columns
    emergency = table \
        .filter("NM_EVNT2 = 'CONSULTAPS'")\
        .select(
            "CD_APLC",
            "CD_EMPR_GRPO",
            beneficiary_id,
            "DT_ATND",
            "NM_EVNT2",
            F.concat(
                F.lit('PS'),
                F.col("CD_APLC"),
                F.col("CD_EMPR_GRPO"),
                F.col(beneficiary_id),
                F.col("DT_ATND")
            ).alias('CHAVE_PRONTO_SOCORRO_AON')
        ) \
        .distinct()

    columns = [col for col in table.columns if col not in {
        'NOME_EVENTO', 'CHAVE_INTERNACAO_AON', 'CHAVE_TERAPIA_COMPLEXA_AON'}]
    table = table \
        .join(
            emergency,
            [
                "CD_APLC",
                "CD_EMPR_GRPO",
                beneficiary_id,
                "DT_ATND",
                "NM_EVNT2"
            ],
            "LEFT"
        ) \
        .withColumn("NOME_EVENTO",
                F.when(
                    F.col("CHAVE_PRONTO_SOCORRO_AON").isNotNull(),
                    F.lit('CONSULTAPS')
                )
                .when(
                    F.col("NOME_EVENTO").isNotNull(),
                    F.col("NOME_EVENTO")
                ))\
        .withColumn('CHAVE_INTERNACAO_AON',
                F.when(
                    F.col("CHAVE_PRONTO_SOCORRO_AON").isNotNull(),
                    F.lit(None)
                )
                .otherwise(
                    F.col("CHAVE_INTERNACAO_AON")
                ))\
        .withColumn('CHAVE_TERAPIA_COMPLEXA_AON',
                F.when(
                    F.col("CHAVE_PRONTO_SOCORRO_AON").isNotNull(),
                    F.lit(None)
                )
                .otherwise(
                    F.col("CHAVE_TERAPIA_COMPLEXA_AON")
                )
                )

    return table


def determine_event(table: DataFrame) -> DataFrame:
    """
    Determines the event name for each row in the given DataFrame.

    Parameters:
    table (DataFrame): The input DataFrame containing event data.

    Returns:
    DataFrame: A DataFrame with the determined event names.
    """
    table = table.withColumn("NOME_EVENTO",
                             F.when(
                                 F.col("NOME_EVENTO").isNotNull(),
                                 F.col("NOME_EVENTO")
                             )
                             .when(
                                 F.col("NOME_EVENTO").isNull()
                                 & F.col("NM_EVNT2").isNotNull(),
                                 F.col("NM_EVNT2")
                             )
                             .otherwise(F.lit("AMBULATORIO"))
                             )
    return table


def apply_claims_hospitalization_rules(table: DataFrame) -> DataFrame:
    """
    Apply a series of hospitalization rules to a claims table.

    This function performs the following steps:
    1. Retrieves classifications from a Snowflake table.
    2. Filter SAS claims.
    3. Generates a hospitalization key.
    4. Classifies events based on the retrieved classifications.
    5. Defines main procedures.
    6. Defines the entrance type.
    7. Generates a complex treatment key.
    8. Generates an emergency care key.
    9. Applies service translation.

    Args:
        table (pd.DataFrame): The input claims table.

    Returns:
        pd.DataFrame: The processed claims table with hospitalization rules applied.
    """
    classifications = conn.get_table_from_snowflake(
        'AUXILIAR_PROCEDIMENTOS', 'GOLD')
    table = filter_sas_claims(table)
    table = generate_hospitalization_key(table)
    table = classify_events(table, classifications)
    table = define_entrance_type(table)
    table = generate_complex_treatment_key(table)
    table = define_main_procedures(table)
    table = generate_emergency_care_key(table)
    table = determine_event(table)
    table = fill_complex_therapy(table)
    table = fill_emergency(table)
    return table
