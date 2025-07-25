from typing import Iterator, List, Any

def group_by_days(partition_data: Iterator[Any]) -> Iterator[List[Any]]:
    """
    Groups partition data by days based on hospitalization keys and dates.

    Parameters:
    partition_data (Iterator[Any]): An iterator of rows containing hospitalization data.

    Yields:
    Iterator[List[Any]]: A list containing grouped data for each hospitalization period.
    """
    internation_flag = False
    last_hospitalization_date = ''
    current_hospitalization_key = ''
    start_date = ''
    for row in partition_data:
        if internation_flag is False:
            if row.CHAVE_INTERNACAO_AON is None:
                continue

            internation_flag = True
            current_hospitalization_key = row.CHAVE_INTERNACAO_AON
            start_date = row.DT_ATND

        if row.CHAVE_INTERNACAO_AON is not None:
            last_hospitalization_date = row.DT_ATND

        if row.NEXT_DAY is None or abs((last_hospitalization_date - row.DT_ATND).days) > 10:
            internation_flag = False
            end_date = last_hospitalization_date
            yield [
                row.CD_APLC,
                row.CD_EMPR_GRPO,
                row.CODIGO_BENEFICIARIO,
                current_hospitalization_key,
                start_date,
                end_date
            ]

#========================================================================================================

def group_hospitalizations_by_days(df: DataFrame) -> DataFrame:
    """
    Agrupa internações por beneficiário, considerando intervalo entre atendimentos maior que 10 dias
    ou término de sequência (NEXT_DAY nulo). Cria chave agrupadora.
    """
    window_spec = Window.partitionBy("CODIGO_BENEFICIARIO").orderBy("DT_ATND")
    df = df.withColumn("DT_ATND_PREV", F.lag("DT_ATND").over(window_spec)) \
           .withColumn("DIFF_DIAS", when(col("DT_ATND_PREV").isNotNull(),
                                         F.datediff(col("DT_ATND"), col("DT_ATND_PREV")))
                       .otherwise(lit(0))) \
           .withColumn("is_new_group", when(
               (col("CHAVE_INTERNACAO_AON").isNotNull()) &
               ((col("DIFF_DIAS") > 10) | col("NEXT_DAY").isNull()),
               1
           ).otherwise(0)) \
        .withColumn("group_id", F.sum("is_new_group").over(window_spec))
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


