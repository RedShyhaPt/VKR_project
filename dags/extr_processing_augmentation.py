from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/home/azat/airflow/dags/programs')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2022, 10, 30),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    'end_date': datetime(2023, 1, 1),
}

dag = DAG("extr_proc_aug_pipl", default_args=default_args, schedule_interval="30 * * * *",)

# t1, t2 and t3 are examples of tasks created by instantiating operators
t1 = BashOperator(task_id="extract_data", bash_command="date", dag=dag)
t3 = BashOperator(task_id="processing_data", bash_command="sleep 5", retries=3, dag=dag)

t2 = BashOperator(task_id="extract_text", bash_command="date", dag=dag)
t4 = BashOperator(task_id="processing_text", bash_command="sleep 5", retries=3, dag=dag)
t5 = BashOperator(task_id="augmentaition", bash_command="sleep 5", retries=3, dag=dag)
t6 = BashOperator(task_id="train_bert_text", bash_command="sleep 5", retries=3, dag=dag)
t7 = BashOperator(task_id="merge_text_d", bash_command="sleep 5", retries=3, dag=dag)

t8 = BashOperator(task_id="merge_text_n_data", bash_command="sleep 5", retries=3, dag=dag)

templated_command = """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7)}}"
        echo "{{ params.my_param }}"
    {% endfor %}
"""
t1 >> t3 >> t8
t2 >> t4 >> t5 >> t6 >> t7 >> t8