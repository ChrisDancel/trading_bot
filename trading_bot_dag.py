import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

PROJECT_DIR = '/Users/christopherdancel/Documents/github/trading_bot/'
PROJECT_MAIN = os.path.join(PROJECT_DIR, 'main.py')
PYTHON_ENV = '/Users/christopherdancel/.virtualenvs/trading_bot_test/bin/python'

default_args = {
    'owner': 'me',
    'schedule_interval': None,
    'start_date': datetime(2015, 12, 1),
    'retries': 0,
}
dag = DAG('trading_bot', catchup=False, default_args=default_args)

t1 = BashOperator(
    task_id='build_history',
    bash_command=f"cd {PROJECT_DIR} && {PYTHON_ENV} {PROJECT_MAIN} --build_history",
    dag=dag)

t2 = BashOperator(
    task_id='forecast',
    bash_command=f'cd {PROJECT_DIR} && {PYTHON_ENV} {PROJECT_MAIN} --forecast',
    dag=dag)

t3 = BashOperator(
    task_id='email',
    bash_command=f'cd {PROJECT_DIR} && {PYTHON_ENV} {PROJECT_MAIN} --email',
    dag=dag)

t1 >> t2 >> t3
