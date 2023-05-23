#open_	high_	low_	close_	vol_	price_	total_	weekday_ month_ hour_	year_	day_
import numpy as np 
import pickle
from flask import Flask, jsonify, abort, make_response, request, render_template
import json
import psycopg2
import pandas.io.sql as sqlio
from random import randrange

# Create flask app
model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
conn = psycopg2.connect(
    host="172.28.0.24",
    database="stock_predicting",
    user="stock",
    password="stock",
    port="5432"
)
cursor = conn.cursor()

def suggestion(prediction,last_values):
    avg_value_pred=0
    for i in prediction:
        avg_value_pred+=i
    avg_value_pred=(avg_value_pred/len(prediction))
    avg_last = last_values['price_'].mean()
    differents = avg_value_pred[0] - avg_last

    if differents > 5.0:
        return "Советуем к продаже"
    elif -5.0 <= differents <= 5.0:
        return "Советуем удержаться от покупки/продажи"
    else:
        return "Советуем к покупке"

def select_companies():
    return sqlio.read_sql_query('select ticker from companies', conn)

def select_company(company_ticker):
    return sqlio.read_sql_query(f'select id, ticker from companies where ticker=\'{company_ticker}\'', conn)


def select_quote(company_id):
    return sqlio.read_sql_query(
        f'select date_, time_, open_ from quotes where company_id = {company_id} order by date_ desc, time_ desc limit 100;', conn)


def select_news(company_id):
    return sqlio.read_sql_query(
        f'select news.id, news.content, news.title from news where company_id={company_id} order by date_time desc limit 5',
        conn
    )

@app.route('/api/company/<string:company_ticker>', methods=['GET'])
def get_company(company_ticker):  # put application's code here
    company = select_company(company_ticker)
    if company.shape[0] == 0:
        abort(404)
    company_id = company.iloc[0]['id']
    news = select_news(company_id)
    qoutes = select_quote(company_id)

    json_news = news.to_json(orient="table")
    parsed_news = json.loads(json_news)
    json.dumps(parsed_news, indent=4)

    json_qoutes = qoutes.to_json(orient="table")
    parsed_qoutes = json.loads(json_qoutes)
    json.dumps(parsed_qoutes, indent=4)

    num = randrange(1,6)
    array2 = sqlio.read_sql_query(f'select open_, high_, low_, close_, vol_ from quotes where company_id = {company_id} order by date_ desc, time_ desc limit 1;', conn)
    array3 = sqlio.read_sql_query(f'select total_,weekday_,month_,hour_,year_,day_ from aggregate_data where quotes_id IN (select id from quotes where company_id = {company_id} order by date_ desc, time_ desc limit 1) order by date_time desc;', conn)
    last_values = sqlio.read_sql_query(f'select price_ from aggregate_data where quotes_id IN (select id from quotes where company_id = {company_id} order by date_ desc, time_ desc limit {num*2} offset 1) order by date_time desc;', conn)
    array4=array2.join(array3)
    prediction = model.predict(array4)
    suggest=suggestion(prediction,last_values)
    a_set = set()
    for i in prediction:
        a_set.update(set(i))
    y = [round(i,2) for i in a_set]

    #price = str(y)

    return jsonify({'news': parsed_news['data'], 'quotes': parsed_qoutes['data'],'price':y,'suggest': suggest})


@app.route('/api/companies', methods=['GET'])
def get_companies():
    tickers = select_companies()
    json_tickers = tickers.to_json(orient="table")
    parsed_tickers = json.loads(json_tickers)
    json.dumps(parsed_tickers, indent=4)
    return jsonify({'tickers': parsed_tickers['data']})

@app.route("/api/predict/<string:company_ticker>", methods = ['GET'])
def predict(company_ticker):
    company = select_company(company_ticker)
    if company.shape[0] == 0:
        abort(404)
    company_id = company.iloc[0]['id']
    num = randrange(1,6)
    array2 = sqlio.read_sql_query(f'select open_, high_, low_, close_, vol_ from quotes where company_id = {company_id} order by date_ desc, time_ desc limit {num};', conn)
    array3 = sqlio.read_sql_query(f'select total_,weekday_,month_,hour_,year_,day_ from aggregate_data where quotes_id IN (select id from quotes where company_id = {company_id} order by date_ desc, time_ desc limit {num}) order by date_time desc;', conn)
    array4=array2.join(array3)
    prediction = model.predict(array4)
    a_set = set()
    for i in prediction:
        a_set.update(set(i))
    y = [round(i,2) for i in a_set]
    return y

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
# docker build -t azat/stock_predicting .
# docker run -p 5432:5432 azat/stock_predicting
# alter user postgres with password 'Bdfyd5'
