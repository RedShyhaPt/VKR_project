import numpy as np
from flask import 
import pickle
from flask import Flask, jsonify, abort, make_response, request, render_template
import json
import psycopg2
import pandas.io.sql as sqlio
import logging
import os
from datetime import timedelta
from decimal import Decimal
from matplotlib import pyplot as plt
from dataclasses import dataclass
from tinkoff.invest import MoneyValue, Quotation
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.strategies.base.account_manager import AccountManager
from tinkoff.invest.strategies.moving_average.plotter import MovingAverageStrategyPlotter
from tinkoff.invest.strategies.moving_average.signal_executor import MovingAverageSignalExecutor
from tinkoff.invest.strategies.moving_average.strategy import MovingAverageStrategy
from tinkoff.invest.strategies.moving_average.strategy_settings import MovingAverageStrategySettings
from tinkoff.invest.strategies.moving_average.strategy_state import MovingAverageStrategyState
from tinkoff.invest.strategies.moving_average.supervisor import MovingAverageStrategySupervisor,
from tinkoff.invest.strategies.moving_average.trader import MovingAverageStrategyTrader


token = os.environ.get('TINKOFF_TOKEN')
account_id = os.environ.get('TINKOFF_ACCOUNT')

# Create flask app


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/get_company", methods = ["GET"])
def get_company():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)

model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
conn = psycopg2.connect(
    host="localhost",
    database="stock_predicting",
    user="stock",
    password="stock",
    port="5432"
)
cursor = conn.cursor()

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

    suggest = "example"

    return jsonify({'news': parsed_news['data'], 'quotes': parsed_qoutes['data'], 'suggest': suggest})


@app.route('/api/companies', methods=['GET'])
def get_companies():
    tickers = select_companies()
    json_tickers = tickers.to_json(orient="table")
    parsed_tickers = json.loads(json_tickers)
    json.dumps(parsed_tickers, indent=4)
    return jsonify({'tickers': parsed_tickers['data']})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def backtest(robot):
    stats = robot.backtest(
        TradeStrategyParams(instrument_balance=0, currency_balance=15000, pending_orders=[]),
        train_duration=datetime.timedelta(days=5), test_duration=datetime.timedelta(days=30))
    stats.save_to_file('backtest_stats.pickle')


def trade(robot):
    stats = robot.trade()
    stats.save_to_file('stats.pickle')


def strategy():
    robot_factory = TradingRobotFactory(token=token, account_id=account_id, ticker='YNDX', class_code='TQBR',
                                        logger_level='INFO')
    robot = robot_factory.create_robot(MAEStrategy(visualizer=Visualizer('YNDX', 'RUB')), sandbox_mode=True)

    backtest(robot)

    trade(robot)


@dataclass(init=False, order=True)
class Money:
    units: int
    nano: int
    MOD: int = 10 ** 9

    def __init__(self, value: int | float | Quotation | MoneyValue, nano: int = None):
        if nano:
            assert isinstance(value, int), 'if nano is present, value must be int'
            assert isinstance(nano, int), 'nano must be int'
            self.units = value
            self.nano = nano
        else:
            match value:
                case int() as value:
                    self.units = value
                    self.nano = 0
                case float() as value:
                    self.units = int(math.floor(value))
                    self.nano = int((value - math.floor(value)) * self.MOD)
                case Quotation() | MoneyValue() as value:
                    self.units = value.units
                    self.nano = value.nano
                case _:
                    raise ValueError(f'{type(value)} is not supported as initial value for Money')

    def __float__(self):
        return self.units + self.nano / self.MOD

    def to_float(self):
        return float(self)

    def to_quotation(self):
        return Quotation(self.units, self.nano)

    def to_money_value(self, currency: str):
        return MoneyValue(currency, self.units, self.nano)

    def __add__(self, other: Money) -> Money:
        print(self.units + other.units + (self.nano + other.nano) // self.MOD)
        print((self.nano + other.nano) % self.MOD)
        return Money(
            self.units + other.units + (self.nano + other.nano) // self.MOD,
            (self.nano + other.nano) % self.MOD
        )

    def __neg__(self) -> Money:
        return Money(-self.units, -self.nano)

    def __sub__(self, other: Money) -> Money:
        return self + -other

    def __mul__(self, other: int) -> Money:
        return Money(self.units * other + (self.nano * other) // self.MOD, (self.nano * other) % self.MOD)

    def __str__(self) -> str:
        return f'<Money units={self.units} nano={self.nano}>'

from tinkoff.invest import (
    AccessLevel,
    AccountStatus,
    AccountType,
    Candle,
    CandleInstrument,
    CandleInterval,
    Client,
    InfoInstrument,
    Instrument,
    InstrumentIdType,
    MarketDataResponse,
    MoneyValue,
    OrderBookInstrument,
    OrderDirection,
    OrderExecutionReportStatus,
    OrderState,
    PostOrderResponse,
    Quotation,
    TradeInstrument,
)
from tinkoff.invest.exceptions import InvestError
from tinkoff.invest.services import MarketDataStreamManager, Services

from robotlib.strategy import TradeStrategyBase, TradeStrategyParams, RobotTradeOrder
from robotlib.stats import TradeStatisticsAnalyzer
from robotlib.money import Money


@dataclass
class OrderExecutionInfo:
    direction: OrderDirection
    lots: int = 0
    amount: float = 0.0


class TradingRobot:  # pylint:disable=too-many-instance-attributes
    APP_NAME: str = 'karpp'

    token: str
    account_id: str
    trade_strategy: TradeStrategyBase
    trade_statistics: TradeStatisticsAnalyzer
    orders_executed: dict[str, OrderExecutionInfo]  # order_id -> executed lots
    logger: logging.Logger
    instrument_info: Instrument
    sandbox_mode: bool

    def __init__(self, token: str, account_id: str, sandbox_mode: bool, trade_strategy: TradeStrategyBase,
                 trade_statistics: TradeStatisticsAnalyzer, instrument_info: Instrument, logger: logging.Logger):
        self.token = token
        self.account_id = account_id
        self.trade_strategy = trade_strategy
        self.trade_statistics = trade_statistics
        self.orders_executed = {}
        self.logger = logger
        self.instrument_info = instrument_info
        self.sandbox_mode = sandbox_mode

    def trade(self) -> TradeStatisticsAnalyzer:
        self.logger.info('Starting trading')

        self.trade_strategy.load_candles(
            list(self._load_historic_data(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1))))

        with Client(self.token, app_name=self.APP_NAME) as client:
            trading_status = client.market_data.get_trading_status(figi=self.instrument_info.figi)
            if not trading_status.market_order_available_flag:
                self.logger.warning('Market trading is not available now.')

            market_data_stream: MarketDataStreamManager = client.create_market_data_stream()
            if self.trade_strategy.candle_subscription_interval:
                market_data_stream.candles.subscribe([
                    CandleInstrument(
                        figi=self.instrument_info.figi,
                        interval=self.trade_strategy.candle_subscription_interval)
                ])
            if self.trade_strategy.order_book_subscription_depth:
                market_data_stream.order_book.subscribe([
                    OrderBookInstrument(
                        figi=self.instrument_info.figi,
                        depth=self.trade_strategy.order_book_subscription_depth)
                ])
            if self.trade_strategy.trades_subscription:
                market_data_stream.trades.subscribe([
                    TradeInstrument(figi=self.instrument_info.figi)
                ])
            market_data_stream.info.subscribe([
                InfoInstrument(figi=self.instrument_info.figi)
            ])
            self.logger.debug(f'Subscribed to MarketDataStream, '
                              f'interval: {self.trade_strategy.candle_subscription_interval}')
            try:
                for market_data in market_data_stream:
                    self.logger.debug(f'Received market_data {market_data}')
                    if market_data.candle:
                        self._on_update(client, market_data)
                    if market_data.trading_status and market_data.trading_status.market_order_available_flag:
                        self.logger.info(f'Trading is limited. Current status: {market_data.trading_status}')
                        break
            except InvestError as error:
                self.logger.info(f'Caught exception {error}, stopping trading')
                market_data_stream.stop()
            return self.trade_statistics

    def backtest(self, initial_params: TradeStrategyParams, test_duration: datetime.timedelta,
                 train_duration: datetime.timedelta = None) -> TradeStatisticsAnalyzer:

        trade_statistics = TradeStatisticsAnalyzer(
            positions=initial_params.instrument_balance,
            money=initial_params.currency_balance,
            instrument_info=self.instrument_info,
            logger=self.logger
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        if train_duration:
            train = self._load_historic_data(now - test_duration - train_duration, now - test_duration)
            self.trade_strategy.load_candles(list(train))
        test = self._load_historic_data(now - test_duration)

        params = initial_params
        for candle in test:
            price = self.convert_from_quotation(candle.close)
            robot_decision = self.trade_strategy.decide_by_candle(candle, params)

            trade_order = robot_decision.robot_trade_order
            if trade_order:
                assert trade_order.quantity > 0
                if trade_order.direction == OrderDirection.ORDER_DIRECTION_SELL:
                    assert trade_order.quantity >= params.instrument_balance, \
                        f'Cannot execute order {trade_order}. Params are {params}'  # TODO: better logging
                    params.instrument_balance -= trade_order.quantity
                    params.currency_balance += trade_order.quantity * price * self.instrument_info.lot
                else:
                    assert trade_order.quantity * self.instrument_info.lot * price <= params.currency_balance, \
                        f'Cannot execute order {trade_order}. Params are {params}'  # TODO: better logging
                    params.instrument_balance += trade_order.quantity
                    params.currency_balance -= trade_order.quantity * price * self.instrument_info.lot

                trade_statistics.add_backtest_trade(
                    quantity=trade_order.quantity, price=candle.close, direction=trade_order.direction)

        return trade_statistics

    @staticmethod
    def convert_from_quotation(amount: Quotation | MoneyValue) -> float | None:
        if amount is None:
            return None
        return amount.units + amount.nano / (10 ** 9)

    def _on_update(self, client: Services, market_data: MarketDataResponse):
        self._check_trade_orders(client)
        params = TradeStrategyParams(instrument_balance=self.trade_statistics.get_positions(),
                                     currency_balance=self.trade_statistics.get_money(),
                                     pending_orders=self.trade_statistics.get_pending_orders())

        self.logger.debug(f'Received market_data {market_data}. Running strategy with params {params}')
        strategy_decision = self.trade_strategy.decide(market_data, params)
        self.logger.debug(f'Strategy decision: {strategy_decision}')

        if len(strategy_decision.cancel_orders) > 0:
            self._cancel_orders(client=client, orders=strategy_decision.cancel_orders)

        trade_order = strategy_decision.robot_trade_order
        if trade_order and self._validate_strategy_order(order=trade_order, candle=market_data.candle):
            self._post_trade_order(client=client, trade_order=trade_order)

    def _validate_strategy_order(self, order: RobotTradeOrder, candle: Candle):
        if order.direction == OrderDirection.ORDER_DIRECTION_BUY:
            price = order.price or Money(candle.close)
            total_cost = price * self.instrument_info.lot * order.quantity
            balance = self.trade_statistics.get_money()
            if total_cost.to_float() > self.trade_statistics.get_money():
                self.logger.warning(f'Strategy decision cannot be executed. '
                                    f'Requested buy cost: {total_cost}, balance: {balance}')
                return False
        else:
            instrument_balance = self.trade_statistics.get_positions()
            if order.quantity > instrument_balance:
                self.logger.warning(f'Strategy decision cannot be executed. '
                                    f'Requested sell quantity: {order.quantity}, balance: {instrument_balance}')
                return False
        return True

    def _load_historic_data(self, from_time: datetime.datetime, to_time: datetime.datetime = None):
        try:
            with Client(self.token, app_name=self.APP_NAME) as client:
                yield from client.get_all_candles(
                    from_=from_time,
                    to=to_time,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
                    figi=self.instrument_info.figi,
                )
        except InvestError as error:
            self.logger.error(f'Failed to load historical data. Error: {error}')

    def _cancel_orders(self, client: Services, orders: list[OrderState]):
        for order in orders:
            try:
                client.orders.cancel_order(account_id=self.account_id, order_id=order.order_id)
                self.trade_statistics.cancel_order(order_id=order.order_id)
            except InvestError as error:
                self.logger.error(f'Failed to cancel order {order.order_id}. Error: {error}')

    def _post_trade_order(self, client: Services, trade_order: RobotTradeOrder) -> PostOrderResponse | None:
        try:
            if self.sandbox_mode:
                order = client.sandbox.post_sandbox_order(
                    figi=self.instrument_info.figi,
                    quantity=trade_order.quantity,
                    price=trade_order.price.to_quotation() if trade_order.price is not None else None,
                    direction=trade_order.direction,
                    account_id=self.account_id,
                    order_type=trade_order.order_type,
                    order_id=str(uuid.uuid4())
                )
            else:
                order = client.orders.post_order(
                    figi=self.instrument_info.figi,
                    quantity=trade_order.quantity,
                    price=trade_order.price.to_quotation() if trade_order.price is not None else None,
                    direction=trade_order.direction,
                    account_id=self.account_id,
                    order_type=trade_order.order_type,
                    order_id=str(uuid.uuid4())
                )
        except InvestError as error:
            self.logger.error(f'Posting trade order failed :(. Order: {trade_order}; Exception: {error}')
            return
        self.logger.info(f'Placed trade order {order}')
        self.orders_executed[order.order_id] = OrderExecutionInfo(direction=trade_order.direction)
        self.trade_statistics.add_trade(order)
        return order

    def _check_trade_orders(self, client: Services):
        self.logger.debug(f'Updating trade orders info. Current trade orders num: {len(self.orders_executed)}')
        orders_executed = list(self.orders_executed.items())
        for order_id, execution_info in orders_executed:
            if self.sandbox_mode:
                order_state = client.sandbox.get_sandbox_order_state(
                    account_id=self.account_id, order_id=order_id
                )
            else:
                order_state = client.orders.get_order_state(
                    account_id=self.account_id, order_id=order_id
                )

            self.trade_statistics.add_trade(trade=order_state)
            match order_state.execution_report_status:
                case OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
                    self.logger.info(f'Trade order {order_id} has been FULLY FILLED')
                    self.orders_executed.pop(order_id)
                case OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED:
                    self.logger.warning(f'Trade order {order_id} has been REJECTED')
                    self.orders_executed.pop(order_id)
                case OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED:
                    self.logger.warning(f'Trade order {order_id} has been CANCELLED')
                    self.orders_executed.pop(order_id)
                case OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL:
                    self.logger.info(f'Trade order {order_id} has been PARTIALLY FILLED')
                    self.orders_executed[order_id] = OrderExecutionInfo(lots=order_state.lots_executed,
                                                                        amount=order_state.total_order_amount,
                                                                        direction=order_state.direction)
                case _:
                    self.logger.debug(f'No updates on order {order_id}')

        self.logger.debug(f'Successfully updated trade orders. New trade orders num: {len(self.orders_executed)}')


class TradingRobotFactory:
    APP_NAME = 'karpp'
    instrument_info: Instrument
    token: str
    account_id: str
    logger: logging.Logger
    sandbox_mode: bool

    def __init__(self, token: str, account_id: str, figi: str = None,  # pylint:disable=too-many-arguments
                 ticker: str = None, class_code: str = None, logger_level: int | str = 'INFO'):
        self.instrument_info = self._get_instrument_info(token, figi, ticker, class_code).instrument
        self.token = token
        self.account_id = account_id
        self.logger = self.setup_logger(logger_level)
        self.sandbox_mode = self._validate_account(token, account_id, self.logger)

    def setup_logger(self, logger_level: int | str):
        logger = logging.getLogger(f'robot.{self.instrument_info.ticker}')
        logger.setLevel(logger_level)
        formatter = logging.Formatter(fmt=('%(asctime)s %(levelname)s: %(message)s'))  # todo: fixit
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_robot(self, trade_strategy: TradeStrategyBase, sandbox_mode: bool = True) -> TradingRobot:
        money, positions = self._get_current_postitions()
        trade_strategy.load_instrument_info(self.instrument_info)
        stats = TradeStatisticsAnalyzer(
            positions=positions,
            money=money.to_float(),  # todo: change to Money
            instrument_info=self.instrument_info,
            logger=self.logger.getChild(trade_strategy.strategy_id).getChild('stats')
        )
        return TradingRobot(token=self.token, account_id=self.account_id, sandbox_mode=sandbox_mode,
                            trade_strategy=trade_strategy, trade_statistics=stats, instrument_info=self.instrument_info,
                            logger=self.logger.getChild(trade_strategy.strategy_id))

    def _get_current_postitions(self) -> tuple[Money, int]:
        # amount of money and instrument balance
        with Client(self.token, app_name=self.APP_NAME) as client:
            positions = client.operations.get_positions(account_id=self.account_id)

            instruments = [sec for sec in positions.securities if sec.figi == self.instrument_info.figi]
            if len(instruments) > 0:
                instrument = instruments[0].balance
            else:
                instrument = 0

            moneys = [m for m in positions.money if m.currency == self.instrument_info.currency]
            if len(moneys) > 0:
                money = Money(moneys[0].units, moneys[0].nano)
            else:
                money = Money(0, 0)

            return money, instrument

    @staticmethod
    def _validate_account(token: str, account_id: str, logger: logging.Logger) -> bool:
        try:
            with Client(token, app_name=TradingRobotFactory.APP_NAME) as client:
                accounts = [acc for acc in client.users.get_accounts().accounts if acc.id == account_id]
                sandbox_mode = False
                if len(accounts) == 0:
                    sandbox_mode = True
                    accounts = [acc for acc in client.sandbox.get_sandbox_accounts().accounts if acc.id == account_id]
                    if len(accounts) == 0:
                        logger.error(f'Account {account_id} not found.')
                        raise ValueError('Account not found')

                account = accounts[0]
                if account.type not in [AccountType.ACCOUNT_TYPE_TINKOFF, AccountType.ACCOUNT_TYPE_INVEST_BOX]:
                    logger.error(f'Account type {account.type} is not supported')
                    raise ValueError('Unsupported account type')
                if account.status != AccountStatus.ACCOUNT_STATUS_OPEN:
                    logger.error(f'Account status {account.status} is not supported')
                    raise ValueError('Unsupported account status')
                if account.access_level != AccessLevel.ACCOUNT_ACCESS_LEVEL_FULL_ACCESS:
                    logger.error(f'No access to account. Current level is {account.access_level}')
                    raise ValueError('Insufficient access level')

                return sandbox_mode

        except InvestError as error:
            logger.error(f'Failed to validate account. Exception: {error}')
            raise error

    @staticmethod
    def _get_instrument_info(token: str, figi: str = None, ticker: str = None, class_code: str = None):
        with Client(token, app_name=TradingRobotFactory.APP_NAME) as client:
            if figi is None:
                if ticker is None or class_code is None:
                    raise ValueError('figi or both ticker and class_code must be not None')
                return client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                                                            class_code=class_code, id=ticker)
            return client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi)

if __name__ == '__main__':
    app.run()
# docker build -t azat/stock_predicting .
# docker run -p 5432:5432 azat/stock_predicting
# alter user postgres with password 'Bdfyd5'

# Create flask app