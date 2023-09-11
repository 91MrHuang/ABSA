from locust_plugins.csvreader import CSVReader
from locust import HttpUser, between, task, tag, events
import json
from json import JSONDecodeError
import logging
import datetime

asc_reader = CSVReader('test_data/asc.csv')

class ASCUser(HttpUser):
    weight = 3  # the weight of this type of user
    wait_time = between(1, 5)  # time to wait (in seconds) after each task

    # An on_start method that's additionally declared. A method with this name will be called for each simulated user when they start.
    # You could do some initialization stuff like login.
    # def on_start(self):
        # self.client.post("/login", {
        #     "username": "test_user",
        #     "password": ""
        # })

    # Similarly, a method with on_stop will be called for each simulated user when they stop.
    # def on_stop(self):
        # self.client.get('/')

    # Methods decorated with @task are the core of your locust file. For every running user, Locust creates a greenlet (micro-thread), that will call those methods.
    # Note that only methods decorated with @task will be picked
    @tag('load testing for Aspect level Sentiment Classification')
    @task(1)
    def do_asc(self):
        test_sentences = json.loads(next(asc_reader)[0].replace('\'', '"'))
        with self.client.post('', json=test_sentences, catch_response=True) as response:
            try:
                response_json = response.json()
                # ...
            except JSONDecodeError:
                response.failure('Response could not be decoded as JSON')

    @events.quitting.add_listener
    def _(environment, **kw):
        fh = logging.FileHandler('asc_load_test.log')
        logger = logging.getLogger('asc_load_test')
        logger.addHandler(fh)

        today = datetime.datetime.today()
        if environment.stats.total.fail_ratio > 0.01:
            logger.error('{} - Test failed due to failure ratio > 1%'.format(today))
            environment.process_exit_code = 1
        elif environment.stats.total.avg_response_time > 500:
            logger.error('{} - Test failed due to average response time > 500 ms'.format(today))
            environment.process_exit_code = 1
        elif environment.stats.total.get_response_time_percentile(0.95) > 800:
            logger.error('{} - Test failed due to 95th percentile response time > 800 ms'.format(today))
            environment.process_exit_code = 1
        else:
            logger.info('{} - Test succeeded'.format(today))
            environment.process_exit_code = 0
