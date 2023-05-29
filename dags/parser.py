# Газпром новости
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
      }
url = 'https://iz.ru/tag/gazprom/publications'

# Подзагрузка ресурсов по кнопке "Загрузить ещё"
driver = webdriver.Chrome(executable_path="path to driver")
driver.get('https://example.com')
button = driver.find_element_by_xpath('xpath of the li you are trying to access')
button.click()

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'lxml')
news = soup.find_all('div')
#print(news)
for tag in soup.find_all('Газпром'):
        print(f'{tag.name}: {tag.text}')


