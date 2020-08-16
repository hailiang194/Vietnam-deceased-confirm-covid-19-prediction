import requests
import urllib

link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-15-2020.csv"

f = urllib.request.urlopen(link)
myfile = f.read()

# print(myfile)
