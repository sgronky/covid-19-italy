# Covid-19 DataSet & Reports

In the following I'm importing all the data and setting the pre-requisites for the analysis. The data is hosted by John Hopkins University and can be found in the link [here](https://github.com/CSSEGISandData/COVID-19). There are three files that contain **Total Confirmed Cases**, **Deaths** and **Recoveries**.


```python
## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
#%matplotlib inline

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams['figure.figsize'] = [15, 5]

from IPython import display
from ipywidgets import interact, widgets
```


```python
## Read Data for Cases, Deaths and Recoveries
confirmed_raw = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_raw = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_raw = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

```


```python
confirmed_raw.head(10)
```


```python
deaths_raw.head(10)
```


```python
recoveries_raw.head(10)
```

### Field description

- **Province/State**: China - province name; US/Canada/Australia/ - city name, state/province name; Others - name of the event (e.g., "Diamond Princess" cruise ship); other countries - blank.
- **Country/Region**: country/region name conforming to WHO (will be updated).
- **Last Update**: MM/DD/YYYY HH:mm (24 hour format, in UTC).
- **Confirmed**: the number of confirmed cases. For Hubei Province: from Feb 13 (GMT +8), we report both clinically diagnosed and lab-confirmed cases. For lab-confirmed cases only (Before Feb 17), please refer to who_covid_19_situation_reports. For Italy, diagnosis standard might be changed since Feb 27 to "slow the growth of new case numbers." (Source)
- **Deaths**: the number of deaths.
- **Recovered**: the number of recovered cases.

The following is used to clean the datasets:


```python
### Melt the dateframe into the right shape and set index
def cleandata(df_raw):
    df_cleaned=df_raw.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_name='Cases',var_name='Date')
    df_cleaned=df_cleaned.set_index(['Country/Region','Province/State','Date'])
    return df_cleaned 

# Clean all datasets
confirmed = cleandata(confirmed_raw)
deaths = cleandata(deaths_raw)
recoveries = cleandata(recoveries_raw)
```


```python
confirmed.head(10)
```

## Country Level Data


```python
### Get Countrywise Data
def countrydata(df_cleaned,oldname,newname):
    df_country = df_cleaned.groupby(['Country/Region','Date'])['Cases'].sum().reset_index()
    df_country = df_country.set_index(['Country/Region','Date'])
    df_country.index = df_country.index.set_levels([df_country.index.levels[0], pd.to_datetime(df_country.index.levels[1])])
    df_country = df_country.sort_values(['Country/Region','Date'],ascending=True)
    df_country = df_country.rename(columns={oldname:newname})
    return df_country
  
confirmed_country = countrydata(confirmed,'Cases','Total Confirmed Cases')
deaths_country = countrydata(deaths,'Cases','Total Deaths')
recoveries_country = countrydata(recoveries,'Cases','Total Recoveries')
```

Using the cumulative data provided, in the following some calculations are performed and added to the data set.


```python
### Get DailyData from Cumulative sum
def dailydata(dfcountry,oldname,newname):
    dfcountrydaily = dfcountry.groupby(level=0).diff().fillna(0)
    dfcountrydaily = dfcountrydaily.rename(columns={oldname:newname})
    return dfcountrydaily

newcases_country = dailydata(confirmed_country,'Total Confirmed Cases','Daily New Cases')
newdeaths_country = dailydata(deaths_country,'Total Deaths','Daily New Deaths')
newrecoveries_country = dailydata(recoveries_country,'Total Recoveries','Daily New Recoveries')
```


```python
country_consolidated = pd.merge(confirmed_country,newcases_country,how='left',left_index=True,right_index=True)
country_consolidated = pd.merge(country_consolidated,newdeaths_country,how='left',left_index=True,right_index=True)
country_consolidated = pd.merge(country_consolidated,deaths_country,how='left',left_index=True,right_index=True)
country_consolidated = pd.merge(country_consolidated,recoveries_country,how='left',left_index=True,right_index=True)
country_consolidated = pd.merge(country_consolidated,newrecoveries_country,how='left',left_index=True,right_index=True)
```

In the following some additional calculated fields are added. In particular:

$$
active\_cases = confirmed - recoveries - deaths
$$


```python
country_consolidated['Active Cases'] = country_consolidated['Total Confirmed Cases'] - country_consolidated['Total Deaths'] - country_consolidated['Total Recoveries']
country_consolidated['Share of Recoveries - Closed Cases'] = np.round(country_consolidated['Total Recoveries'] / (country_consolidated['Total Recoveries'] + country_consolidated['Total Deaths']),2)
country_consolidated['Death to Cases Ratio'] = np.round(country_consolidated['Total Deaths']/country_consolidated['Total Confirmed Cases'],3)

country_consolidated.head(10)
```

The following aggregates all the cases globally.


```python
## Get totals for all metrics
global_tot = country_consolidated.reset_index().groupby('Date').sum()
global_tot['Share of Recoveries - Closed Cases'] = np.round(global_tot['Total Recoveries'] / (global_tot['Total Recoveries'] + global_tot['Total Deaths']),2)
global_tot['Death to Cases Ratio'] = np.round(global_tot['Total Deaths'] / global_tot['Total Confirmed Cases'],3)
global_tot.tail(2)
```

Main charts on global key metrics


```python
# Create Plots that show Key Metrics For the Covid-19
chartcol='red'
fig = make_subplots(rows=3, cols=2,shared_xaxes=True,
                    specs=[[{}, {}],[{},{}],
                       [{"colspan": 2}, None]],
                    subplot_titles=('Total Confirmed Cases','Active Cases','Deaths','Recoveries','Death to Cases Ratio'))
fig.add_trace(go.Scatter(x=global_tot.index,y = global_tot['Total Confirmed Cases'],
                         mode='lines+markers',
                         name='Confirmed Cases',
                         line=dict(color=chartcol,width=2)),
                         row=1,col=1)

fig.add_trace(go.Scatter(x=global_tot.index,y = global_tot['Active Cases'],
                         mode='lines+markers',
                         name='Active Cases',
                         line=dict(color=chartcol,width=2)),
                         row=1,col=2)

fig.add_trace(go.Scatter(x=global_tot.index,y=global_tot['Total Deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color=chartcol,width=2)),
                         row=2,col=1)

fig.add_trace(go.Scatter(x=global_tot.index,y=global_tot['Total Recoveries'],
                         mode='lines+markers',
                         name='Recoveries',
                         line=dict(color=chartcol,width=2)),
                         row=2,col=2)

fig.add_trace(go.Scatter(x=global_tot.index,y=global_tot['Death to Cases Ratio'],
                         mode='lines+markers',
                         line=dict(color=chartcol,width=2)),
                         row=3,col=1)

fig.update_layout(showlegend=False)
```


```python
df_countries = country_consolidated.groupby(['Country/Region', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
df_countries = df_countries.drop_duplicates(subset = ['Country/Region'])
df_countries = df_countries[df_countries['Total Confirmed Cases']>0]

df_countries.head(10)
```


```python
last_date = df_countries['Date'].max()
last_date
```


```python
fig = go.Figure(data=go.Choropleth(
    locations = df_countries['Country/Region'],
    locationmode = 'country names',
    z = np.log10(df_countries['Total Confirmed Cases']),
    colorscale = 'Bluered',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Confirmed Cases (logarithmic scale) as of ' + last_date.strftime("%d %B, %Y"),
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)
```

In the following, we dig deeper into the top twenty countries.


```python
tot_country = country_consolidated.max(level=0)['Total Confirmed Cases'].reset_index().set_index('Country/Region')
tot_country = tot_country.sort_values(by='Total Confirmed Cases',ascending=False)
tot_country_excl_china = tot_country[~tot_country.index.isin(['China','Others'])]
top20_excl_china = tot_country_excl_china.head(20)
tot_country_top20 = tot_country.head(20)
```


```python
fig = go.Figure(go.Bar(x=tot_country_top20.index, y=top20_excl_china['Total Confirmed Cases'],
                      text=tot_country_top20['Total Confirmed Cases'],
            textposition='outside'))
fig.update_layout(title_text='Top 20 Countries by Total Confirmed Cases Excluding China')
fig.update_yaxes(showticklabels=False)

fig.show()
```


```python
df_countrydate = country_consolidated[country_consolidated.index.get_level_values("Country/Region").isin(tot_country_top20.index.get_level_values(0))]
df_countrydate = df_countrydate[["Total Confirmed Cases"]]
df_countrydate = df_countrydate.sort_values(by=["Date","Total Confirmed Cases"], ascending=[True, False]).reset_index()
df_countrydate["Date"] = df_countrydate["Date"].dt.strftime("%y-%m-%d")
```


```python
# Creating the visualization
fig = px.bar(df_countrydate,
                x=df_countrydate['Country/Region'],
                y=df_countrydate['Total Confirmed Cases'], 
                animation_frame="Date",
                color="Country/Region",
                log_y=True,
                orientation='v',
                title="Global Spread of Coronavirus over Time by Country"
            ).update_xaxes(categoryorder='total descending')
    
fig.show()
```

## Italy

Now, let's see how the cases have grown over time.


```python
italy_first_case = country_consolidated.loc['Italy']['Total Confirmed Cases'].reset_index().set_index('Date')
italy_growth = italy_first_case[italy_first_case.ne(0)].dropna().reset_index()

italy_growth.head(10)
```


```python
fig = go.Figure(go.Scatter(x=italy_growth["Date"], y=italy_growth['Total Confirmed Cases'],
                    text=italy_growth['Total Confirmed Cases'],
                    mode='lines+markers',                    
                    line=dict(color=chartcol,width=2),
                    textposition='top left'))
fig.update_layout(title_text='Total confirmed cases in Italy over time')
fig.update_yaxes(showticklabels=False)

fig.show()
```

The **logistic model** has been widely used to describe the **growth of a population**. An infection can be described as the growth of the population of a pathogen agent, so a logistic model seems reasonable.

The most generic expression of a logistic function is:

$$
f(x, a, b, c) = \frac{c}{1 + e^{-(x-b)/a}}
$$



```python
from scipy.optimize import curve_fit

def logistic(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

x = italy_growth.index
y = italy_growth['Total Confirmed Cases']

popt, pcov = curve_fit(logistic, x, y, p0=(1, 1e-6, 1))

# xx = italy_growth.index
xx = np.arange(200)
yy = logistic(xx, *popt)
```

The following values represent the parameters fitting the curve:


```python
popt[0], popt[1], popt[2]
```

The function returns the covariance matrix too, whose diagonal values are the variances of the parameters. Taking their square root we can calculate the standard errors.


```python
errors = [np.sqrt(pcov[i][i]) for i in [0,1,2]]
errors
```

Considering the errors above:

- The expected **number of infected people** at infection end is $127168 \pm 1860$.
- The **infection peak** is expected around **april 20**
- The **expected infection end** can be calculated as that particular day at which the cumulative infected people count is equal to the $c$ parameter rounded to the nearest integer.

To numerically find the root of the equation that defines the infection end day:


```python
from scipy.optimize import fsolve

sol = int(fsolve(lambda x : logistic(x,popt[0],popt[1],popt[2]) - int(popt[2]),popt[1]))
```


```python
import datetime
italy_growth.iloc[0]['Date'] + datetime.timedelta(days=sol)
```


```python
fig = go.Figure(go.Scatter(x=italy_growth.index, y=italy_growth['Total Confirmed Cases'],
                    text=italy_growth['Total Confirmed Cases'],
                    mode='markers',                    
                    line=dict(color=chartcol,width=2),
                    textposition='top left',
                    name='Total confirmed cases'))
fig.update_layout(title_text='Total confirmed cases in Italy over time')
fig.update_yaxes(showticklabels=False)

trace2 = go.Scatter(
                    x=xx,
                    y=yy,
                    mode='lines',
                    line=dict(color='mediumblue', width=2),
                    name='Logistic Fit'
                  )

fig.add_trace(trace2)
fig.show()
```

The following chart shows active cases, recoveries and deaths.


```python
italy_details = country_consolidated.loc['Italy'].reset_index().set_index('Date')
italy_details = italy_details[italy_details.ne(0)].dropna().reset_index()

italy_details.head(10)
```


```python
fig = go.Figure(data=[
    go.Bar(name='Active cases', x=italy_details['Date'], y=italy_details['Active Cases']),
    go.Bar(name='Deaths', x=italy_details['Date'], y=italy_details['Total Deaths']),
    go.Bar(name='Recoveries', x=italy_details['Date'], y=italy_details['Total Recoveries'])
])
# Change the bar mode
fig.update_layout(title_text='Active cases, deaths and recoveries')
fig.update_layout(barmode='stack')
fig.show()
```


```python
fig = go.Figure(data=[
    go.Scatter(name='Daily New Cases', x=italy_details['Date'], y=italy_details['Daily New Cases'], mode='lines+markers'),
    go.Scatter(name='Daily New Deaths', x=italy_details['Date'], y=italy_details['Daily New Deaths'], mode='lines+markers'),
    go.Scatter(name='Daily New Recoveries', x=italy_details['Date'], y=italy_details['Daily New Recoveries'], mode='lines+markers')
])

# Change the bar mode
fig.update_layout(title_text='Daily Growth')
fig.show()
```

The following chart illustrates how the cases are closed with recoveries.


```python
fig = go.Figure(go.Bar(x=italy_details['Date'], y=italy_details['Share of Recoveries - Closed Cases'],
                      text=italy_details['Share of Recoveries - Closed Cases'],
            textposition='outside'))
fig.update_layout(title_text='Share of Recoveries - Closed Cases')
fig.update_yaxes(showticklabels=False)

fig.show()
```

## Analysis of Italy's cases by province

Italy's Civil Protection Department collects detailed about the virus, coming from the entire geographic territory. The data are organized according to regions and provinces. In the following, some further details regarding the different provinces are analyzed according to the same approach conducted so far.


```python
## Read Data from Civil Protection Department repository
dpc_province_raw = pd.read_json('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-province.json')
dpc_province_raw.head(10)
```

### Field Description

- **data**: Reference date.
- **stato**: Always ITA for Italy.
- **codice_regione**: Region's id.
- **denominazione_regione**: Name of the region.
- **codice_provincia**: Province's id.
- **denominazione_provincia**: Name of the province.
- **sigla_provincia**: Short code to represent the province.
- **lat**: Latitude of a reference point inside the province's territory.
- **long**: Longitude of a reference point inside the province's territory.
- **totale_casi**: Number of cases.

The following code is used to clean the re-organize the data.


```python
dpc_province = dpc_province_raw.set_index(['codice_provincia', 'data'])
dpc_province.index = dpc_province.index.set_levels([dpc_province.index.levels[0], pd.to_datetime(dpc_province.index.levels[1])])
dpc_province = dpc_province.sort_values(['codice_provincia','data'], ascending = True)
```


```python
dpc_province.head(10)
```

An example for Milan and Bergamo areas.


```python
dpc_milan = dpc_province.loc[15]['totale_casi'].reset_index().set_index('data')
dpc_bergamo = dpc_province.loc[16]['totale_casi'].reset_index().set_index('data')
```


```python
dpc_milan.head(10)
```


```python
fig = go.Figure(data=[
            go.Bar(name='Cases in Milan area', x=dpc_milan.index, y=dpc_milan['totale_casi']),
            go.Bar(name='Cases in Bergamo area', x=dpc_bergamo.index, y=dpc_bergamo['totale_casi'])
        ])
fig.update_layout(title_text='Total cases for Milan and Bergamo areas')
fig.update_layout(barmode='group')
fig.update_yaxes(showticklabels=False)

fig.show()
```


```python
tot_provinces = dpc_province.max(level=0)[['denominazione_provincia','totale_casi']].reset_index().set_index('codice_provincia')
tot_provinces = tot_provinces[tot_provinces['denominazione_provincia'] != "In fase di definizione/aggiornamento"]
tot_provinces = tot_provinces.sort_values(by='totale_casi',ascending=False)
tot_provinces_top30 = tot_provinces.head(30)
```


```python
fig = go.Figure(data=[
            go.Bar(name='Cases by provinces', 
                   x=tot_provinces_top30['denominazione_provincia'], 
                   y=tot_provinces_top30['totale_casi'],)
        ])
fig.update_layout(title_text='Total cases by provinces')
fig.update_yaxes(showticklabels=False)

fig.show()
```

## Swabs

We use a different file to get more insights on the whole national trend, e.g. the number of swabs vs the total number of positive cases.


```python
## Read data regarding the national trend
dpc_national_trend_raw = pd.read_json('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-andamento-nazionale.json')
dpc_national_trend_raw.set_index('data')
dpc_national_trend = dpc_national_trend_raw.drop(columns=['stato'])
dpc_national_trend['percentuale_tamponi_positivi'] = np.round(dpc_national_trend['totale_casi'] / dpc_national_trend['tamponi'], 3)

dpc_national_trend.head(10)
```


```python
fig = go.Figure(data=[
    go.Scatter(name='Cases', x=dpc_national_trend['data'], y=dpc_national_trend['totale_casi'], mode='lines+markers'),
    go.Scatter(name='Swabs', x=dpc_national_trend['data'], y=dpc_national_trend['tamponi'], mode='lines+markers')
])

# Change the bar mode
fig.update_layout(title_text='Total Number of swabs vs Total number of cases')
fig.show()
```


```python
fig = go.Figure(go.Bar(x=dpc_national_trend['data'], y=dpc_national_trend['percentuale_tamponi_positivi'],
                      text=dpc_national_trend['percentuale_tamponi_positivi'],
            textposition='outside'))
fig.update_layout(title_text='Share of Positive Swabs')
fig.update_yaxes(showticklabels=False)

fig.show()
```

## Italy regions


```python
dpc_regions = dpc_province_raw.groupby(['data','codice_regione']).sum().reset_index()
dpc_regions = dpc_regions.drop_duplicates(subset = ['codice_regione'])
dpc_regions.head(10)
```
