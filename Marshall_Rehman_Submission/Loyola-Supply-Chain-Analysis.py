#!/usr/bin/env python
# coding: utf-8

# # Loyola Supply Chain Data Analysis Competition

# ## Contributers:
# 
#     Aarij Rehman
#     Charlie Marshall

# ## Loading Packages

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


# ## Loading in Data (Charlie):
# 
# Aarij will have to use different directories for his data.

# In[2]:


FreightWaves=pd.read_excel("/Users/charlesmarshall/Desktop/Loyola-Supply-Chain-Competition/FreightWaves External Data.xlsx")


# In[3]:


ShipperData=pd.read_excel("/Users/charlesmarshall/Desktop/Loyola-Supply-Chain-Competition/Loyola 2nd Annual Data Competition - Shipper Data Set (Trailing 2-Years).xlsx")


# In[4]:


#FreightWaves=pd.read_excel("/Users/aarij/Desktop/python-projects/loyola-supply-chain/FreightWaves External Data.xlsx")


# In[5]:


#ShipperData=pd.read_excel("/Users/aarij/Desktop/python-projects/loyola-supply-chain/Loyola 2nd Annual Data Competition - Shipper Data Set (Trailing 2-Years).xlsx")


# In[6]:


ShipperData.head()


# ## EDA
# 
# Understanding the data:

# In[7]:


ShipperData['Book Date']=pd.to_datetime(ShipperData['Book Date'])
ShipperData['Pickup Date']=pd.to_datetime(ShipperData['Pickup Date'])
ShipperData['Delivery Date']=pd.to_datetime(ShipperData['Delivery Date'])


# In[8]:


set(ShipperData['Origin Market'])


# In[9]:


set(ShipperData['Mode'])


# In[10]:


set(ShipperData['Spot/Contract'])


# In[11]:


set(ShipperData['Load Type'])


# In[12]:


set(FreightWaves['index_name'])


# ## Preprocessing
# 
# Some preproccessing that needs to be done to the whole data set

# ### 7 FreightWaves Futures Lanes:
#     VLS (LA -> Seattle)
#     VSL (Seattle -> LA)
#     VLD (LA -> Dallas)
#     VDL (Dallas -> LA)
#     VCA (Chicago -> Atlanta)
#     VAP (Atlanta -> Philadelphia)
#     VPC (Philadelphia -> Chicago)
#     
#     If the load is not delivered in one of these lanes, it will be marked 'Other'

# In[13]:


# Categorizing the Future Lanes
Lane = []
org = list(ShipperData['Origin Market'])
dst = list(ShipperData['Dest Market'])
los = 'Los Angeles'
sea = 'Seattle'
dal = 'Dallas'
chi = 'Chicago'
atl = 'Atlanta'
phi = 'Philadelphia'

futures = {}
futures[(los, sea)] = 'VLS'
futures[(sea, los)] = 'VSL'
futures[(los, dal)] = 'VLD'
futures[(dal, los)] = 'VDL'
futures[(chi, atl)] = 'VCA'
futures[(atl, phi)] = 'VAP'
futures[(phi, chi)] = 'VPC'


# In[14]:


Lane=[]
for i in range(len(org)):
    if (org[i], dst[i]) in futures:
        Lane.append(futures[(org[i], dst[i])])
    else:
        Lane.append('Other')


# In[15]:


ShipperData['Lane'] = Lane


# In[16]:


ShipperData.head()


# ### Rate per Mile
# 
#     Calculating the Rate per Mile (RPM) for all loads

# In[17]:


rpm=np.empty(len(ShipperData))


# In[18]:


for i in range(len(ShipperData)):
    if ShipperData['Mileage'][i]==0 or ShipperData['Revenue'][i]==0:
        rpm[i]=np.nan
    else:
        try:
            rpm[i]= ShipperData['Revenue'][i]/ShipperData['Mileage'][i]
        except TypeError:
            rpm[i]=np.nan


# In[19]:


ShipperData['RPM'] = rpm


# In[20]:


ShipperData=ShipperData.dropna(subset=['RPM'])


# ## Questions to Ask:

# #### Part 1    
#     1) Does 2019 mean the book, pickup, or delivery date is in 2019? There are many times when the one of these three meaures are in another year while other measures are in 2019.
#     - Something to think about: when the deal is booked is when the rate is agreed, so it might be more appropriate to do year by book date.
#     2) When you say "Truckload" RPM", does this mean only for trucks and dry vans?
#     3) How should we deal with instances where the mileage is 0 or blank? Outlier
#     4) How would you define "volatility"? Does it have something to do with spot vs contract market prices and how they change? Maybe the spread of the prices?

# ## Question 1
# 
#     Calculating overall average Truckload rate per mile in 2019?  Hint: rate per mile is total revenue for a move divided by total mileage.  Please also note that “Dry Van” and “Truck” mean the same thing here:
# 
#     Steps:
#     Year = 2019
#     Truckload = "Dry Van" or "Truck" in Mode
#     Create column of Rate per mile (RPM), divide revenue by mileage for each truckload
#     Sum the RPM column, divide by number of instances to find AVERAGE
#     

# ### Converting all date columns to datetime

# In[21]:


Shipper2019=ShipperData[ShipperData['Book Date'].dt.year == 2019]


# In[22]:


Shipper2019.head()


# In[23]:


Q1=Shipper2019[Shipper2019['Mode'].isin(['Dry Van','Truck'])].copy()


# In[24]:


Q1=Q1.reset_index(drop=True)


# In[25]:


Q1[Q1['Mileage']==0].head()


# In[26]:


Q1[Q1['Revenue']==0].head()


# In[27]:


Average2019RPM=sum(Q1['RPM'])/len(Q1)


# In[28]:


Average2019RPM


# ### Question 1a
# 
#     Using this Redwood rate per mile data, rank the 7 FreightWaves Futures lanes (page 2 at link) by rate volatility for full year 2019.
#     
#     Volatility = sd
#     So, the Futures lanes will be ranked by sd of RPM data. Highest to lowest.

# In[29]:


LaneVolitility2019=pd.DataFrame(Q1.groupby('Lane')['RPM'].std(),columns=['RPM']).drop(['Other']).sort_values(by=['RPM'],ascending=False)


# In[30]:


LaneVolitility2019


# ### Question 1b
# 
#     Rank these lanes by total Redwood Truckload volume for full year 2019.
#     
#     Defining "volume" as the number (count) of truckloads delivered in each lane in 2019.

# In[31]:


LaneVolume2019=pd.DataFrame(Q1.groupby('Lane')['RPM'].count()).drop(['Other']).sort_values(by=['RPM'],ascending=False).rename(columns={'RPM':'Volume'})


# In[32]:


LaneVolume2019


# In[33]:


Q1['Lane'].value_counts()


# ## Question 2
# 
#     Rank the lanes provided in the FreightWaves data by total intermodal volume for full year 2019.

# In[34]:


Q2=FreightWaves[FreightWaves['ticker']=='ORAIL53L']


# In[35]:


Q2=Q2[Q2['data_timestamp'].dt.year == 2019]


# In[36]:


Q2.groupby(['granularity1'])['data_value'].sum().sort_values(ascending=False)


# ## Question 3
# 
#     Graph monthly volatility in fuel prices by markets provided in the FreightWaves data.  Do any markets stand out to you in particular?

# In[37]:


FreightWaves['data_timestamp']=pd.to_datetime(FreightWaves['data_timestamp'])


# In[38]:


FreightWaves.head()


# In[39]:


set(FreightWaves['ticker'])


# In[40]:


Q3=FreightWaves[FreightWaves['ticker']=='DTS']


# In[41]:


Q3=Q3.reset_index(drop=True)


# In[42]:


Q3.head()


# In[43]:


set(Q3['Description'])


# In[44]:


set(Q3['granularity1'])


# In[45]:


per = Q3.data_timestamp.dt.to_period("M")


# In[46]:


DTS_Monthly_Volatility=Q3.groupby(['granularity1',per]).std()


# In[47]:


DTS_Monthly_Volatility.head()


# In[48]:


months=['2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01','2020-02']
y_atl=DTS_Monthly_Volatility['data_value'][0:24]
y_chi=DTS_Monthly_Volatility['data_value'][24:48]
y_dal=DTS_Monthly_Volatility['data_value'][48:72]
y_lax=DTS_Monthly_Volatility['data_value'][72:96]
y_phl=DTS_Monthly_Volatility['data_value'][96:120]
y_sea=DTS_Monthly_Volatility['data_value'][120:144]


# In[49]:


fig1 = plt.figure(figsize=(25,10))
plt.plot( months, y_atl, marker='', color='b', linewidth=2,label="ATL")
plt.plot( months, y_chi, marker='', color='g', linewidth=2,label="CHI")
plt.plot( months, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot( months, y_lax, marker='', color='c', linewidth=2,label="LAX")
plt.plot( months, y_phl, marker='', color='m', linewidth=2,label="PHL")
plt.plot( months, y_sea, marker='', color='y', linewidth=2,label="SEA")
plt.title('Monthly Volatility in Fuel Prices by Markets',fontsize='xx-large')
plt.ylabel('Volatility ($)',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')
plt.legend(fontsize='xx-large')


# ## Question 4
# 
#     Briefly explain how the mixture of spot and contract (% of total volume) of Dry-Van freight in the Redwood Data varied throughout 2018-2019.  What factors do you think would cause a shipper to sign a forward-looking Truckload rate contract or rely on the spot market?

# In[50]:


Shipper2019 = ShipperData[(ShipperData['Pickup Date'].dt.year == 2019) & (ShipperData['Mode'].isin(['Dry Van', 'Truck']))]
Shipper2018 = ShipperData[(ShipperData['Pickup Date'].dt.year == 2018) & (ShipperData['Mode'].isin(['Dry Van', 'Truck']))]


# In[51]:


spot18 = []
spot19 = []
for month in range(1, 13):
    spot18.append(Shipper2018[Shipper2018['Pickup Date'].dt.month == month]['Spot/Contract'].value_counts(normalize=True)['Spot'])
    spot19.append(Shipper2019[Shipper2019['Pickup Date'].dt.month == month]['Spot/Contract'].value_counts(normalize=True)['Spot'])
spot = spot18 + spot19


# In[52]:


cont = [1-i for i in spot]


# In[53]:


months=['2018-01', '2018-02', '2018-03', '2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12']

fig = plt.figure(figsize=(25,10))
plt.plot(months, spot, marker='', color='b', linewidth=2,label="Spot")
plt.plot(months, cont, marker='', color='g', linewidth=2,label="Contract")

plt.title('Relative Volume of Spot and Contract Shipments',fontsize='xx-large')
plt.ylabel('Relative Volume',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')
plt.legend(fontsize='xx-large')


# ## Question 5
# 
#     Based on initial analysis, please list and explain the (3) “external factors” from the FreightWaves data you think look the most promising for predicting 2020 dry van shipper rates as a Shipper.  

# In[54]:


Q5=ShipperData[ShipperData['Mode'].isin(['Dry Van','Truck'])]


# In[55]:


Q5=Q5.reset_index(drop=True)


# In[56]:


Q5.head()


# In[57]:


FreightWaves.head()


# ### Feature Engineering
# 
#     Task is to find the features in the Freight Waves data that are best at predicting RPM in 2020. The tricky part is how to represent each of these features.
# 
# ### Potential Features:
# 
#     1) Past "performance" of indecies
#         - past month should be used because it is improper to use future data to predict current performance. For instance, you can use a current monthly average to train the data on, but when predicting on future data, there will be no way to calculate the average for a current month because the month is not yet over.
#         - what period of time is relevant? this most likely changes based on the index but multiple time periods should be explored.
#     2) Comparing Y2Y rates might also be smart.

# ### EDA
# 
# Exploring each of the indecies to better understand them and how they should be utilized in a model.

# In[58]:


set(FreightWaves['index_name'])


# In[59]:


set(FreightWaves['ticker'])


# ### COSP

# In[60]:


COSP=FreightWaves[FreightWaves['ticker']=='COSP']


# In[61]:


COSP=COSP.sort_values(by='data_timestamp').reset_index(drop=True)


# In[62]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12']

fig2 = plt.figure(figsize=(25,10))
plt.plot(months, COSP['data_value'], marker='o', color='b', linewidth=2)
plt.title('Monthly Total Construction Spending',fontsize='xx-large')
plt.ylabel('Total Construction Spending ($)',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')


# ### CSTM

# In[63]:


CSTM=FreightWaves[FreightWaves['ticker']=='CSTM']


# In[64]:


set(CSTM['granularity1'])


# In[65]:


CSTM_value=CSTM.groupby(['granularity1','data_timestamp']).mean()


# In[66]:


CSTM_value = CSTM_value.reset_index()


# In[67]:


cstm_chi=list(CSTM_value[CSTM_value['granularity1']=='CHI']['data_timestamp'])
cstm_dal=list(CSTM_value[CSTM_value['granularity1']=='DAL']['data_timestamp'])
cstm_lax=list(CSTM_value[CSTM_value['granularity1']=='LAX']['data_timestamp'])
cstm_phl=list(CSTM_value[CSTM_value['granularity1']=='PHL']['data_timestamp'])
cstm_sea=list(CSTM_value[CSTM_value['granularity1']=='SEA']['data_timestamp'])


# In[68]:


y_chi=CSTM_value[CSTM_value['granularity1']=='CHI']['data_value']
y_dal=CSTM_value[CSTM_value['granularity1']=='DAL']['data_value']
y_lax=CSTM_value[CSTM_value['granularity1']=='LAX']['data_value']
y_phl=CSTM_value[CSTM_value['granularity1']=='PHL']['data_value']
y_sea=CSTM_value[CSTM_value['granularity1']=='SEA']['data_value']


# In[69]:


fig3 = plt.figure(figsize=(25,10))
plt.plot(cstm_chi, y_chi, marker='', color='g', linewidth=2,label="CHI")
plt.title('US Customs Maritime Import Shipments Time Series for CHI',fontsize='xx-large')
plt.ylabel('US Customs Maritime Import Shipments',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')


# In[70]:


fig4 = plt.figure(figsize=(25,10))
plt.plot(cstm_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.title('US Customs Maritime Import Shipments Time Series for DAL',fontsize='xx-large')
plt.ylabel('US Customs Maritime Import Shipments',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')


# In[71]:


fig5 = plt.figure(figsize=(25,10))
plt.plot(cstm_lax, y_lax, marker='', color='c', linewidth=2,label="LAX")
plt.title('US Customs Maritime Import Shipments Time Series for LAX',fontsize='xx-large')
plt.ylabel('US Customs Maritime Import Shipments',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')


# In[72]:


fig6 = plt.figure(figsize=(25,10))
plt.plot(cstm_phl, y_phl, marker='', color='m', linewidth=2,label="PHL")
plt.title('US Customs Maritime Import Shipments Time Series for PHL',fontsize='xx-large')
plt.ylabel('US Customs Maritime Import Shipments',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')


# In[73]:


fig7 = plt.figure(figsize=(25,10))
plt.plot(cstm_sea, y_sea, marker='', color='y', linewidth=2,label="SEA")
plt.title('US Customs Maritime Import Shipments Time Series for SEA',fontsize='xx-large')
plt.ylabel('US Customs Maritime Import Shipments',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')


# ### DATVF

# In[74]:


DATVF=FreightWaves[FreightWaves['ticker']=='DATVF']


# In[75]:


set(DATVF['granularity1'])


# In[76]:


#per = DATVF.data_timestamp.dt.to_period("D")


# In[77]:


DATVF_value=DATVF.groupby(['granularity1','data_timestamp']).mean()


# In[78]:


DATVF_value=DATVF_value.reset_index()


# In[79]:


datvf_vap=list(DATVF_value[DATVF_value['granularity1']=='ATLPHL']['data_timestamp'])
datvf_vca=list(DATVF_value[DATVF_value['granularity1']=='CHIATL']['data_timestamp'])
datvf_vdl=list(DATVF_value[DATVF_value['granularity1']=='DALLAX']['data_timestamp'])
datvf_vld=list(DATVF_value[DATVF_value['granularity1']=='LAXDAL']['data_timestamp'])
datvf_vls=list(DATVF_value[DATVF_value['granularity1']=='LAXSEA']['data_timestamp'])
datvf_vpc=list(DATVF_value[DATVF_value['granularity1']=='PHLCHI']['data_timestamp'])
datvf_vsl=list(DATVF_value[DATVF_value['granularity1']=='SEALAX']['data_timestamp'])


# In[80]:


y_vap=DATVF_value[DATVF_value['granularity1']=='ATLPHL']['data_value']
y_vca=DATVF_value[DATVF_value['granularity1']=='CHIATL']['data_value']
y_vdl=DATVF_value[DATVF_value['granularity1']=='DALLAX']['data_value']
y_vld=DATVF_value[DATVF_value['granularity1']=='LAXDAL']['data_value']
y_vls=DATVF_value[DATVF_value['granularity1']=='LAXSEA']['data_value']
y_vpc=DATVF_value[DATVF_value['granularity1']=='PHLCHI']['data_value']
y_vsl=DATVF_value[DATVF_value['granularity1']=='SEALAX']['data_value']


# In[81]:


fig8 = plt.figure(figsize=(25,10))
plt.plot(datvf_vap, y_vap, marker='', color='g', linewidth=2,label="ATLPHL")
plt.plot(datvf_vca, y_vca, marker='', color='r', linewidth=2,label="CHIATL")
plt.plot(datvf_vdl, y_vdl, marker='', color='b', linewidth=2,label="DALLAX")
plt.plot(datvf_vld, y_vld, marker='', color='y', linewidth=2,label="LAXDAL")
plt.plot(datvf_vls, y_vls, marker='', color='c', linewidth=2,label="LAXSEA")
plt.plot(datvf_vpc, y_vpc, marker='', color='m', linewidth=2,label="PHLCHI")
plt.plot(datvf_vsl, y_vsl, marker='', color='k', linewidth=2,label="SEALAX")
plt.title('DAT Longhaul Van Freight Rates Time Series',fontsize='xx-large')
plt.ylabel('DAT Longhaul Van Freight Rate',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### DTS

# In[82]:


DTS=FreightWaves[FreightWaves['ticker']=='DTS']


# In[83]:


set(DTS['granularity1'])


# In[84]:


#per = DTS.data_timestamp.dt.to_period("D")


# In[85]:


DTS_value=DTS.groupby(['granularity1','data_timestamp']).mean()


# In[86]:


DTS_value=DTS_value.reset_index()


# In[87]:


dts_atl=list(DTS_value[DTS_value['granularity1']=='ATL']['data_timestamp'])
dts_chi=list(DTS_value[DTS_value['granularity1']=='CHI']['data_timestamp'])
dts_dal=list(DTS_value[DTS_value['granularity1']=='DAL']['data_timestamp'])
dts_lax=list(DTS_value[DTS_value['granularity1']=='LAX']['data_timestamp'])
dts_phl=list(DTS_value[DTS_value['granularity1']=='PHL']['data_timestamp'])
dts_sea=list(DTS_value[DTS_value['granularity1']=='SEA']['data_timestamp'])


# In[88]:


y_atl=DTS_value[DTS_value['granularity1']=='ATL']['data_value']
y_chi=DTS_value[DTS_value['granularity1']=='CHI']['data_value']
y_dal=DTS_value[DTS_value['granularity1']=='DAL']['data_value']
y_lax=DTS_value[DTS_value['granularity1']=='LAX']['data_value']
y_phl=DTS_value[DTS_value['granularity1']=='PHL']['data_value']
y_sea=DTS_value[DTS_value['granularity1']=='SEA']['data_value']


# In[89]:


fig9 = plt.figure(figsize=(25,10))
plt.plot(dts_atl, y_atl, marker='', color='g', linewidth=2,label="ATL")
plt.plot(dts_chi, y_chi, marker='', color='b', linewidth=2,label="CHI")
plt.plot(dts_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot(dts_lax, y_lax, marker='', color='m', linewidth=2,label="LAX")
plt.plot(dts_phl, y_phl, marker='', color='y', linewidth=2,label="PHL")
plt.plot(dts_sea, y_sea, marker='', color='k', linewidth=2,label="SEA")
plt.title('Diesel Truck Stop Actual Price Per Gallon Time Series',fontsize='xx-large')
plt.ylabel('Diesel Truck Stop Actual Price Per Gallon',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### HAUL

# In[90]:


HAUL=FreightWaves[FreightWaves['ticker']=='HAUL']


# In[91]:


set(HAUL['granularity1'])


# In[92]:


#per = HAUL.data_timestamp.dt.to_period("D")


# In[93]:


HAUL_value=HAUL.groupby(['granularity1','data_timestamp']).mean()


# In[94]:


HAUL_value=HAUL_value.reset_index()


# In[95]:


haul_atl=list(HAUL_value[HAUL_value['granularity1']=='ATL']['data_timestamp'])
haul_chi=list(HAUL_value[HAUL_value['granularity1']=='CHI']['data_timestamp'])
haul_dal=list(HAUL_value[HAUL_value['granularity1']=='DAL']['data_timestamp'])
haul_lax=list(HAUL_value[HAUL_value['granularity1']=='LAX']['data_timestamp'])
haul_phl=list(HAUL_value[HAUL_value['granularity1']=='PHL']['data_timestamp'])
haul_sea=list(HAUL_value[HAUL_value['granularity1']=='SEA']['data_timestamp'])


# In[96]:


y_atl=HAUL_value[HAUL_value['granularity1']=='ATL']['data_value']
y_chi=HAUL_value[HAUL_value['granularity1']=='CHI']['data_value']
y_dal=HAUL_value[HAUL_value['granularity1']=='DAL']['data_value']
y_lax=HAUL_value[HAUL_value['granularity1']=='LAX']['data_value']
y_phl=HAUL_value[HAUL_value['granularity1']=='PHL']['data_value']
y_sea=HAUL_value[HAUL_value['granularity1']=='SEA']['data_value']


# In[97]:


fig10 = plt.figure(figsize=(25,10))
plt.plot(haul_atl, y_atl, marker='', color='g', linewidth=2,label="ATL")
plt.plot(haul_chi, y_chi, marker='', color='b', linewidth=2,label="CHI")
plt.plot(haul_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot(haul_lax, y_lax, marker='', color='m', linewidth=2,label="LAX")
plt.plot(haul_phl, y_phl, marker='', color='y', linewidth=2,label="PHL")
plt.plot(haul_sea, y_sea, marker='', color='k', linewidth=2,label="SEA")
plt.title('Haulhead Index Time Series',fontsize='xx-large')
plt.ylabel('Haulhead Index',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### INTRM

# In[98]:


INTRM=FreightWaves[FreightWaves['ticker']=='INTRM']


# In[99]:


set(INTRM['granularity1'])


# In[100]:


#per = INTRM.data_timestamp.dt.to_period("D")


# In[101]:


INTRM_value=INTRM.groupby(['granularity1','data_timestamp']).mean()


# In[102]:


INTRM_value=INTRM_value.reset_index()


# In[103]:


intrm_van=list(INTRM_value[INTRM_value['granularity1']=='ATLLIN']['data_timestamp'])
intrm_vca=list(INTRM_value[INTRM_value['granularity1']=='CHIATL']['data_timestamp'])
intrm_vdl=list(INTRM_value[INTRM_value['granularity1']=='DALLAX']['data_timestamp'])
intrm_vld=list(INTRM_value[INTRM_value['granularity1']=='LAXDAL']['data_timestamp'])
intrm_vls=list(INTRM_value[INTRM_value['granularity1']=='LAXSEA']['data_timestamp'])
intrm_vnc=list(INTRM_value[INTRM_value['granularity1']=='LINCHI']['data_timestamp'])
intrm_vsl=list(INTRM_value[INTRM_value['granularity1']=='SEALAX']['data_timestamp'])


# In[104]:


y_van=INTRM_value[INTRM_value['granularity1']=='ATLLIN']['data_value']
y_vca=INTRM_value[INTRM_value['granularity1']=='CHIATL']['data_value']
y_vdl=INTRM_value[INTRM_value['granularity1']=='DALLAX']['data_value']
y_vld=INTRM_value[INTRM_value['granularity1']=='LAXDAL']['data_value']
y_vls=INTRM_value[INTRM_value['granularity1']=='LAXSEA']['data_value']
y_vnc=INTRM_value[INTRM_value['granularity1']=='LINCHI']['data_value']
y_vsl=INTRM_value[INTRM_value['granularity1']=='SEALAX']['data_value']


# In[105]:


fig11 = plt.figure(figsize=(25,10))
plt.plot(intrm_van, y_van, marker='', color='g', linewidth=2,label="ATLLIN")
plt.plot(intrm_vca, y_vca, marker='', color='r', linewidth=2,label="CHIATL")
plt.plot(intrm_vdl, y_vdl, marker='', color='b', linewidth=2,label="DALLAX")
plt.plot(intrm_vld, y_vld, marker='', color='y', linewidth=2,label="LAXDAL")
plt.plot(intrm_vls, y_vls, marker='', color='c', linewidth=2,label="LAXSEA")
plt.plot(intrm_vnc, y_vnc, marker='', color='m', linewidth=2,label="LINCHI")
plt.plot(intrm_vsl, y_vsl, marker='', color='k', linewidth=2,label="SEALAX")
plt.title('Intermodal Rates Time Series',fontsize='xx-large')
plt.ylabel('Intermodal Rates ($/mile)',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### IPRO

# In[106]:


IPRO=FreightWaves[FreightWaves['ticker']=='IPRO']


# In[107]:


set(IPRO['granularity1'])


# In[108]:


IPRO_FBEVT=IPRO[IPRO['granularity1']=='FBEVT']


# In[109]:


IPRO_FBEVT=IPRO_FBEVT.sort_values(by='data_timestamp').reset_index(drop=True)


# In[110]:


IPRO_USA=IPRO[IPRO['granularity1']=='USA']


# In[111]:


IPRO_USA=IPRO_USA.sort_values(by='data_timestamp').reset_index(drop=True)


# In[112]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01']

fig12 = plt.figure(figsize=(25,10))
plt.plot(months, IPRO_FBEVT['data_value'], marker='o', color='g', linewidth=2,label="FBEVT")
plt.plot(months, IPRO_USA['data_value'], marker='o', color='r', linewidth=2,label="USA")
plt.title('Industrial Production Time Series',fontsize='xx-large')
plt.ylabel('Industrial Production',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### ISM

# In[113]:


ISM=FreightWaves[FreightWaves['ticker']=='ISM']


# In[114]:


ISM=ISM.sort_values(by='data_timestamp').reset_index(drop=True)


# In[115]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01']

fig13 = plt.figure(figsize=(25,10))
plt.plot(months, ISM['data_value'], marker='o', color='b', linewidth=2)
plt.title('Monthly Institute of Supply Management Metrics',fontsize='xx-large')
plt.ylabel('Institute of Supply Management Metrics',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')


# ### ORAIL53L

# In[116]:


ORAIL53L=FreightWaves[FreightWaves['ticker']=='ORAIL53L']


# In[117]:


set(ORAIL53L['granularity1'])


# In[118]:


#per = ORAIL53L.data_timestamp.dt.to_period("D")


# In[119]:


ORAIL53L_value=ORAIL53L.groupby(['granularity1','data_timestamp']).mean()


# In[120]:


ORAIL53L_value=ORAIL53L_value.reset_index()


# In[121]:


orail53l_vap=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='ATLPHL']['data_timestamp'])
orail53l_vca=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='CHIATL']['data_timestamp'])
orail53l_vdl=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='DALLAX']['data_timestamp'])
orail53l_vld=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='LAXDAL']['data_timestamp'])
orail53l_vls=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='LAXSEA']['data_timestamp'])
orail53l_vpc=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='PHLCHI']['data_timestamp'])
orail53l_vsl=list(ORAIL53L_value[ORAIL53L_value['granularity1']=='SEALAX']['data_timestamp'])


# In[122]:


y_vap=ORAIL53L_value[ORAIL53L_value['granularity1']=='ATLPHL']['data_value']
y_vca=ORAIL53L_value[ORAIL53L_value['granularity1']=='CHIATL']['data_value']
y_vdl=ORAIL53L_value[ORAIL53L_value['granularity1']=='DALLAX']['data_value']
y_vld=ORAIL53L_value[ORAIL53L_value['granularity1']=='LAXDAL']['data_value']
y_vls=ORAIL53L_value[ORAIL53L_value['granularity1']=='LAXSEA']['data_value']
y_vpc=ORAIL53L_value[ORAIL53L_value['granularity1']=='PHLCHI']['data_value']
y_vsl=ORAIL53L_value[ORAIL53L_value['granularity1']=='SEALAX']['data_value']


# In[123]:


fig14 = plt.figure(figsize=(25,10))
plt.plot(orail53l_vap, y_vap, marker='', color='g', linewidth=2,label="ATLPHL")
plt.plot(orail53l_vca, y_vca, marker='', color='r', linewidth=2,label="CHIATL")
plt.plot(orail53l_vdl, y_vdl, marker='', color='b', linewidth=2,label="DALLAX")
plt.plot(orail53l_vld, y_vld, marker='', color='y', linewidth=2,label="LAXDAL")
plt.plot(orail53l_vls, y_vls, marker='', color='c', linewidth=2,label="LAXSEA")
plt.plot(orail53l_vpc, y_vpc, marker='', color='m', linewidth=2,label="PHLCHI")
plt.plot(orail53l_vsl, y_vsl, marker='', color='k', linewidth=2,label="SEALAX")
plt.title("Outbound Rail Volume 53' Containers (Loaded) Time Series",fontsize='xx-large')
plt.ylabel("Outbound Rail Volume 53' Containers (Loaded)",fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### ORDERS

# In[124]:


ORDERS=FreightWaves[FreightWaves['ticker']=='ORDERS']


# In[125]:


ORDERS=ORDERS.sort_values(by='data_timestamp').reset_index(drop=True)


# In[126]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12']

fig15 = plt.figure(figsize=(25,10))
plt.plot(months, ORDERS['data_value'], marker='o', color='k', linewidth=2)
plt.title('Monthly CL8 Trucks Orders',fontsize='xx-large')
plt.ylabel('CL8 Trucks Orders',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')


# ### OTRI

# In[127]:


OTRI=FreightWaves[FreightWaves['ticker']=='OTRI']


# In[128]:


set(OTRI['granularity1'])


# In[129]:


#per = OTRI.data_timestamp.dt.to_period("D")


# In[130]:


OTRI_value=OTRI.groupby(['granularity1','data_timestamp']).mean()


# In[131]:


OTRI_value=OTRI_value.reset_index()


# In[132]:


otri_atl=list(OTRI_value[OTRI_value['granularity1']=='ATL']['data_timestamp'])
otri_chi=list(OTRI_value[OTRI_value['granularity1']=='CHI']['data_timestamp'])
otri_dal=list(OTRI_value[OTRI_value['granularity1']=='DAL']['data_timestamp'])
otri_lax=list(OTRI_value[OTRI_value['granularity1']=='LAX']['data_timestamp'])
otri_phl=list(OTRI_value[OTRI_value['granularity1']=='PHL']['data_timestamp'])
otri_sea=list(OTRI_value[OTRI_value['granularity1']=='SEA']['data_timestamp'])


# In[133]:


y_atl=OTRI_value[OTRI_value['granularity1']=='ATL']['data_value']
y_chi=OTRI_value[OTRI_value['granularity1']=='CHI']['data_value']
y_dal=OTRI_value[OTRI_value['granularity1']=='DAL']['data_value']
y_lax=OTRI_value[OTRI_value['granularity1']=='LAX']['data_value']
y_phl=OTRI_value[OTRI_value['granularity1']=='PHL']['data_value']
y_sea=OTRI_value[OTRI_value['granularity1']=='SEA']['data_value']


# In[134]:


fig16 = plt.figure(figsize=(25,10))
plt.plot(otri_atl, y_atl, marker='', color='g', linewidth=2,label="ATL")
plt.plot(otri_chi, y_chi, marker='', color='b', linewidth=2,label="CHI")
plt.plot(otri_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot(otri_lax, y_lax, marker='', color='m', linewidth=2,label="LAX")
plt.plot(otri_phl, y_phl, marker='', color='y', linewidth=2,label="PHL")
plt.plot(otri_sea, y_sea, marker='', color='k', linewidth=2,label="SEA")
plt.title('Outbound Tender Reject Index Time Series',fontsize='xx-large')
plt.ylabel('Outbound Tender Reject Index',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### OTVI

# In[135]:


OTVI=FreightWaves[FreightWaves['ticker']=='OTVI']


# In[136]:


set(OTVI['granularity1'])


# In[137]:


#per = OTVI.data_timestamp.dt.to_period("D")


# In[138]:


OTVI_value=OTVI.groupby(['granularity1','data_timestamp']).mean()


# In[139]:


OTVI_value=OTVI_value.reset_index()


# In[140]:


otvi_atl=list(OTVI_value[OTVI_value['granularity1']=='ATL']['data_timestamp'])
otvi_chi=list(OTVI_value[OTVI_value['granularity1']=='CHI']['data_timestamp'])
otvi_dal=list(OTVI_value[OTVI_value['granularity1']=='DAL']['data_timestamp'])
otvi_lax=list(OTVI_value[OTVI_value['granularity1']=='LAX']['data_timestamp'])
otvi_phl=list(OTVI_value[OTVI_value['granularity1']=='PHL']['data_timestamp'])
otvi_sea=list(OTVI_value[OTVI_value['granularity1']=='SEA']['data_timestamp'])


# In[141]:


y_atl=OTVI_value[OTVI_value['granularity1']=='ATL']['data_value']
y_chi=OTVI_value[OTVI_value['granularity1']=='CHI']['data_value']
y_dal=OTVI_value[OTVI_value['granularity1']=='DAL']['data_value']
y_lax=OTVI_value[OTVI_value['granularity1']=='LAX']['data_value']
y_phl=OTVI_value[OTVI_value['granularity1']=='PHL']['data_value']
y_sea=OTVI_value[OTVI_value['granularity1']=='SEA']['data_value']


# In[142]:


fig17 = plt.figure(figsize=(25,10))
plt.plot(otvi_atl, y_atl, marker='', color='g', linewidth=2,label="ATL")
plt.plot(otvi_chi, y_chi, marker='', color='b', linewidth=2,label="CHI")
plt.plot(otvi_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot(otvi_lax, y_lax, marker='', color='m', linewidth=2,label="LAX")
plt.plot(otvi_phl, y_phl, marker='', color='y', linewidth=2,label="PHL")
plt.plot(otvi_sea, y_sea, marker='', color='k', linewidth=2,label="SEA")
plt.title('Inbound Tender Reject Index Time Series',fontsize='xx-large')
plt.ylabel('Inbound Tender Reject Index',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### PPI

# In[143]:


PPI=FreightWaves[FreightWaves['ticker']=='PPI']


# In[144]:


PPI=PPI.sort_values(by='data_timestamp').reset_index(drop=True)


# In[145]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12']

fig18 = plt.figure(figsize=(25,10))
plt.plot(months, PPI['data_value'], marker='o', color='k', linewidth=2)
plt.title('Monthly Producer Price Index',fontsize='xx-large')
plt.ylabel('Producer Price Index',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')


# ### RESLG

# In[146]:


RESLG=FreightWaves[FreightWaves['ticker']=='RESLG']


# In[147]:


RESLG=RESLG.sort_values(by='data_timestamp').reset_index(drop=True)


# In[148]:


months=['2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01']

fig19 = plt.figure(figsize=(25,10))
plt.plot(months, RESLG['data_value'], marker='o', color='k', linewidth=2)
plt.title('Monthly Retail Sales YoY Change',fontsize='xx-large')
plt.ylabel('Retail Sales YoY Change',fontsize='xx-large')
plt.xlabel('Date (Year-Month)',fontsize='xx-large')


# ### TLT

# In[149]:


TLT=FreightWaves[FreightWaves['ticker']=='TLT']


# In[150]:


set(TLT['granularity1'])


# In[151]:


#per = TLT.data_timestamp.dt.to_period("D")


# In[152]:


TLT_value=TLT.groupby(['granularity1','data_timestamp']).mean()


# In[153]:


TLT_value=TLT_value.reset_index()


# In[154]:


tlt_atl=list(TLT_value[TLT_value['granularity1']=='ATL']['data_timestamp'])
tlt_chi=list(TLT_value[TLT_value['granularity1']=='CHI']['data_timestamp'])
tlt_dal=list(TLT_value[TLT_value['granularity1']=='DAL']['data_timestamp'])
tlt_lax=list(TLT_value[TLT_value['granularity1']=='LAX']['data_timestamp'])
tlt_phl=list(TLT_value[TLT_value['granularity1']=='PHL']['data_timestamp'])
tlt_sea=list(TLT_value[TLT_value['granularity1']=='SEA']['data_timestamp'])


# In[155]:


x_atl=[]
x_chi=[]
x_dal=[]
x_lax=[]
x_phl=[]
x_sea=[]

for i in range(len(tlt_atl)):
    x_atl.append(tlt_atl[i].strftime('%Y-%m-%d'))

for i in range(len(tlt_chi)):
    x_chi.append(tlt_chi[i].strftime('%Y-%m-%d'))
    
for i in range(len(tlt_dal)):
    x_dal.append(tlt_dal[i].strftime('%Y-%m-%d'))
    
for i in range(len(tlt_lax)):
    x_lax.append(tlt_lax[i].strftime('%Y-%m-%d'))
    
for i in range(len(tlt_phl)):
    x_phl.append(tlt_phl[i].strftime('%Y-%m-%d'))
    
for i in range(len(tlt_sea)):
    x_sea.append(tlt_sea[i].strftime('%Y-%m-%d'))


# In[156]:


y_atl=TLT_value[TLT_value['granularity1']=='ATL']['data_value']
y_chi=TLT_value[TLT_value['granularity1']=='CHI']['data_value']
y_dal=TLT_value[TLT_value['granularity1']=='DAL']['data_value']
y_lax=TLT_value[TLT_value['granularity1']=='LAX']['data_value']
y_phl=TLT_value[TLT_value['granularity1']=='PHL']['data_value']
y_sea=TLT_value[TLT_value['granularity1']=='SEA']['data_value']


# In[157]:


fig20 = plt.figure(figsize=(25,10))
plt.plot(tlt_atl, y_atl, marker='', color='g', linewidth=2,label="ATL")
plt.plot(tlt_chi, y_chi, marker='', color='b', linewidth=2,label="CHI")
plt.plot(tlt_dal, y_dal, marker='', color='r', linewidth=2,label="DAL")
plt.plot(tlt_lax, y_lax, marker='', color='m', linewidth=2,label="LAX")
plt.plot(tlt_phl, y_phl, marker='', color='y', linewidth=2,label="PHL")
plt.plot(tlt_sea, y_sea, marker='', color='k', linewidth=2,label="SEA")
plt.title('Tender Lead Time Time Series',fontsize='xx-large')
plt.ylabel('Tender Lead Time',fontsize='xx-large')
plt.xlabel('Date (Year-Month-Day)',fontsize='xx-large')
plt.legend(fontsize='large')


# ### Rates

# ### Preprocessing

# In[158]:


Q5=Q5.drop(columns=['Pickup Date','Delivery Date','Origin State','Dest State','Revenue'])


# In[159]:


Q5=Q5.replace('Atlanta', 'ATL')
Q5=Q5.replace('Chicago', 'CHI')
Q5=Q5.replace('Dallas', 'DAL')
Q5=Q5.replace('Philadelphia', 'PHL')
Q5=Q5.replace('Seattle', 'SEA')
Q5=Q5.replace('Los Angeles', 'LAX')


# In[160]:


Q5.head()


# In[161]:


sum(Q5['RPM']>20)/len(Q5)


# In[162]:


fig21 = plt.figure(figsize=(25,10))
plt.scatter(Q5[Q5['Origin Market']=='ATL']['Book Date'], Q5[Q5['Origin Market']=='ATL']['RPM'],color='r',label='ATL')
plt.scatter(Q5[Q5['Origin Market']=='CHI']['Book Date'], Q5[Q5['Origin Market']=='CHI']['RPM'],color='b',label='CHI')
plt.scatter(Q5[Q5['Origin Market']=='DAL']['Book Date'], Q5[Q5['Origin Market']=='DAL']['RPM'],color='y',label='DAL')
plt.scatter(Q5[Q5['Origin Market']=='PHL']['Book Date'], Q5[Q5['Origin Market']=='PHL']['RPM'],color='g',label='PHL')
plt.scatter(Q5[Q5['Origin Market']=='LAX']['Book Date'], Q5[Q5['Origin Market']=='LAX']['RPM'],color='k',label='LAX')
plt.scatter(Q5[Q5['Origin Market']=='SEA']['Book Date'], Q5[Q5['Origin Market']=='SEA']['RPM'],color='m',label='SEA')
plt.ylim(0, 20)
plt.title('Evolution of Rates Per Mile over Time', fontsize='xx-large')
plt.ylabel('Rates Per Mile', fontsize='xx-large')
plt.xlabel('Date (Year-Month)', fontsize='xx-large')
plt.show()


# In[163]:


FreightWaves.head()


# ### Feature Engineering

# In[164]:


set(FreightWaves['ticker'])


# ### macro_fn (Macro function moved to be a part of the larger function for sake of efficiency.)
# 
#     Function which takes in a timestamp for a Shipper Book date and a Macro Indicator df and returns the data value for the Macro Indicator that the shipper would've referenced to make their rate decision (i.e. the most recent reporting of the macro indicator)
#     
#    #### Relevant macro indicators: ISM, IRPO_USA, IPRO_FBEVT, PPI, COSP, RESLG, and ORDERS

# ### the_new_min_function
# 
#     Takes in a city, a day (timestamp), and a number of days to look back (int). It returns a list of index values that are at earliest the number of days back from the given day. 
#     
#     Example:
#     
#     f('LAX', Timestamp('2018-02-25 00:00:00')) -> (12.1, 1500, 13.5, 0, 0, 5.4)
#     
#    #### Relevant indecies include: OTVI, OTRI, HAUL, TLT, DTS, and CSTM

# In[165]:


otvi_to_val = {}
for i in zip(list(OTVI_value.granularity1),list(OTVI_value.data_timestamp), list(OTVI_value.data_value)):
    otvi_to_val[(i[0], i[1])] = i[2]

otri_to_val = {}
for i in zip(list(OTRI_value.granularity1),list(OTRI_value.data_timestamp), list(OTRI_value.data_value)):
    otri_to_val[(i[0], i[1])] = i[2] 

haul_to_val = {}
for i in zip(list(HAUL_value.granularity1),list(HAUL_value.data_timestamp), list(HAUL_value.data_value)):
    haul_to_val[(i[0], i[1])] = i[2] 

tlt_to_val = {}
for i in zip(list(TLT_value.granularity1),list(TLT_value.data_timestamp), list(TLT_value.data_value)):
    tlt_to_val[(i[0], i[1])] = i[2] 

dts_to_val = {}
for i in zip(list(DTS_value.granularity1),list(DTS_value.data_timestamp), list(DTS_value.data_value)):
    dts_to_val[(i[0], i[1])] = i[2] 

cstm_to_val = {}
for i in zip(list(CSTM_value.granularity1),list(CSTM_value.data_timestamp), list(CSTM_value.data_value)):
    cstm_to_val[(i[0], i[1])] = i[2]


# In[166]:


def the_new_min_function(city: str, time: pd.Timestamp, days_back) -> list:
    cur = time
    days_back += 1
    otvi = np.nan
    otri = np.nan
    haul = np.nan
    tlt = np.nan
    dts = np.nan
    cstm = np.nan

    for i in range(days_back):
        if (city, cur) in otvi_to_val:
            otvi = otvi_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    for i in range(days_back):
        if (city, cur) in otri_to_val:
            otri = otri_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    for i in range(days_back):
        if (city, cur) in haul_to_val:
            haul = haul_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    for i in range(days_back):
        if (city, cur) in tlt_to_val:
            tlt = tlt_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    for i in range(days_back):
        if (city, cur) in dts_to_val:
            dts = dts_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    for i in range(days_back):
        if (city, cur) in cstm_to_val:
            cstm = cstm_to_val[(city, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time

    return [otvi, otri, haul, tlt, dts, cstm]


# ### the_new_lane_function
# 
#     Takes in a lane, a day (timestamp), and a number of days to look back (int). It returns a list of index values that are at earliest the number of days back from the given day. 
#     
#     Example:
#     
#     f('VLS', Timestamp('2018-02-25 00:00:00')) -> (12.1, 1500, 13.5, 0, 0, 5.4)
#     
#    #### Relevant indecies include: DATVF, ORAIL53L, and INTRM,

# In[167]:


ltc = {}
ltc['Other'] = np.nan
ltc['VLS'] = 'LAXSEA'
ltc['VSL'] = 'SEALAX'
ltc['VLD'] = 'LAXDAL'
ltc['VDL'] = 'DALLAX'
ltc['VCA'] = 'CHIATL'
ltc['VAP'] = 'ATLPHL'
ltc['VPC'] = 'PHLCHI'


# In[168]:


datvf_to_val = {}
for i in zip(list(DATVF_value.granularity1),list(DATVF_value.data_timestamp), list(DATVF_value.data_value)):
    datvf_to_val[(i[0], i[1])] = i[2]
    
orail53l_to_val = {}
for i in zip(list(ORAIL53L_value.granularity1),list(ORAIL53L_value.data_timestamp), list(ORAIL53L_value.data_value)):
    orail53l_to_val[(i[0], i[1])] = i[2]
    
intrm_to_val = {}
for i in zip(list(INTRM_value.granularity1),list(INTRM_value.data_timestamp), list(INTRM_value.data_value)):
    intrm_to_val[(i[0], i[1])] = i[2]


# In[169]:


def the_new_lane_function(lane:str, time: pd.Timestamp, days_back:int) -> list:
    lane = ltc[lane]
    if lane == np.nan:
        return [np.nan, np.nan, np.nan]

    cur = time
    days_back += 1
    datv = np.nan
    orai = np.nan
    intr = np.nan

    for i in range(days_back):
        if (lane, cur) in datvf_to_val:
            datv = datvf_to_val[(lane, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time 

    for i in range(days_back):
        if (lane, cur) in orail53l_to_val:
            orai = orail53l_to_val[(lane, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time 

    for i in range(days_back):
        if (lane, cur) in intrm_to_val:
            intr = intrm_to_val[(lane, cur)]
            break
        cur = time - pd.Timedelta(days=i)
    cur = time 

    return [datv, orai, intr]


# ### Preprocessing notes:
#     - Take out all rows with book dates before 2018-02-28, or the last indicator date, so that there should be no NaN values in the df
#     - Remove all rows where the book date is after the delivery date

# In[170]:


def pred_df(df):
    pred_df=[]
    rates=list(df['RPM'])
    contract=list(df['Spot/Contract'])
    load_type = list(df['Load Type'])
    mileage = list(df['Mileage'])
    dist = []

    cities = list(df['Origin Market'])
    times = list(df['Book Date'])
    
    lanes = list(df['Lane'])

    ism_values = ISM['data_value']
    ipro_usa_values=IPRO_USA['data_value']
    ipro_fbevt_values=IPRO_FBEVT['data_value']
    ppi_values=PPI['data_value']
    cosp_values=COSP['data_value']
    reslg_values=RESLG['data_value']
    orders_values=ORDERS['data_value']

    for i in range(len(df)):    
        
        #dist.append(to_dist_ind(mileage[i]))
        if mileage[i]=='(blank)' or mileage[i]<100:
            dist.append('C')
        elif mileage[i] <= 250:
            dist.append('S')
        elif mileage[i] <= 450:
            dist.append('M')
        elif mileage[i] <= 800:
            dist.append('T')
        else:
            dist.append('L')
        
        # Macro:
        date = df.loc[i]['Book Date']

        dates1=list(PPI['data_timestamp'])
        dates1.insert(0,date)
        dates1.sort()
        index1 = dates1.index(date)

        dates2=list(ISM['data_timestamp'])
        dates2.insert(0,date)
        dates2.sort()
        index2 = dates2.index(date)

        if index1 > 0 and index2 > 0:
            ism=ism_values[index2-1]
            ipro_usa=ipro_usa_values[index2-1]
            ipro_fbevt=ipro_fbevt_values[index2-1]
            ppi=ppi_values[index1-1]
            cosp=cosp_values[index1-1]
            reslg=reslg_values[index2-1]
            orders=orders_values[index1-1]
        else:
            ism=np.nan
            ipro_usa=np.nan
            ipro_fbevt=np.nan
            ppi=np.nan
            cosp=np.nan
            reslg=np.nan
            orders=np.nan

        # Micro Location Data     
        #loc = df.loc[i]['Origin Market']

        city = cities[i]
        time = times[i] - pd.Timedelta(days=1)
        days_back=5

        micro_loc = the_new_min_function(city, time, days_back)
        otvi=micro_loc[0]
        otri=micro_loc[1]
        haul=micro_loc[2]
        tlt=micro_loc[3]
        dts=micro_loc[4]
        cstm=micro_loc[5]

        lane = lanes[i]

        micro_lane = the_new_lane_function(lane, time, days_back)
        datvf=micro_lane[0]
        orail53l=micro_lane[1]
        intrm=micro_lane[2]

        pred_df.append([date,dist[i],contract[i],load_type[i],ism,ipro_usa,ipro_fbevt,ppi,cosp,reslg,orders,otvi,otri,haul,tlt,dts,cstm,datvf,orail53l,intrm,rates[i]])
    
    return pred_df;


# In[171]:


pred_df=pd.DataFrame(pred_df(Q5),columns=['date','dist','contract','load_type','ism','ipro_usa','ipro_fbevt','ppi','cosp','reslg','orders','otvi','otri','haul','tlt','dts','cstm','datvf','orail53l','intrm','RPM'])


# In[172]:


pred_df.head()


# ## Taking out dates before 3-13-2018

# In[173]:


from datetime import *
new_df=pred_df[pred_df['date'] > date(2018,3,12)]


# ## Lanes

# In[174]:


df=pred_df.dropna()


# In[175]:


len(df)


# In[176]:


df=df.drop(columns=['date','dist'])


# In[177]:


df=pd.get_dummies(df, prefix=['contract', 'load_type'])


# In[178]:


df.head()


# In[179]:


y = df.loc[:, df.columns == 'RPM']
X = df.loc[:, df.columns != 'RPM']


# ### Correlation Matrix

# In[180]:


#Using Pearson Correlation
fig26=plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[181]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features


# Nothing is very highly correlated with RPM, the dependent variable. However, certain independent variables are highly correlated with one another.

# ### Recursive Feature Elimination

# In[182]:


y_ext = df.loc[:, df.columns == 'RPM']
X_ext = df.iloc[:,0:16]


# In[183]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[184]:


X_ext.columns


# ### RFE selects IPRO_USA, DATVF, and TLT when given only external factors

# In[185]:


print(df[['ipro_usa','tlt','datvf']].corr())


# The correlation between these external factors is not particularly high.

# In[186]:


fig21.savefig('RPM.png')
fig21


# In[187]:


# TLT
fig20.savefig('TLT.png')
fig20


# In[188]:


# IPRO
fig12.savefig('IPRO.png')
fig12


# In[189]:


# DATVF
fig8.savefig('DATVF.png')
fig8


# ## Question 6
# 
#     What other insights can you provide a shipper based on these data sets?

# In[190]:


y = df.loc[:, df.columns == 'RPM']
X = df.loc[:, df.columns != 'RPM']


# In[191]:


len(X.columns)


# #### RFE selects all Load Type categorical variables

# In[192]:


#no of features
nof_list=np.arange(1,25)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[193]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 23)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[194]:


predictors=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'otvi', 'otri',
       'haul', 'tlt', 'dts', 'cstm', 'datvf', 'orail53l', 'intrm',
       'contract_Contract', 'contract_Spot', 'load_type_Ad-hoc',
       'load_type_Consistent', 'load_type_Critical', 'load_type_Dedicated',
       'load_type_Standard', 'load_type_White Glove']
X_l=X[predictors]
y_l=y['RPM']


# In[195]:


X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size=0.2, random_state=0)


# In[196]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[197]:


print(model.intercept_)


# In[198]:


print(model.coef_)


# In[199]:


y_pred = model.predict(X_test)


# In[200]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[201]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[202]:


model.score(X_test,y_test)


# ## Breaking into dfs based on distance:

# ### City Distance

# In[203]:


city_df=new_df[new_df['dist']=='C']


# In[204]:


city_df=city_df.drop(columns=['datvf','orail53l','intrm','date','dist']).dropna()


# In[205]:


Q5.loc[173]


# In[206]:


city_df=pd.get_dummies(city_df, prefix=['contract', 'load_type'])


# In[207]:


city_df.head()


# ### Correlation Matrix

# In[208]:


#Using Pearson Correlation
fig27=plt.figure(figsize=(12,10))
cor = city_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[209]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# Nothing is very highly correlated with RPM, the dependent variable. However, certain independent variables are highly correlated with one another.

# In[210]:


y_ext = city_df.loc[:, city_df.columns == 'RPM']
X_ext = city_df.iloc[:,0:13]


# ### Backward Elimination

# In[211]:


#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X_ext)
#Fitting sm.OLS model
model = sm.OLS(y_ext,X_1).fit()
model.pvalues


# In[212]:


#Backward Elimination
cols = list(X_ext.columns)
pmax = 1

while (len(cols)>0):
    p= []
    X_1 = X_ext[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_ext,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print(selected_features_BE)


# ### Recursive Feature Elimination

# In[213]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[214]:


X_ext.columns


# #### RFE selects: IPRO_USA, TLT and DTS when given only external factors

# In[215]:


y = city_df.loc[:, city_df.columns == 'RPM']
X = city_df.loc[:, city_df.columns != 'RPM']


# In[216]:


#no of features
nof_list=np.arange(1,23)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[217]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 20)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[218]:


predictors=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'otvi', 'otri',
       'haul', 'tlt', 'dts', 'cstm', 'contract_Contract', 'contract_Spot',
       'load_type_Ad-hoc', 'load_type_Consistent', 'load_type_Critical',
       'load_type_Dedicated', 'load_type_Standard', 'load_type_White Glove']
X_c=X[predictors]
y_c=y['RPM']


# In[219]:


X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=0)


# In[220]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[221]:


print(model.intercept_)


# In[222]:


print(model.coef_)


# In[223]:


y_pred = model.predict(X_test)


# In[224]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[225]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[226]:


model.score(X_test,y_test)


# ### Polynomial Regression

# In[227]:


from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(X)


# In[228]:


X_poly.size/len(X_poly)


# In[229]:


########### Don't run this, it takes too long. Just trust that the following variables were outputted ###########
#no of features
#nof_list=np.arange(1,254)            
#high_score=0

#Variable to store the optimum features
#nof=0           
#score_list =[]
#for n in range(len(nof_list)):
    #X_train, X_test, y_train, y_test = train_test_split(X_poly,y, test_size = 0.5, random_state = 0)
    #model = LinearRegression()
    #rfe = RFE(model,nof_list[n])
    #X_train_rfe = rfe.fit_transform(X_train,y_train)
    #X_test_rfe = rfe.transform(X_test)
    #model.fit(X_train_rfe,y_train)
    #score = model.score(X_test_rfe,y_test)
    #score_list.append(score)
    #if(score>high_score):
        #high_score = score
        #nof = nof_list[n]
#print("Optimum number of features: %d" %nof)
#print("Score with %d features: %f" % (nof, high_score))


# In[230]:


X_poly=pd.DataFrame(X_poly)


# In[231]:


#cols = list(X_poly.columns)
#model = LinearRegression()

#Initializing RFE model
#rfe = RFE(model, 27)             

#Transforming data using RFE
#X_rfe = rfe.fit_transform(X_poly,y)  

#Fitting the data to model
#model.fit(X_rfe,y)              
#temp = pd.Series(rfe.support_,index = cols)
#selected_features_rfe = temp[temp==True].index
#print(selected_features_rfe)


# In[232]:


predictors=[  2,  11,  14,  15,  16,  17,  19,  20,  55,  56, 192, 193, 195,
            196, 217, 221, 224, 225, 226, 227, 228, 229, 230, 232, 238, 247,
            250]
X_poly_preds=X_poly[predictors]
y_poly=y['RPM']


# In[233]:


X_train, X_test, y_train, y_test = train_test_split(X_poly_preds, y_poly, test_size=0.2, random_state=0)


# In[234]:


poly.fit(X_train, y_train) 
model2 = LinearRegression() 
results2=model2.fit(X_train, y_train)


# In[235]:


print(model2.intercept_)


# In[236]:


print(model2.coef_)


# In[237]:


y_pred = model2.predict(X_test)


# In[238]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[239]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[240]:


model2.score(X_test,y_test)


# #### Polynomial regression with poly = 2 provides only minor gains compared to linear regression

# ### Trying Log transformation of RPM data

# ### Quick Linear Regression Model

# In[241]:


predictors1=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'orders',
       'otvi', 'tlt', 'dts','cstm'] 
predictors2=['contract_Contract',
       'contract_Spot', 'load_type_Ad-hoc', 'load_type_Consistent',
       'load_type_Critical', 'load_type_Dedicated', 'load_type_Standard',
       'load_type_White Glove']
X_log1=np.log(X[predictors1])
X_log2=X[predictors2]
X_log=pd.concat([X_log1,X_log2],axis=1)
y_log=y['RPM']


# In[242]:


X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=0)


# In[243]:


model_log = LinearRegression()
result=model_log.fit(X_train,y_train)


# In[244]:


print(model_log.intercept_)


# In[245]:


print(model_log.coef_)


# In[246]:


y_pred = model_log.predict(X_test)


# In[247]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[248]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# #### Log transform performs worse than the other two models

# ## Short Distance

# In[249]:


short_df = new_df[new_df['dist']=='S']


# In[250]:


short_df=short_df.drop(columns=['datvf','orail53l','intrm','date','dist']).dropna()


# In[251]:


short_df=pd.get_dummies(short_df, prefix=['contract', 'load_type'])


# In[252]:


short_df.head()


# ### Correlation Matrix

# In[253]:


#Using Pearson Correlation
fig22=plt.figure(figsize=(12,10))
cor = short_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[254]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# Nothing is very highly correlated with RPM, the dependent variable. However, certain independent variables are highly correlated with one another.

# ### Recursive Feature Elimination

# In[255]:


y_ext = short_df.loc[:, short_df.columns == 'RPM']
X_ext = short_df.iloc[:,0:13]


# In[256]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[257]:


X_ext.columns


# #### RFE selects IPRO_FBEVT, TLT, and DTS when given only external factors

# In[258]:


y = short_df.loc[:, short_df.columns == 'RPM']
X = short_df.loc[:, short_df.columns != 'RPM']


# In[259]:


#no of features
nof_list=np.arange(1,23)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[260]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 21)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[261]:


predictors=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'orders',
       'otvi', 'otri', 'haul', 'tlt', 'dts', 'cstm', 'contract_Contract',
       'contract_Spot', 'load_type_Ad-hoc', 'load_type_Consistent',
       'load_type_Critical', 'load_type_Dedicated', 'load_type_Standard',
       'load_type_White Glove']
X_s=X[predictors]
y_s=y['RPM']


# In[262]:


X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=0)


# In[263]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[264]:


print(model.intercept_)


# In[265]:


print(model.coef_)


# In[266]:


y_pred = model.predict(X_test)


# In[267]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[268]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[269]:


model.score(X_test,y_test)


# ### Middle Distance

# In[270]:


mid_df=new_df[new_df['dist']=='M']


# In[271]:


mid_df=mid_df.drop(columns=['datvf','orail53l','intrm','date','dist']).dropna()


# In[272]:


mid_df=pd.get_dummies(mid_df, prefix=['contract', 'load_type'])


# In[273]:


mid_df.head()


# ### Correlation Matrix

# In[274]:


#Using Pearson Correlation
fig23=plt.figure(figsize=(12,10))
cor = mid_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[275]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features


# Nothing is very highly correlated with RPM, the dependent variable. However, certain independent variables are highly correlated with one another.

# ### Recursive Feature Elimination

# In[276]:


y_ext = mid_df.loc[:, mid_df.columns == 'RPM']
X_ext = mid_df.iloc[:,0:13]


# In[277]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[278]:


X_ext.columns


# #### RFE selects IPRO_USA, TLT, and DTS when given only external factors

# In[279]:


y = mid_df.loc[:, mid_df.columns == 'RPM']
X = mid_df.loc[:, mid_df.columns != 'RPM']


# In[280]:


#no of features
nof_list=np.arange(1,23)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[281]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 21)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[282]:


predictors=['ipro_usa', 'tlt', 'dts', 'cstm', 'contract_Contract',
       'contract_Spot', 'load_type_Ad-hoc', 'load_type_Consistent',
       'load_type_Critical', 'load_type_Dedicated', 'load_type_Standard',
       'load_type_White Glove']
X_s=X[predictors]
y_s=y['RPM']


# In[283]:


X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=0)


# In[284]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[285]:


print(model.intercept_)


# In[286]:


print(model.coef_)


# In[287]:


y_pred = model.predict(X_test)


# In[288]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[289]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[290]:


model.score(X_test,y_test)


# ### Tweener Distance

# In[291]:


tween_df=new_df[new_df['dist']=='T']


# In[292]:


tween_df.head()


# In[293]:


tween_nonlane=tween_df[tween_df['datvf'].isna()]


# In[294]:


tween_lane=tween_df[tween_df['datvf'].notna()]


# In[295]:


tween_nonlane=tween_nonlane.drop(columns=['datvf','orail53l','intrm','date','dist']).dropna()


# In[296]:


tween_nonlane=pd.get_dummies(tween_nonlane, prefix=['contract', 'load_type'])


# In[297]:


tween_nonlane.head()


# ### Correlation Matrix

# In[298]:


#Using Pearson Correlation
fig24=plt.figure(figsize=(12,10))
cor = tween_nonlane.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[299]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features


# Nothing is very highly correlated with RPM, the dependent variable. However, certain independent variables are highly correlated with one another.

# ### Recursive Feature Elimination

# In[300]:


y_ext = tween_nonlane.loc[:, tween_nonlane.columns == 'RPM']
X_ext = tween_nonlane.iloc[:,0:13]


# In[301]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[302]:


X_ext.columns


# #### RFE selects IPRO_USA, TLT, and DTS when given only external factors

# In[303]:


y = tween_nonlane.loc[:, tween_nonlane.columns == 'RPM']
X = tween_nonlane.loc[:, tween_nonlane.columns != 'RPM']


# In[304]:


len(tween_nonlane.columns)


# #### RFE selects all Load Type categorical variables

# In[305]:


#no of features
nof_list=np.arange(1,23)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[306]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 21)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[307]:


predictors=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'orders',
       'otvi', 'otri', 'haul', 'tlt', 'dts', 'cstm', 'contract_Contract',
       'contract_Spot', 'load_type_Ad-hoc', 'load_type_Consistent',
       'load_type_Critical', 'load_type_Dedicated', 'load_type_Standard',
       'load_type_White Glove']
X_s=X[predictors]
y_s=y['RPM']


# In[308]:


X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=0)


# In[309]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[310]:


print(model.intercept_)


# In[311]:


print(model.coef_)


# In[312]:


y_pred = model.predict(X_test)


# In[313]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[314]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[315]:


model.score(X_test,y_test)


# ### Long Distance

# In[316]:


long_df=new_df[new_df['dist']=='L']


# In[317]:


long_df.head()


# In[318]:


long_nonlane=long_df[long_df['datvf'].isna()]


# In[319]:


long_lane=long_df[long_df['datvf'].notna()]


# In[320]:


long_nonlane=long_nonlane.drop(columns=['datvf','orail53l','intrm','date','dist']).dropna()


# In[321]:


long_nonlane=pd.get_dummies(long_nonlane, prefix=['contract', 'load_type'])


# In[322]:


long_nonlane.head()


# ### Correlation Matrix

# In[323]:


#Using Pearson Correlation
fig25=plt.figure(figsize=(12,10))
cor = long_nonlane.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[324]:


cor_target = abs(cor['RPM'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.4]
relevant_features


# In[325]:


print(long_nonlane[['contract_Contract','contract_Spot','load_type_Ad-hoc','load_type_Standard']].corr())


# Some variables are correlated highly with RPM, the dependent variable. However, these independent variables are highly correlated with one another.

# ### Recursive Feature Elimination

# In[326]:


y_ext = long_nonlane.loc[:, long_nonlane.columns == 'RPM']
X_ext = long_nonlane.iloc[:,0:13]


# In[327]:


model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 3)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_ext,y_ext)  

#Fitting the data to model
model.fit(X_rfe,y_ext)
print(rfe.support_)
print(rfe.ranking_)


# In[328]:


X_ext.columns


# #### RFE selects IPRO_USA, TLT, and DTS when given only external factors

# In[329]:


y = long_nonlane.loc[:, long_nonlane.columns == 'RPM']
X = long_nonlane.loc[:, long_nonlane.columns != 'RPM']


# In[330]:


len(long_nonlane.columns)


# #### RFE selects all Load Type categorical variables

# In[331]:


#no of features
nof_list=np.arange(1,23)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[332]:


cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 21)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### Quick Linear Regression Model

# In[333]:


predictors=['ism', 'ipro_usa', 'ipro_fbevt', 'ppi', 'cosp', 'reslg', 'orders',
       'otvi', 'otri', 'haul', 'tlt', 'dts', 'cstm', 'contract_Contract',
       'contract_Spot', 'load_type_Ad-hoc', 'load_type_Consistent',
       'load_type_Critical', 'load_type_Dedicated', 'load_type_Standard',
       'load_type_White Glove']
X_l=X[predictors]
y_l=y['RPM']


# In[334]:


X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size=0.2, random_state=0)


# In[335]:


model = LinearRegression()
result=model.fit(X_train,y_train)


# In[336]:


print(model.intercept_)


# In[337]:


print(model.coef_)


# In[338]:


y_pred = model.predict(X_test)


# In[339]:


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.head()


# In[340]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[341]:


model.score(X_test,y_test)


# In[ ]:




