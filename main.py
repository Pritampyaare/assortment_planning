# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style("darkgrid")
#plt.style.use('ggplot')

#import cufflinks as cf
#import plotly.express as px
#import plotly.offline as py
#from plotly.offline import plot
#import plotly.graph_objects as go
#import plotly.graph_objs as go

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'This is my first API call!'

@app.route('/post', methods=["POST"])
def testpost():
     input_json = request.get_json(force=True) 
     dictToReturn = {'text':input_json['text']}
     return jsonify(dictToReturn)

@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):
    from apyori import apriori
    import numpy as np
    import pandas as pd
    import os

    df = pd.read_csv("Groceries_dataset.csv")
    #df.head()

    #print("We have :", df.shape[0], "items in the Dataset")
    #print(f"And null values :\n{df.isnull().sum()}")

    #print(f"And nan values :\n{df.isna().sum()} ")

    #print(f'There are : {len(df["itemDescription"].unique())} unique items')

    df["Year"] = df["Date"].str.split("-").str[-1]
    df["Months_Year"] = df["Date"].str.split("-").str[1] + "-" + df["Date"].str.split("-").str[-1] 
    #df.head()

    #df["Months_Year"].unique()

    #Graph : Months/Year by count
    '''fig = px.bar(df["Months_Year"].value_counts(ascending=False), orientation="v", color=df["Months_Year"].value_counts(ascending=False), color_continuous_scale=px.colors.sequential.Plasma, 
                log_x=False, labels={'value':'Count', 
                                    'index':'Date',
                                    'color':'None'
                                    })

    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="green",
        title_text="Date by count"
    )

    #fig.show()

    #Graph : Item by count
    fig = px.bar(df["itemDescription"].value_counts()[:20], orientation="v", color=df["itemDescription"].value_counts()[:20], color_continuous_scale=px.colors.sequential.Plasma, 
                log_x=False, labels={'value':'Count', 
                                    'index':'Item',
                                    'color':'None'
                                    })

    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="green",
        title_text="Item by count"
    )

    #fig.show()'''

    products = df["itemDescription"].unique()
    df_count = df["itemDescription"].value_counts()

    one_hot = pd.get_dummies(df['itemDescription'])

    df.drop(['itemDescription'], inplace=True, axis=1)
    df = df.join(one_hot)
    #df.head()

    records = df.groupby(["Member_number","Date"])[products[:]].sum()
    records = records.reset_index()[products]

    #records

    def get_product_names(x):
        for product in products:
            if x[product] != 0:
                x[product] = product
        return x

    records = records.apply(get_product_names, axis=1)
    records.head()

    #print(f"Total transactions: {len(records)}")

    x = records.values
    x = [sub[~(sub == 0)].tolist() for sub in x if sub[sub != 0].tolist()]
    transactions = x

    association_rules = apriori(transactions,min_support=0.0003, min_confidance=0.0001, min_lift=3, min_length=2,target="rules")
    association_results = list(association_rules)

    for item in association_results:

        pair = item[0] 
        items = [x for x in pair]
        
        '''print("Rule : ", items[0], " -> " + items[1])
        print("Support : ", str(item[1]))
        print("Confidence : ",str(item[2][0][2]))
        print("Lift : ", str(item[2][0][3]))
        
        print("=====================================")'''

    items = []
    for item in association_results:
        pair = item[0] 
        items.append([x for x in pair])
        #print([x for x in pair])
    #print(items)

    item_counts = df_count.to_dict()
    #item_counts

    res = []
    for i in range(len(items)):
        sum = 0
        for item in items[i]:
            if item in item_counts:
                sum += item_counts[item]
        res.append([sum,items[i]])

    from operator import itemgetter
    res = sorted(res, key=itemgetter(0),reverse=True)
    #res

    df_res = pd.DataFrame(res, columns = ['Revenue', 'Items Group'])
    #df_res = df_res.sort_values('Revenue',ascending=False)
    #df_res.head(40)

    temp_set = set()
    items_unique = []
    for item in res:
        flag = 0
        for i in item[1]:
            if i in temp_set:
                flag = 1
            else:
                temp_set.add(i)
        if flag == 0:
            items_unique.append(item)
                
    #items_unique

    temp_set = set()
    items_unique_list = []

    for item in res:
        flag = 0
        temp = []
        for i in item[1]:
            if i not in temp_set:
                temp.append(i)
                temp_set.add(i)
                
        items_unique_list.append(temp)
                
    items_unique_list = list(filter(None,items_unique_list))
    #items_unique_list

    for item in item_counts:
        if item not in temp_set:
            items_unique.append([item_counts[item],[item]])
            
    items_unique = sorted(items_unique, key=itemgetter(0),reverse=True)
    #items_unique

    ress = []
    for i in range(len(items_unique_list)):
        sum = 0
        for item in items_unique_list[i]:
            if item in item_counts:
                sum += item_counts[item]
        ress.append([sum,items_unique_list[i]])

    count = list(item_counts.values())

    gi = pd.read_csv("Grocery_items_dataset.csv")
    gi.head()
    gi['price in euros'] = gi['price']/81.41
    gi['price in euros'] = gi['price in euros'].round(decimals=2)

    gi.to_csv('item_list.csv')

    gi['count'] = count
    #gi.head()

    gi['revenue'] = gi['price']*gi['count']
    #gi.head(20)
    gi = gi.reset_index()

    gc = gi.groupby('category').sum().sort_values("revenue",ascending=False)
    gc = gc.reset_index()
    #gc

    #import matplotlib.pyplot as plt
    #import seaborn as sns

    #from wordcloud import WordCloud

    #plt.rcParams['figure.figsize'] = (15, 15)
    #wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(gc['category']))
    #plt.imshow(wordcloud)
    #plt.axis('off')
    #plt.title('category',fontsize = 20)
    #plt.show()

    #fig = px.treemap(gc, path=['category'], values='revenue')
    #fig.show()

    ggs = gi.groupby(['category']).apply(lambda x: x.sort_values(['revenue'],ascending=False))


    df_temp = pd.DataFrame()
    for i in range(len(gc.index)):
        df_temp = df_temp.append(ggs['category'][gc['category'][i]].to_frame().reset_index())
    gj = df_temp.drop(columns=['category'])

    ls = pd.DataFrame()
    for i in gj['index']:
        ls = ls.append(gi.iloc[[i]])

    def result(gi):
        gc = gi.groupby('category').sum().sort_values("revenue",ascending=False)
        gc = gc.reset_index()
        #gc
        #fig = px.treemap(gc, path=['category'], values='revenue')
        #fig.show()
        ggs = gi.groupby(['category']).apply(lambda x: x.sort_values(['revenue'],ascending=False))
        #ggs
        df_temp = pd.DataFrame()
        for i in range(len(gc.index)):
            df_temp = df_temp.append(ggs['category'][gc['category'][i]].to_frame().reset_index())
        gj = df_temp.drop(columns=['category'])
        #gj
        ls = pd.DataFrame()
        for i in gj['index']:
            ls = ls.append(gi.iloc[[i]])
        return ls

    def add_assortment(res):
        category_item_count_dict = {}
        for cat in seq:
            category_item_count_dict[cat] = (len(res[res['category'] == cat]))
        #category_item_count_dict
        cat_seq = []
        for i in res['category'].tolist():
            if i not in cat_seq:
                cat_seq.append(i)
        #cat_seq
        import math
        assortment = []
        for c in cat_seq:
            i = category_item_count_dict[c]
            j = math.ceil(i/4)
            while(i!=0):
                for k in range(1,j+1):
                    if i == 0:
                        break
                    assortment.append('2x'+str(k))
                    i-=1
                for k in range(1,j+1):
                    if i == 0:
                        break
                    assortment.append('1x'+str(k))
                    i-=1
                for k in range(1,j+1):
                    if i == 0:
                        break
                    assortment.append('3x'+str(k))
                    i-=1
                for k in range(1,j+1):
                    if i == 0:
                        break
                    assortment.append('4x'+str(k))
                    i-=1
        #len(assortment)
        res['assortment'] = assortment
        res[['assortment_row','assortment_column']] = res['assortment'].str.split('x',expand=True)
        return res

    res = result(gi)
    #print("Store 1")
    #fig.show()
    #res.head(10)

    store1_item = gi
    store2 = pd.read_csv("store2_grocery_items_count.csv")
    store3 = pd.read_csv("store3_grocery_items_count.csv")
    event = pd.read_csv("christmas_items_count.csv")

    temp = gi.drop(columns=['count','revenue'])
    #temp

    def item_category_function(item_counts):
        
        a = {}
        for item in item_counts:
            a[item] = (gi.loc[gi['item'] == item]['category']).tolist()[0]
        return a
    item_category = item_category_function(item_counts)

    def get_sequence_items(items_unique_list):
        seq = []
        s = set()
        for item_list in items_unique_list:
            for item in item_list:
                cat  = item_category[item]
                if cat not in s:
                    seq.append(cat)
                    s.add(cat)
        li = gc['category'].tolist()
        for item in li:
            cat  = item
            if cat not in s:
                seq.append(cat)
                s.add(cat)
        return seq
    seq = get_sequence_items(items_unique_list)

    store2_item = temp.copy(deep=True)
    store2_item['count'] = store2['count']
    store2_item['revenue'] = store2_item['count']*store2_item['price']

    store3_item = temp.copy(deep=True)
    store3_item['count'] = store3['count']
    store3_item['revenue'] = store3_item['count']*store3_item['price']

    event_item = temp.copy(deep=True)
    event_item['count'] = event['count']
    event_item['revenue'] = event_item['count']*event_item['price']


    if(num == 1):
        res = result(store1_item)
        #fig.show()
        res = add_assortment(res)
        #res.to_csv('store1.csv')
        return res.to_json()

    if(num == 2):
        res = result(store2_item)
        #fig.show()
        res = add_assortment(res)
        #res.to_csv('store2.csv')
        return res.to_json()

    if(num == 3):
        res = result(store3_item)
        #fig.show()
        res = add_assortment(res)
        #res.to_csv('store3.csv')
        return res.to_json()

    if(num == 4):
        res = result(event_item)
        #fig.show()
        res = add_assortment(res)
        #res.to_csv('event.csv')
        return res.to_json()
        
    return jsonify({'data': num**2})

try:
    import apyori
except:
    #!pip install apyori
    print("Apyori not installed")

from apyori import apriori

df = pd.read_csv("Groceries_dataset.csv")
#df.head()

#print("We have :", df.shape[0], "items in the Dataset")
#print(f"And null values :\n{df.isnull().sum()}")

#print(f"And nan values :\n{df.isna().sum()} ")

#print(f'There are : {len(df["itemDescription"].unique())} unique items')

df["Year"] = df["Date"].str.split("-").str[-1]
df["Months_Year"] = df["Date"].str.split("-").str[1] + "-" + df["Date"].str.split("-").str[-1] 
#df.head()

#df["Months_Year"].unique()

#Graph : Months/Year by count
'''fig = px.bar(df["Months_Year"].value_counts(ascending=False), orientation="v", color=df["Months_Year"].value_counts(ascending=False), color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Count', 
                                'index':'Date',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Date by count"
)

#fig.show()

#Graph : Item by count
fig = px.bar(df["itemDescription"].value_counts()[:20], orientation="v", color=df["itemDescription"].value_counts()[:20], color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Count', 
                                'index':'Item',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Item by count"
)

#fig.show()'''

products = df["itemDescription"].unique()
df_count = df["itemDescription"].value_counts()

one_hot = pd.get_dummies(df['itemDescription'])

df.drop(['itemDescription'], inplace=True, axis=1)
df = df.join(one_hot)
#df.head()

records = df.groupby(["Member_number","Date"])[products[:]].sum()
records = records.reset_index()[products]

#records

def get_product_names(x):
    for product in products:
        if x[product] != 0:
            x[product] = product
    return x

records = records.apply(get_product_names, axis=1)
records.head()

#print(f"Total transactions: {len(records)}")

x = records.values
x = [sub[~(sub == 0)].tolist() for sub in x if sub[sub != 0].tolist()]
transactions = x

association_rules = apriori(transactions,min_support=0.0003, min_confidance=0.0001, min_lift=3, min_length=2,target="rules")
association_results = list(association_rules)

for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    
    '''print("Rule : ", items[0], " -> " + items[1])
    print("Support : ", str(item[1]))
    print("Confidence : ",str(item[2][0][2]))
    print("Lift : ", str(item[2][0][3]))
    
    print("=====================================")'''

items = []
for item in association_results:
    pair = item[0] 
    items.append([x for x in pair])
    #print([x for x in pair])
#print(items)

item_counts = df_count.to_dict()
#item_counts

res = []
for i in range(len(items)):
    sum = 0
    for item in items[i]:
        if item in item_counts:
            sum += item_counts[item]
    res.append([sum,items[i]])

from operator import itemgetter
res = sorted(res, key=itemgetter(0),reverse=True)
#res

df_res = pd.DataFrame(res, columns = ['Revenue', 'Items Group'])
#df_res = df_res.sort_values('Revenue',ascending=False)
#df_res.head(40)

temp_set = set()
items_unique = []
for item in res:
    flag = 0
    for i in item[1]:
        if i in temp_set:
            flag = 1
        else:
            temp_set.add(i)
    if flag == 0:
        items_unique.append(item)
            
#items_unique

temp_set = set()
items_unique_list = []

for item in res:
    flag = 0
    temp = []
    for i in item[1]:
        if i not in temp_set:
            temp.append(i)
            temp_set.add(i)
            
    items_unique_list.append(temp)
            
items_unique_list = list(filter(None,items_unique_list))
#items_unique_list

for item in item_counts:
    if item not in temp_set:
        items_unique.append([item_counts[item],[item]])
        
items_unique = sorted(items_unique, key=itemgetter(0),reverse=True)
#items_unique

ress = []
for i in range(len(items_unique_list)):
    sum = 0
    for item in items_unique_list[i]:
        if item in item_counts:
            sum += item_counts[item]
    ress.append([sum,items_unique_list[i]])

count = list(item_counts.values())

gi = pd.read_csv("Grocery_items_dataset.csv")
gi.head()
gi['price in euros'] = gi['price']/81.41
gi['price in euros'] = gi['price in euros'].round(decimals=2)

gi.to_csv('item_list.csv')

gi['count'] = count
#gi.head()

gi['revenue'] = gi['price']*gi['count']
#gi.head(20)
gi = gi.reset_index()

gc = gi.groupby('category').sum().sort_values("revenue",ascending=False)
gc = gc.reset_index()
#gc

#import matplotlib.pyplot as plt
#import seaborn as sns

#from wordcloud import WordCloud

#plt.rcParams['figure.figsize'] = (15, 15)
#wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(gc['category']))
#plt.imshow(wordcloud)
#plt.axis('off')
#plt.title('category',fontsize = 20)
#plt.show()

#fig = px.treemap(gc, path=['category'], values='revenue')
#fig.show()

ggs = gi.groupby(['category']).apply(lambda x: x.sort_values(['revenue'],ascending=False))


df_temp = pd.DataFrame()
for i in range(len(gc.index)):
    df_temp = df_temp.append(ggs['category'][gc['category'][i]].to_frame().reset_index())
gj = df_temp.drop(columns=['category'])

ls = pd.DataFrame()
for i in gj['index']:
    ls = ls.append(gi.iloc[[i]])

def result(gi):
    gc = gi.groupby('category').sum().sort_values("revenue",ascending=False)
    gc = gc.reset_index()
    #gc
    #fig = px.treemap(gc, path=['category'], values='revenue')
    #fig.show()
    ggs = gi.groupby(['category']).apply(lambda x: x.sort_values(['revenue'],ascending=False))
    #ggs
    df_temp = pd.DataFrame()
    for i in range(len(gc.index)):
        df_temp = df_temp.append(ggs['category'][gc['category'][i]].to_frame().reset_index())
    gj = df_temp.drop(columns=['category'])
    #gj
    ls = pd.DataFrame()
    for i in gj['index']:
        ls = ls.append(gi.iloc[[i]])
    return ls

def add_assortment(res):
    category_item_count_dict = {}
    for cat in seq:
        category_item_count_dict[cat] = (len(res[res['category'] == cat]))
    #category_item_count_dict
    cat_seq = []
    for i in res['category'].tolist():
        if i not in cat_seq:
            cat_seq.append(i)
    #cat_seq
    import math
    assortment = []
    for c in cat_seq:
        i = category_item_count_dict[c]
        j = math.ceil(i/4)
        while(i!=0):
            for k in range(1,j+1):
                if i == 0:
                    break
                assortment.append('2x'+str(k))
                i-=1
            for k in range(1,j+1):
                if i == 0:
                    break
                assortment.append('1x'+str(k))
                i-=1
            for k in range(1,j+1):
                if i == 0:
                    break
                assortment.append('3x'+str(k))
                i-=1
            for k in range(1,j+1):
                if i == 0:
                    break
                assortment.append('4x'+str(k))
                i-=1
    #len(assortment)
    res['assortment'] = assortment
    res[['assortment_row','assortment_column']] = res['assortment'].str.split('x',expand=True)
    return res

res = result(gi)
#print("Store 1")
#fig.show()
#res.head(10)

store1_item = gi
store2 = pd.read_csv("store2_grocery_items_count.csv")
store3 = pd.read_csv("store3_grocery_items_count.csv")
event = pd.read_csv("christmas_items_count.csv")

temp = gi.drop(columns=['count','revenue'])
#temp

def item_category_function(item_counts):
    
    a = {}
    for item in item_counts:
        a[item] = (gi.loc[gi['item'] == item]['category']).tolist()[0]
    return a
item_category = item_category_function(item_counts)

def get_sequence_items(items_unique_list):
    seq = []
    s = set()
    for item_list in items_unique_list:
        for item in item_list:
            cat  = item_category[item]
            if cat not in s:
                seq.append(cat)
                s.add(cat)
    li = gc['category'].tolist()
    for item in li:
        cat  = item
        if cat not in s:
            seq.append(cat)
            s.add(cat)
    return seq
seq = get_sequence_items(items_unique_list)

store2_item = temp.copy(deep=True)
store2_item['count'] = store2['count']
store2_item['revenue'] = store2_item['count']*store2_item['price']

store3_item = temp.copy(deep=True)
store3_item['count'] = store3['count']
store3_item['revenue'] = store3_item['count']*store3_item['price']

event_item = temp.copy(deep=True)
event_item['count'] = event['count']
event_item['revenue'] = event_item['count']*event_item['price']

res = result(store1_item)
#fig.show()
res = add_assortment(res)
res.to_csv('store1.csv')

res = result(store2_item)
#fig.show()
res = add_assortment(res)
res.to_csv('store2.csv')

res = result(store3_item)
#fig.show()
res = add_assortment(res)
res.to_csv('store3.csv')

res = result(event_item)
#fig.show()
res = add_assortment(res)
res.to_csv('event.csv')