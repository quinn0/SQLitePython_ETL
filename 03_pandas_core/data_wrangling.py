# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Week 2 (Data Wrangling): Data Wrangling ----

# IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as ply
import plotly.graph_objects as go
from matplotlib import colors
from my_pandas_extensions.database import collect_data
from mizani.formatters import date_format, currency_format
from plotnine import( 
    ggplot, aes, geom_col,geom_histogram,
    geom_line, geom_smooth, geom_bar,
    facet_wrap, scale_y_continuous,
    scale_x_datetime, labs, theme,
    theme_minimal, theme_matplotlib,
    expand_limits, element_text,
    element_blank, element_rect,
    element_line, theme_seaborn,
    position_dodge2,coord_flip
)

# DATA
df=collect_data()
usd = currency_format(prefix = "$", digits= 0 , big_mark= ',')

# 1.0 SELECTING COLUMNS

# Select by name

df[['order_date', "order_id", "order_line"]]

# Select by position
df.iloc[:, -3:]#last 3 cols
#double brackets --> df, single --> series


# Select by text matching
df.filter(regex="(^model)|(price$)", axis = 1) #axis = 1 default

# Rearranging columns MOST EFFICIENT
l = df.columns.tolist()
cols_to_front = ['model', 'terrain', 'terrain2']
back = [col for col in l if col not in cols_to_front]
df[[*cols_to_front, *back]]

# Select by data types Less INNNNEFICIENT

df1 = df.select_dtypes(include="object")
df2 = df.select_dtypes(exclude="object")
df3 = pd.concat([df1, df2], axis=1)

# Dropping Columns (De-selecting)
df1 = df[cols_to_front]
df2 = df.drop(cols_to_front, axis=1)
df3 = pd.concat([df1,df2], axis=1)
# ARRANGING ROWS ----

df.sort_values('total_price', ascending= False)

df.sort_values(['terrain', "order_date"], ascending= False)

df["price"].sort_values(ascending=False)

# FILTERING  ----

# Simpler Filters
df[df.order_date >= pd.to_datetime("2015-01-01")]

df[df.model == "Trigger Carbon 1"]
df[df.model.str.startswith("Trigger")]
df[df.model.str.contains("Carbon", case = False)]
# Query
price_threshold1 = 9000
price_threshold2 = 5000
df.query("(price >= @price_threshold2)|(price >= @price_threshold2)")
df.query(f"price >= {price_threshold2}")######### I like this best
# Filtering Items in a List
df["terrain2"].unique()
df["terrain2"].value_counts()
df[df["terrain2"].isin(['Triathalon','Over Mountain'])]
#negatation tilde
df[~ df["terrain2"].isin(['Triathalon','Over Mountain'])]

# Slicing

df[5:]#head
df[:5] #tail
df.iloc[:5, [1,3,5]]

df.iloc[:5,:]
# Unique / Distinct Values
df.drop_duplicates(subset= ['model', 'terrain', 'terrain2'],keep = "first")
#last, first and False where false drops all dups

# Top / Bottom
df.nlargest(20, columns= ["price", "total_price"])
df.nsmallest(10, columns= ["price", "total_price"])

# Sampling Rows
df.sample(n = 10, random_state= 123 )
df.sample(frac = 0.10, random_state= 123 ) # SAMPLE 10%

# ADDING CALCULATED COLUMNS  ----
# assign (Great for method chaining)

df.assign(frame_material = lambda x : x["frame_material"].str.lower())
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price = lambda x: np.log(x['price']))\
    .set_index('model')\
    .plot(kind='hist')


# Adding Flags (True/False)
df.assign(flag_six = lambda x : x['model'].str.lower().str.contains('supersix'))



# Binning
bins = pd.cut(df['price'], bins=3, labels = ["low", 'medium', "high"]).astype('str')
df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group = \
        lambda x:pd.cut(df['price'], 
                        bins=3, 
                        labels = ["low", 'medium', "high"])\
                            .astype('str')
            ) \
    .pivot(index = 'model',
           columns = 'price_group',
           values = 'price'
           ).style.background_gradient(cmap="magma")
#good Categorical color map: tab20b

qts = list(np.multiply(np.array(list(range(0,101,25))), .01))

df[['model', 'price']]\
    .drop_duplicates()\
    .assign(price_group = \
        lambda x:pd.qcut(df["price"],
                         q = qts,
                         labels = ["<= Q1",
                                   "<= Q2",
                                   "<= Q3",
                                   "TOP25"]))\
    .pivot(index = 'model',
           columns = 'price_group',
           values = 'price'
           ).style.background_gradient(cmap="magma", axis=None)
#axis None to include range of cols and rows 
# ####(full heatmap instead of columnwise heatmap)


# Aggregations (No Grouping)
df.agg([np.sum, np.mean, np.std])
df.agg({'quantity':np.sum,
        'price' : [np.mean, np.std]})
# Common Summaries
df[['model', "terrain"]].value_counts()
df.nunique()
df.isna().sum()
#Groupby + Agg
df.groupby(by = ["city", "state"]).agg(np.sum)
df.groupby(by = ["city", "state"])\
    .agg(dict(
            quantity = [np.sum,np.mean],
            total_price = np.sum
            )
         )

# Get the sum and median by groups
summary1_df = df[['terrain', 'terrain2', 'total_price']]\
    .groupby(['terrain', "terrain2"])\
    .agg([np.sum, np.median])\
    .reset_index()
# Apply Summary Functions to Specific Columns
summary2_df = df[['terrain', 'terrain2', 'total_price', "quantity"]]\
    .groupby(by=["terrain", "terrain2"])\
    .agg(dict(
            total_price = np.mean,
            quantity = np.sum
            )
         )\
    .reset_index()#.style.background_gradient(cmap= "magma", axis = 0)

#note multi level index 
summary1_df.columns
summary1_df.isna().sum()


#Groupby + Transform 

summary3_df = df[['terrain2', 'order_date', 'total_price', "quantity"]]\
    .set_index("order_date")\
    .groupby(by = "terrain2")\
    .resample("W")\
    .agg(np.sum)\
    .reset_index()

summary3_df\
    .set_index("order_date")\
    .groupby("terrain2")\
    .apply(lambda x: (x.total_price -\
        x.total_price.mean())/x.total_price.std())\
    .reset_index() \
    .pivot(index = "order_date", columns="terrain2", values = "total_price" )\
    .plot()
##same but over quantity using multiple indices
summary3_df\
    .set_index(["order_date", "terrain2"])\
    .groupby("terrain2")\
    .apply(lambda x: (x - x.mean())/ \
        x.total_price.std())\
    .reset_index() 

# Groupby + Filter
# convenient tail 

summary3_df\
    .groupby("terrain2")\
        .tail()

summary3_df\
    .groupby("terrain2")\
    .apply(lambda x: x.iloc[10:]) #grab specific range


# RENAMING FIELDS

# Single Index
summary2_df\
    .rename(
        columns = lambda x: x.replace("_", " ").title()
    )#.style.background_gradient(cmap= "magma", axis = 0)

# Targeting specific columns
summary2_df\
    .rename(columns = {"total_price" : "Revenue"})

# - Mult-Index

summary1_df.columns.to_list()
##COLLAPSING INDICES
summary1_df\
    .set_axis(["_".join(col).rstrip("_") for col in\
        summary1_df.columns.to_list()], axis = 1)



#RESHAPING (MELT & PIVOT_TABLE) ----

# Aggregate Revenue by Bikeshop by Category 1 
df[["bikeshop_name", "terrain", "total_price"]]\
    .groupby(["bikeshop_name", "terrain"])\
    .sum()\
        .reset_index()\
    .sort_values("total_price", ascending= False)\
    .rename(columns = lambda x : x.replace("_", " ").title())


# Pivot (Pivot Wider)
rev_wide_df = df[["bikeshop_name", "terrain", "total_price"]]\
    .groupby(["bikeshop_name", "terrain"])\
    .sum()\
        .reset_index()\
    .sort_values("total_price", ascending= False)\
    .rename(columns = lambda x : x.replace("_", " ").title())\
    .pivot(index   = "Bikeshop Name", 
           columns = "Terrain",
           values  = "Total Price")\
    .reset_index()\
    .set_axis(["Bikeshop Name", "Mountain", "Road"], 
              axis = 1)\
    .sort_values("Mountain")
    # .plot(x = "Bikeshop Name",
    #     y = ["Mountain", "Road"],
    #     kind = "barh")
    
##Revenue by category of each bikeshop
### answers: what types of bikes do the bikeshops sell most freq

# 7.2 Pivot Table (Pivot + Summarization, Excel Pivot Table)
rev_wide_df.sort_values("Mountain", ascending= False)\
    .style.highlight_max()\
    .format({ "Mountain": lambda x : usd([x])[0],
             "Road"     : lambda x : usd([x])[0]      
    })\
    .to_excel("./03_pandas_core/rev_wide_df.xlsx", index = False)

#wide: matplotlib
#long: ggplot
# Melt (Pivoting Longer)
rev_long_df = pd.read_excel("./03_pandas_core/rev_wide_df.xlsx")\
    .melt(value_vars=("Mountain", "Road"),
          var_name = ("Terrain"),
          id_vars = ("Bikeshop Name"),
          value_name= ("Revenue")
          )

ggplot(aes(x = "Bikeshop Name", 
           y = "Revenue", 
           fill = "Terrain"),
       data = rev_long_df) +\
    geom_col(position = "dodge2") +\
    coord_flip() #+\
    # facet_wrap("Terrain")

# position="dodge2" for dodged


bshop_order = rev_long_df\
    .groupby("Bikeshop Name")\
    .sum()\
    .sort_values("Revenue")\
    .index \
    .to_list()

rev_long_df['Bikeshop Name'] =\
    pd.Categorical(rev_long_df['Bikeshop Name'], categories= bshop_order)

ggplot(aes(x = "Bikeshop Name", 
           y = "Revenue", 
           fill = "Terrain"),
       data = rev_long_df) +\
    geom_col() +\
    coord_flip() +\
    theme_minimal()+\
    facet_wrap("Terrain")
#feature: proportion of mountain/road bikes sold standardized

df\
    .pivot_table(columns = None,
                 values= 'total_price',
                 index = ["terrain", "frame_material"],
                 aggfunc= np.sum).reset_index()

#columns  = None for all
rev_cat12_year=df\
    .assign(year = lambda x : x.order_date.dt.year)\
    .pivot_table(index = ["terrain","frame_material"],
                 values = "total_price",
                 columns = "year",
                 aggfunc= np.sum)
                     #.style.highlight_max(color = "purple", axis = None)
    
# switch columns and index to transpose
# index(year) alone breaks mean out by year

# Stack & Unstack multi-indices ----
rev_cat12_year\
    .unstack(fill_value=0, level = "terrain")
# Unstack - Pivots Wider 1 Level (Pivot)
# Stack - Pivots Longer 1 Level (Melt)

rev_cat12_year\
    .stack(level = "year")
    
rev_cat12_year\
    .stack(level = "year")\
    .unstack(level = ["terrain", "frame_material"])
    
# switched columns to index and index to column 

# 8.0 JOINING DATA ----
# Merge (Joining)
orderlines_df = pd.read_excel("./Mod0/00_data_raw/orderlines.xlsx")
bikes_df = pd.read_excel("./Mod0/00_data_raw/bikes.xlsx")
#left outer join
#many to many yikes
pd.merge(how = "left",
         left = orderlines_df,
         right = bikes_df,
         left_on = "product.id",
         right_on = "bike.id" 
         )
# Concatenate (Binding)

# Columns ##MAKE SURE INDICES ARE RESET OR NaN
## could also use iloc[:,:]
pd.concat([bikes_df.head().reset_index(), bikes_df.tail().reset_index()], axis = 1)

# Rows default axis = 0
pd.concat([bikes_df.head(), bikes_df.tail()])


# SPLITTING COLUMNS AND COMBINING (UNITING) COLUMNS

# Separate
dt_splits_df = df["order_date"]\
    .astype(str).str.split("-", expand = True)\
    .set_axis(["year", "month", "day"], axis=1)\
    .apply(lambda x : x.astype('int')) ##made ints
df2 = pd.concat([df, dt_splits_df], axis = 1)
# Combine

datez = df2['year'].astype("str") + "-" + df2['month'].astype("str") + "-" + df2['day'].astype("str")
test = pd.concat([df2, datez], axis = 1)\
    .rename(columns= {0 : "datez"})

test["datez"] = pd.to_datetime(test["datez"])

# - Apply methods
sales_cat2_daily_df = df[["terrain2", "order_date", "total_price"]]\
    .set_index('order_date')\
    .groupby("terrain2")\
    .resample("D")\
    .agg(np.sum)
    
sales_cat2_daily_df.apply(np.mean, result_type= "broadcast")

sales_cat2_daily_df\
    .groupby("terrain2")\
    .apply(lambda x : np.repeat(np.mean(x), len(x)) )
    

sales_cat2_daily_df\
    .groupby("terrain2")\
    .transform(np.mean)
### repeats aggregate function for length of rows
# PIPING with decorators
# - Functional programming helper for "data" functions
def add_column(df, **kwargs):
    data = df.copy()
    df[list(kwargs.keys())] = pd.DataFrame(kwargs)
    return df
##function to add endless columns in pipe pretty nicely
### its like a multi-assign 
df\
.pipe(add_column, 
      total_price2 = df.total_price *2,
      terrainUP = df["terrain"].str.upper()).T



