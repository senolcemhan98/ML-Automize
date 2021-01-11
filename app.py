import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from pyecharts import options as opts
import streamlit.components.v1 as components
from pyecharts.charts import Bar
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff



st.header('Automatize Project')


st.write('Hello Everyone. Welcome to My Automatize Project!!!')

#-------------------------------------------------------------
# Slidebar Functions
def option_func(model,test_size,target_feature):
    data = pd.read_csv(dataset)
    categorical_columns = list(data.select_dtypes(include=['object']).columns)
    for col in categorical_columns: # Label Encoder Categorical Col
        data[col] = data[col].fillna('0') #Fill categorical nan values
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    data = data.fillna(0)
    
    y = data[target_feature]
    X = data.drop(target_feature,axis=1)

    if model.split(' ')[-1] == 'Classifier':
        sm = SMOTE(random_state=42)  #Oversampling
        X, y = sm.fit_resample(X,y)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = test_size, random_state=42)

    if model=='Random Forest Classifier':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        st.write('Test Accuracy : {}'.format(test_acc))

    elif model=='KNN Classifier':
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        st.write('Test Accuracy : {}'.format(test_acc))

    elif model=='SVM Classifier':
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        st.write('Test Accuracy : {}'.format(test_acc))

    elif model=='XGBoost Classifier':
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        st.write('Test Accuracy : {}'.format(test_acc))

    elif model=='Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_score = r2_score(y_test,y_pred)
        st.write('Mean Squared Error(mse) : {}'.format(mse))
        st.write('R^2 Score               : {}'.format(r_score))

    elif model=='Random Forest Regressor':
        model = RandomForestRegressor(max_depth=4, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_score = r2_score(y_test,y_pred)
        st.write('Mean Squared Error(mse) : {}'.format(mse))
        st.write('R^2 Score               : {}'.format(r_score))


         
#------------------------------------------------
# Main Page Helper Functions 

def data_head(dataset):
    pd.pandas.set_option('display.max_columns', None)
    data = pd.read_csv(dataset)
    head = data.head()
    fig = ff.create_table(head)
    st.plotly_chart(fig)

def data_describe(dataset):
    data = pd.read_csv(dataset)
    describe = (data.describe()).T
    st.write(describe)

def data_isna(dataset):
    data = pd.read_csv(dataset)
    rates = data.isna().sum()*100/len(data)
    st.write(pd.DataFrame(rates))

def data_shape(dataset):
    data = pd.read_csv(dataset)
    st.write(data.shape)

def data_columns(dataset):
    data = pd.read_csv(dataset)
    st.write(data.columns)

#-------------------------
#Visualize Functions

def pd_profiling():
    data = pd.read_csv(dataset)
    pr = ProfileReport(data, explorative=True)

    st.title("Pandas Profiling")
    st.write(data)
    st_profile_report(pr)

def ag_grid():
    data = pd.read_csv(dataset)
    AgGrid(data)

def render_bar(col):
    data = pd.read_csv(dataset)
    x = list(data[col].value_counts().keys())
    y = list(data[col].value_counts())
    
    c = (
        Bar()
        .add_xaxis(x)
        .add_yaxis(str(col), y)
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=str(col)
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
        .render_embed()  # generate a local HTML file
    )
    components.html(c, width=1000, height=1000)

def render_line(col):
    data = pd.read_csv(dataset)
    st.line_chart(data[col])

def render_multibar(x,y):
    data = pd.read_csv(dataset)
    fig = px.bar(data, x=x, y=y)
    st.plotly_chart(fig)

def render_multiline(x,y):
    data = pd.read_csv(dataset)
    fig = px.line(data, x=x, y=y)
    st.plotly_chart(fig)

def render_scatter(x,y):
    data = pd.read_csv(dataset)
    fig = px.scatter(data, x=x, y=y)
    st.plotly_chart(fig)

def render_pie(x,y):
    data = pd.read_csv(dataset)
    fig = px.pie(data,names=x,values=y)
    st.plotly_chart(fig)

def box_plot(col):
    data = pd.read_csv(dataset)
    fig = px.box(data, y=col)
    st.plotly_chart(fig)

def hist_plot(col):
    data = pd.read_csv(dataset)
    fig = px.histogram(data, x=col)
    st.plotly_chart(fig)

def dist_plot(col):
    data = pd.read_csv(dataset)
    fig = ff.create_distplot([data[col]], [col])
    st.plotly_chart(fig)

def data_corr():
    data = pd.read_csv(dataset)
    ze = data.corr()
    cols = list(data.corr().index)
    fig = go.Figure(data=go.Heatmap(z=ze,x=cols,y=cols))
    st.plotly_chart(fig)

    

#--------------------------------------------------

#Side Bar
with st.sidebar.header('1-Import Dataset'):
    dataset = st.sidebar.file_uploader('Upload your input CSV file.', type='csv')

if dataset is not None:
    st.sidebar.header('2- Choose Model')
    option = st.sidebar.selectbox('Choose Your Model',('KNN Classifier','SVM Classifier','Random Forest Classifier','XGBoost Classifier','Linear Regression','Random Forest Regressor'))
    test_size = st.sidebar.slider('Test Size :', 20,50,20,5)
    target_feature = st.sidebar.text_input("Target Feature")
    target_feature = str(target_feature)
    if st.sidebar.button('Select'):
        option_func(option,test_size,target_feature)


    st.sidebar.header('3- Basic Functions')

    if st.sidebar.button('show head'):
        data_head(dataset)
        if st.button('hide head'):
            st.write('')

    if st.sidebar.button('show describe'):
        data_describe(dataset)
        if st.button('hide describe'):
            st.write('')
    
    if st.sidebar.button('Show Null Value Rate'):
        data_isna(dataset)
        if st.button('hide Null Value Rate'):
            st.write('')

    if st.sidebar.button('Show shape'):
        data_shape(dataset)
        if st.button('hide shape'):
            st.write('')

    if st.sidebar.button('Show columns'):
        data_columns(dataset)
        if st.button('hide columns'):
            st.write('')

    if st.sidebar.button('Show Correlation Table'):
        data_corr()
        if st.button('hide Correlation Table'):
            st.write('')

    st.sidebar.header('4- Visualization Tools')
    st.sidebar.subheader('pandas profiling')
    if st.sidebar.button('pandas profiling'):
        pd_profiling()

    st.sidebar.subheader('Ag-Grid Table')
    if st.sidebar.button('Ag-Grid Table'):
        ag_grid()

    st.sidebar.subheader('One Column Chart')
    col = st.sidebar.text_input('Column Name')
    if st.sidebar.button('Bar Plot'):
        render_bar(col)

    if st.sidebar.button('Line Chart'):
        render_line(col)

    st.sidebar.subheader('Two Column Chart')
    x = str(st.sidebar.text_input('x axis column name'))
    y = str(st.sidebar.text_input('y axis column name'))
    if st.sidebar.button('Multi Bar Chart'):
        render_multibar(x,y)

    if st.sidebar.button('Multi Line Chart'):
        render_multiline(x,y)

    if st.sidebar.button('Scatter Plot'):
        render_scatter(x,y)

    st.sidebar.write('Note: x=name(e.g Country col) y=value(e.g Population col)')
    if st.sidebar.button('pie chart'):
        render_pie(x,y)

    st.sidebar.subheader('Statistical Charts')
    sta_col = str(st.sidebar.text_input('Statistical Chart Column'))
    if st.sidebar.button('box plot'):
        box_plot(sta_col)

    if st.sidebar.button('histogram'):
        hist_plot(sta_col)

    if st.sidebar.button('distplot'):
        dist_plot(sta_col)

    






    





    





        
        








