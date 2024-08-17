import streamlit as st
import pickle
import re
import cv2
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import keras
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from PIL import Image
port_stem = PorterStemmer()
import time
from pygoogletranslation import Translator
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')

# from_email = "hamidrezabahrami455@gmail.com"
# to_email = "hamidr.bahraami@gmail.com"
# email_password = "htth eind kniy bgst"

# context = ssl.create_default_context()

st.set_page_config(page_title='جوانی جمعیت / تکریم سالمندان - RoboAi', layout='centered', page_icon='🩺')

vector = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

vectory = pickle.load(open('vectory.pkl', 'rb'))
load_modely = pickle.load(open('modely.pkl', 'rb'))

translator = Translator()

def load_modeli():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_modeli()

global model
modelh5 = tensorflow.keras.models.load_model('model.h5')

model = data['model']
x = data['x']

def stemming(content):
  con = re.sub('[^a-zA-Z]', ' ', content)
  con = con.lower()
  con = con.split()
  con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
  con = ' '.join(con)
  return con

def thought(text):
  text = stemming(text)
  input_text = [text]
  vector1 = vector.transform(input_text)
  prediction = load_model.predict(vector1)
  return prediction

def thoughty(text):
  text = stemming(text)
  input_text = [text]
  vector1 = vectory.transform(input_text)
  prediction = load_modely.predict(vector1)
  return prediction

def show_page():
    st.write("<h3 style='text-align: center; color: blue;'>سامانه تشخیص اختلالات مغزی و روانی سالمندان 🩺</h3>", unsafe_allow_html=True)
    st.write("<h5 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h5>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image('img.png')
        with col3:
            st.write(' ')
        st.divider()
        st.write("<h4 style='text-align: center; color: black;'>تشخیص زوال عقل یا دمانس زودرس</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>با استفاده از تصاویر اسکن مغزی</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>تحلیل افکار کاربر</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>و بررسی پرسشنامه</h4>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: black;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)
    
    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>بررسی ناامیدی ، افسردگی و ناراحتی های عاطفی 📑</h6>", unsafe_allow_html=True)

    text_1 = st.text_area('افکار خود را با من درمیان بگذارید',height=None,max_chars=None,key=None)

    button_1 = st.button('تحلیل')
    if button_1:
        if text_1 == "":
            with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی'''):
                    time.sleep(1)
                    st.success(u'\u2713''تحلیل انجام شد')
                    text1 = 'لطفا متن را وارد کنید'
                    def stream_data1():
                        for word in text1.split(" "):
                            yield word + " "
                            time.sleep(0.09)
                    st.write_stream(stream_data1)
    
        
        else:
            out = translator.translate(text_1)
            prediction_class = thought(out.text)
            if prediction_class == [1]:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، محتوای متن وارد شده دارای افکار منفی و ناامیدکننده است'
                        text2 = 'این به معنی بالا بودن احتمال ابتلا به افسردگی دوره سالمندی است'
                        text3 = 'برای بررسی دقیق تر و پیشگیری از عود به پزشک مراجعه کنید'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)
                        def stream_data3():
                            for word in text3.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data3)

            else:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text4 = 'بر اساس تحلیل من ، محتوای متن وارد شده دارای افکار سالم است'
                        text5 = 'این به معنی پائین بودن احتمال ابتلا به افسردگی دوره سالمندی است'
                        text6 = 'جای هیچ نگرانی ای نیست'
                        def stream_data4():
                            for word in text4.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data4)
                        def stream_data5():
                            for word in text5.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data5)
                        def stream_data6():
                            for word in text6.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data6)

    else:
        pass

    st.divider()

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>بررسی افکار خودکشی 📝</h6>", unsafe_allow_html=True)

    text_2 = st.text_area('افکار خود را بنویسید',height=None,max_chars=None,key=None)

    button_2 = st.button('ارزیابی')

    if button_2:
        if text_2 == "":
            with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی'''):
                    time.sleep(1)
                    st.success(u'\u2713''تحلیل انجام شد')
                    text1 = 'لطفا متن را وارد کنید'
                    def stream_data1():
                        for word in text1.split(" "):
                            yield word + " "
                            time.sleep(0.09)
                    st.write_stream(stream_data1)
    
        
        else:
            out = translator.translate(text_2)
            prediction_class = thought(out.text)
            if prediction_class == [1]:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text01 = 'بر اساس تحلیل من ، محتوای متن وارد شده دارای افکار خودکشی است'
                        text02 = 'این به معنی بالا بودن احتمال خودکشی در دوره سالمندی است'
                        text03 = 'برای بررسی دقیق تر و پیشگیری به روانشناس مراجعه کنید '
                        def stream_data1():
                            for word in text01.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text02.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)
                        def stream_data3():
                            for word in text03.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data3)

            else:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text04 = 'بر اساس تحلیل من ، محتوای متن وارد شده دارای افکار سالم است'
                        text05 = 'این به معنی پائین بودن احتمال خودکشی در دوره سالمندی است'
                        text06 = 'جای هیچ نگرانی ای نیست'
                        def stream_data4():
                            for word in text04.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data4)
                        def stream_data5():
                            for word in text05.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data5)
                        def stream_data6():
                            for word in text06.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data6)

    else:
        pass

    st.divider()

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص آلزایمر و زوال عقل زودرس از متن 💬</h6>", unsafe_allow_html=True)
    container.write("<h6 style='text-align: right; color: gray;'>.برای افزایش دقت تحلیل سامانه ، توسط اطرافیان شخص سالمند پاسخ داده شود ⚠️</h6>", unsafe_allow_html=True)
    container.write("<h6 style='text-align: right; color: gray;'>.علائم شخص سالمند موردنظر را در یک پاراگراف توصیف کنید 🗨️</h6>", unsafe_allow_html=True)
    
    text_3 = st.text_area('شخص سالمند موردنظر اغلب چه رفتارهایی از خود بروز می دهد؟',height=None,max_chars=None,key=None)

    button_3 = st.button('تحلیل رفتار')
    if button_3:
        if text_3 == "":
            with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی'''):
                    time.sleep(1)
                    st.success(u'\u2713''تحلیل انجام شد')
                    text1 = 'لطفا متن را وارد کنید'
                    def stream_data1():
                        for word in text1.split(" "):
                            yield word + " "
                            time.sleep(0.09)
                    st.write_stream(stream_data1)
    
        
        else:
            out = translator.translate(text_3)
            prediction_class = thoughty(out.text)
            if prediction_class == [1]:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، علائمی از آلزایمر و زوال عقل در شخص موردنظر دیده می شود'
                        text2 = 'برای بررسی دقیق تر و کنترل بهتر علائم پیش رونده ، به پزشک مراجعه کنید'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)

            else:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی'''):
                        time.sleep(1)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text3 = 'بر اساس تحلیل من ، علائمی از آلزایمر و زوال عقل در شخص موردنظر دیده نمی شود'
                        def stream_data3():
                            for word in text3.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data3)

    else:
        pass



    st.divider()

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>پرسشنامه پیش بینی احتمال بروز سکته مغزی 📋</h6>", unsafe_allow_html=True)
    container.write("<h6 style='text-align: right; color: gray;'>.برای افزایش دقت تحلیل سامانه ، توسط اطرافیان شخص سالمند پاسخ داده شود ⚠️</h6>", unsafe_allow_html=True)
        
    g = ('مرد' , 'زن')
    g = st.selectbox('جنسیت شما', g)
    if g == 'مرد':
        gender = 1.0
    else:
        gender = 0.0

    age = st.slider('سن شما', 10.0, 100.0, 35.0)

    h = ('بله' , 'خیر')
    h = st.selectbox('آیا فشار خون دارید؟', h)
    if h == 'بله':
        hypertension = 1.0
    else:
        hypertension = 0.0

    hd = ('بله' , 'خیر')
    hd = st.selectbox('آیا بیماری قلبی یا سابقه سکته قلبی دارید؟', hd)
    if hd == 'بله':
        heart_disease = 1.0
    else:
        heart_disease = 0.0

    m = ('بله' , 'خیر')
    m = st.selectbox('آیا تاکنون ازدواج کرده اید؟ - وضعیت تاهل در حال حاضر مدنظر نیست', m)
    if m == 'بله':
        ever_married = 1.0
    else:
        ever_married = 0.0

    wt = ('کسب و کار شخصی دارم / داشتم' , 'شغل دولتی دارم / داشتم', 'هیچ شغلی ندارم / نداشتم', 'فرزندانم خرج را می دهند')
    wt = st.selectbox('وضعیت یا سابقه شغلی خود را مشخص کنید', wt)
    if wt == 'کسب و کار شخصی دارم / داشتم':
        work_type = 1.0
    elif wt == 'شغل دولتی دارم / داشتم':
        work_type = 0.0
    elif wt == 'هیچ شغلی ندارم / نداشتم':
        work_type = 3.0
    else:
        work_type = 2.0
    

    rt = ('شهر' , 'روستا')
    rt = st.selectbox('بیشتر عمر خود شما در کجا سپری شده است؟', rt)
    if rt == 'بله':
        Residence_type = 1.0
    else:
        Residence_type = 0.0

    avg_glucose_level = st.slider('گلوگز خون شما بطور میانگین روی چه عددی است؟', 55.0, 270.0, 75.0)

    bmi = st.slider('شاخص توده بدنی خود را مشخص کنید', 11.0, 92.0, 25.0)

    smt = ('هرگز سیگار / قلیان نکشیده ام' , 'قبلا سیگار / قلیان می کشیدم', 'الان سیگار / قلیان می کشم')
    smt = st.selectbox('وضعیت استعمال دخانیات خود را مشخص کنید', smt)
    if smt == 'هرگز سیگار / قلیان نکشیده ام':
        smoking_status = 0.0
    elif st == 'قبلا سیگار / قلیان می کشیدم':
        smoking_status = 1.0
    else:
        smoking_status = 2.0
    
    button = st.button('پیش بینی')
    if button:
        with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی لطفا صبور باشید'''):
                    time.sleep(2)
                    st.success(u'\u2713''تحلیل انجام شد')
                    x = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level,
                                   bmi, smoking_status]])
        x = x.astype(float)

        y = model.predict(x)
        if y == 1:
            text1 = 'بر اساس تحلیل من ، احتمال بروز سکته مغزی در شما یا کاربر موردنظر بالا است'
            text2 = 'ادامه ی روند زندگی به شکل بالا ، موجب افزایش ریسک سکته قلبی در شما یا شخص سالمند موردنظر خواهد شد'
            text3 = 'لطفا در اسرع وقت به پزشک مراجعه کنید'
            text4 = 'Based on my analysis, you have a high chance of having Brain Stoke , if you or the user you mentioned continue to live like this'
            text5 = 'Please visit doctor as soon as possible'
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
            def stream_data3():
                for word in text3.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)
            def stream_data4():
                for word in text4.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data4)

        elif y == 0:
            text1 = 'بر اساس تحلیل من ، احتمال بروز سکته مغزی در شما یا کاربر موردنظر کم است'
            text2 = 'ادامه ی روند زندگی به شکل بالا ، موجب افزایش ریسک سکته قلبی در شما یا شخص سالمند موردنظر نخواهد شد'
            text3 = 'Based on my analysis, you have a low chance of having Brain Stoke , if you or the user you mentioned continue to live like this'
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
            def stream_data3():
                for word in text3.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)


    st.divider()

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص آلزایمر و تحلیل رفتن مغز با اسکن مغزی 🧠</h6>", unsafe_allow_html=True)

    image = st.file_uploader('آپلود تصویر', type=['jpg', 'jpeg'])
    button = st.button('تحلیل اسکن مغزی')       
    if image is not None:
        file_bytes = np.array(bytearray(image.read()), dtype= np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels= 'BGR', use_column_width= True)
        if button: 
            x = cv2.resize(img, (200, 200))
            x1 = img_to_array(x)
            x1 = x1.reshape((1,) + x1.shape)
            y_pred = modelh5.predict(x1)
            if y_pred == 1:
                with st.chat_message("assistant"):
                    with st.spinner('''در حال تحلیل'''):
                        time.sleep(2)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، بخش هایی از بافت مغز کاربر تحلیل رفته است'
                        text2 = 'این تحلیل رفتگی می تواند ناشی از بروز خاموش زوال عقل ، آلزایمر و بیماری های مغزی مشابه باشد'
                        text3 = 'لطفا در اسرع وقت به پزشک مراجعه کنید'
                        text4 = 'Based on my analysis, owner of this MRI image has partially lost density of their brain tissue'
                        text5 = 'This has occured because of early stages of Dementia , Alzheimer or other brain diseases'
                        text6 = 'Please visit doctor as soon as possible'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)
                        def stream_data3():
                            for word in text3.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data3)
                        def stream_data4():
                            for word in text4.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data4)
                        def stream_data5():
                            for word in text5.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data5)
                        def stream_data6():
                            for word in text6.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data6)
                        
            elif y_pred == 0:
                with st.chat_message("assistant"):
                    with st.spinner('''در حال تحلیل'''):
                        time.sleep(2)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، بافت مغز در اسکن آپلود شده سالم است'
                        text2 = 'Based on my analysis , brain tissue in this MRI image is healthy and untouched'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)

show_page()
