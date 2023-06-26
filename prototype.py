import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import datetime
import cv2
import os
import psutil
from PIL import Image
import time
import plost # streamlit 그래프 관련 라이브러리
from streamlit_elements import elements, mui, nivo # streamlit 그래프 관련 라이브러리

# #################### Load Model ####################
# def load_model(path, device):
#     model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
#     model_.to(device)
#     return model_

# cfg_model_path = 'best.pt' # 미리 학습시킨 best 가중치 포함된 파일
# model = None
# confidence = .25
# device_option = 'cpu'
# model = load_model(cfg_model_path, device_option)

#################### Load Data ####################
fake = pd.read_csv('가상데이터.csv', encoding="cp949") # 가상데이터 10만개 for 불법차량 기록지, 통계 시각화
data = pd.read_csv('tab3_data.csv', encoding='utf-8') # 통계 시각화 위한 가상데이터

# #################### Load Video ####################
# def video_input(data_src):
#     now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     now_day = datetime.datetime.now().strftime('%Y-%m-%d') # 단속일자 출력 위한 변수
#     now_sec = datetime.datetime.now().strftime('%H:%M:%S') # 단속시간 출력 위한 변수
    
#     vid_file = None
#     # 영상 경로 설정, 일단 TOP5 지사를 선택
#     if data_src == '진천지사':
#         vid_file = "진천지사.mp4"
#     elif data_src == '제천지사':
#         vid_file = "제천지사.mp4"
#     elif data_src == '수원지사':
#         vid_file = "수원지사.mp4"
#     elif data_src == '창원지사':
#         vid_file = "창원지사.mp4"
#     else:
#         vid_file = "인천지사.mp4"
 
#     if vid_file:
#         col1, col2, col3 = st.columns([0.25, 0.6, 0.2])
#         with col2:
#             cap = cv2.VideoCapture(vid_file)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = 0

#             output = st.empty()
#             prev_time = 0
#             curr_time = 0

#             #프레임을 저장할 디렉토리를 생성
#             filepath = './images'  # 캡쳐 이미지 저장 경로
#             try:
#                 if not os.path.exists(filepath):
#                     os.makedirs(filepath)
#             except OSError:
#                 print ('Error: Creating directory. ' +  filepath)

#         # 단속 대시보드 구성
#         col11, col12, col13 = st.columns([0.25, 0.6, 0.2])
#         with col12:
#             st.markdown("---")

#         col21, col22, col23, col24, col25 = st.columns([0.25, 0.2, 0.2, 0.2, 0.2])
#         with col22:
#             st.markdown("#### 단속 영업소명")
#             col22_text = st.markdown(f"{data_src}")
#         with col23:
#             st.markdown("#### 단속일자")
#             col23_text = st.markdown(f"{now_day}")
#         with col24:
#             st.markdown("#### 단속시간")
#             col24_text = st.markdown(f"{now_sec}")
            
#         # Updating System stats
#         col31, col32, col33 = st.columns([0.25, 0.2, 0.6])
#         with col32:
#             st.markdown("#### System Stats")

#         col41, col42, col43, col44, col45 = st.columns([0.25, 0.2, 0.2, 0.2, 0.2])
#         with col42:
#             st.markdown("**Memory usage**")
#             col42_text = st.markdown("0")
#         with col43:
#             st.markdown("**CPU Usage**")
#             col43_text = st.markdown("0")
#         with col44:
#             st.markdown("**GPU Memory Usage**")
#             col44_text = st.markdown("0")

#         count = 0
#         sec = 30  # 5초 = 5 * 30
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 st.write("Can't read frame, stream ended? Exiting ....")
#                 break
#             frame = cv2.resize(frame, (width, height))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             output_img = infer_image(frame)
#             output.image(output_img)
#             curr_time = time.time()
#             fps = 1 / (curr_time - prev_time)
#             prev_time = curr_time

#             # 영상 캡쳐             
#             if(int(cap.get(1)) % sec == 0): #앞서 불러온 fps 값을 사용하여 5초마다 추출
#                 cv2.imwrite(filepath + "/second_%d.jpg" % count, frame)
#                 count += 1            

#             col22_text.markdown(f"**{data_src}**")
#             col23_text.markdown(f"**{now_day}**")
#             col24_text.markdown(f"**{now_sec}**")

#             col42_text.write(str(psutil.virtual_memory()[2])+"%")
#             col43_text.write(str(psutil.cpu_percent())+'%')
#             try:
#                 col44_text.write(str(get_gpu_memory())+' MB')
#             except:
#                 col44_text.write(str('N/A'))
            
#         cap.release()

# def infer_image(img, size=None):
#     model.conf = confidence
#     result = model(img, size=size) if size else model(img)
#     result.render()
#     image = Image.fromarray(result.ims[0])
#     return image

# def get_gpu_memory():
#     result = subprocess.check_output(
#         [
#             'nvidia-smi', '--query-gpu=memory.used',
#             '--format=csv,nounits,noheader'
#         ], encoding='utf-8')
#     gpu_memory = [int(x) for x in result.strip().split('\n')]
#     return gpu_memory[0]
       
#################### Sidebar Setting ####################
st.set_page_config(layout="wide")

pages = ['단속 대시보드', '불법차량 기록지', '통계 시각화']
page = st.sidebar.selectbox("Select Tabs", options=pages)

jisa = ['진천지사', '제천지사', '수원지사', '창원지사', '인천지사']

#################### DashBoard ####################
if page == "단속 대시보드":
    col110, col111, col112 = st.columns([0.25, 0.6, 0.2])
    with col111:
        st.title(page)
        
    # Input src option
    data_src = st.sidebar.selectbox("단속 영업소를 선택하세요: ", jisa)
    
    # Confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.7)
    
    # Custom classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
        classes = [model_names.index(name) for name in assigned_class]
        model.classes = classes
    else:
        model.classes = list(model.names.keys())
        
    # video_input(data_src)
    

#################### Detect Log ####################    
elif page == '불법차량 기록지':
    st.title(page)
    st.markdown("#### 최근 단속 차량")
    
    # dispatch_data를 가상데이터 중 단속된 부분만 사용
    dispatch_data = fake[fake['단속여부'] == '단속']
    
    # 단속 중 검측된 번호를 부여(통과된 순번을 의미함)
    dispatch_data.index.name = '검측번호'
    dispatch_data.reset_index(inplace=True)
    
    max_row = dispatch_data.shape[0] - 1
    
    dispatch_data['일별'] = pd.to_datetime(dispatch_data['일시']).dt.strftime("%Y-%m-%d")
    
    tab2_min_date = datetime.datetime.strptime(dispatch_data['일별'][0], "%Y-%m-%d")
    tab2_max_date = datetime.datetime.strptime(dispatch_data['일별'][max_row], "%Y-%m-%d")
    
    slider_date = st.slider("기간", tab2_min_date, tab2_max_date, value=(tab2_min_date, tab2_max_date))
    
    dispatch_data['일별'] = pd.to_datetime(dispatch_data['일시'])
    dispatch_data = dispatch_data[(dispatch_data['일별'] >= slider_date[0]) & (dispatch_data['일별'] <= slider_date[-1] + datetime.timedelta(days=1))]
    dispatch_data = dispatch_data[['일시', '단속 영업소', '검측번호', '차량 번호', '차종', '단속여부', '세부사항', '속도', '차로번호']]
    dispatch_data.reset_index(drop=True, inplace = True)
    
    last_car_num = dispatch_data.shape[0] - 1

    col210, col211, col212 = st.columns([0.2, 0.15, 0.15])
    with col210:
        st.image('Group.PNG', width=380)
    
    with col211:
        col220, col221 = st.columns([0.25, 0.25])
        with col220:
            st.write('##### 일시')
            st.write('##### 검측번호')
            st.write('##### 차량 번호')
            st.write('##### 차종')
            st.write('##### 차로번호')
        with col221:
            st.write(dispatch_data.loc[last_car_num]['일시'])
            st.write(str(dispatch_data.loc[last_car_num]['검측번호']))
            st.write(dispatch_data.loc[last_car_num]['차량 번호'])
            st.write(dispatch_data.loc[last_car_num]['차종'])
            st.write(str(dispatch_data.loc[last_car_num]['차로번호']))
        
    with col212:
        col222, col223 = st.columns([0.25, 0.25])
        with col222:
            st.write('##### 진입속도')
            st.write('##### 과적차량')
            st.write('##### 불법개조')

        with col223:
            st.write(str(dispatch_data.loc[last_car_num]['속도']))
            if dispatch_data.loc[last_car_num]['단속여부'] == 'overload':
                st.write("YES")
                st.write("NO")
                
            else:
                st.write("NO")
                st.write("YES")

    st.markdown("#### 불법차량 로그")
    st.dataframe(dispatch_data, width=950)

#################### Plotting ####################
else:
    st.title(page)
    # Side bar setting
    sel = ['본부별', '지사별', '영업소별']
    select = st.sidebar.radio("Choose buttons", options=sel)
    
    fake['일시'] = pd.to_datetime(fake['일시']).dt.strftime("%Y-%m-%d")
    max_point = fake.shape[0] - 1
    min_date = datetime.datetime.strptime(fake['일시'][0], "%Y-%m-%d")
    max_date = datetime.datetime.strptime(fake['일시'][max_point], "%Y-%m-%d")
    fake['단속여부'] = fake['단속여부'].map({'단속': 1, '통과': 0})

    slider_date = st.sidebar.slider("기간", min_date, max_date, value=(min_date, max_date))
    text = st.sidebar.text_input('영업소 입력')

    # 1~2번째 행(교통량/단속건수 비교 및 순위)
    ## 본부별
    if select == '본부별':
        data_1 = data.groupby(by='본부명', as_index=False)[['교통량', '단속건수', '과적', '불법개조']].sum()
        data_1 = data_1[data_1['본부명'] != '지역본부']
        data_1 = data_1[data_1['본부명'] != '한국도로공사']

        col310, col311 = st.columns([0.3, 0.1])
        with col310:
            st.markdown('###### 본부별 교통량 비교')
            plost.bar_chart(data=data_1,
                            height=200, width=800,
                            bar='본부명',
                            value='교통량',
                            color='powderblue',
                            opacity=0.7
                            )

        with col311:
            data_1_sort = data_1.sort_values(by='교통량', ascending=False)
            data_1_sort.reset_index(drop=True, inplace=True)
            data_1_sort = data_1_sort.iloc[0:2]
            st.markdown('###### 본부별 교통량 Top2')
            plost.bar_chart(data=data_1_sort,
                            height=190, width=200,
                            bar='본부명',
                            value='교통량',
                            direction='horizontal',
                            color='powderblue',
                            opacity=0.7
                            )

        col320, col321 = st.columns([0.3, 0.1])
        with col320:
            st.markdown('###### 본부별 단속건수 비교')
            plost.bar_chart(data=data_1,
                            height=110, width=85,
                            bar='본부명',
                            value=['과적', '불법개조'],
                            group=True,
                            opacity=0.7
                            )

        with col321:
            data_1_sort = data_1.sort_values(by=['과적', '불법개조'], ascending=False)
            data_1_sort.reset_index(drop=True, inplace=True)
            data_1_sort = data_1_sort.iloc[0:2]
            st.markdown('###### 본부별 단속건수 Top2')
            plost.bar_chart(data=data_1_sort,
                            height=65, width=130,
                            bar='본부명',
                            value=['과적', '불법개조'],
                            group=True,
                            direction='horizontal',
                            legend=None,
                            opacity=0.7
                            )

    ## 지사별
    elif select == '지사별':
        data_2 = data.groupby(by='지사명', as_index=False)[['교통량', '단속건수', '과적', '불법개조']].sum()
        data_2 = data_2[data_2['지사명'].str.contains('본부') == False]

        data_2_sort = data_2.sort_values(by='교통량', ascending=False)
        data_2_sort = data_2_sort.iloc[0:10]
        data_2_sort_top = data_2_sort.iloc[0:2]

        col310, col311 = st.columns([0.3, 0.1])
        with col310:
            st.markdown('###### 지사별 교통량 비교(Top10)')
            plost.bar_chart(data=data_2_sort,
                            height=200, width=800,
                            bar='지사명',
                            value='교통량',
                            color='powderblue',
                            opacity=0.7
                            )

        with col311:
            st.markdown('###### 지사별 교통량 Top2')
            plost.bar_chart(data=data_2_sort_top,
                            height=190, width=200,
                            bar='지사명',
                            value='교통량',
                            direction='horizontal',
                            color='powderblue',
                            opacity=0.7
                            )

        data_2_sort_ill = data_2.sort_values(by='단속건수', ascending=False)
        data_2_sort_ill = data_2_sort_ill.iloc[0:10]
        data_2_sort_ill_top = data_2_sort_ill.iloc[0:2]

        col320, col321 = st.columns([0.3, 0.1])
        with col320:
            st.markdown('###### 지사별 단속건수 비교(Top10)')
            plost.bar_chart(data=data_2_sort_ill,
                            height=110, width=65,
                            bar='지사명',
                            value=['과적', '불법개조'],
                            group=True,
                            opacity=0.7
                            )

        with col321:
            st.markdown('###### 지사별 단속건수 Top2')
            plost.bar_chart(data=data_2_sort_ill_top,
                            height=65, width=130,
                            bar='지사명',
                            value=['과적', '불법개조'],
                            group=True,
                            direction='horizontal',
                            legend=None,
                            opacity=0.7
                            )

    ## 영업소별
    else:
        data_3 = data.groupby(by='단속장소', as_index=False)[['교통량', '단속건수', '과적', '불법개조']].sum()
        data_3 = data_3[data_3['단속장소'].str.contains('지사') == False]

        data_3_sort = data_3.sort_values(by='교통량', ascending=False)
        data_3_sort = data_3_sort.iloc[0:10]
        data_3_sort_top = data_3_sort.iloc[0:2]

        col310, col311 = st.columns([0.3, 0.1])
        with col310:
            st.markdown('###### 영업소별 교통량 비교(Top10)')
            plost.bar_chart(data=data_3_sort,
                            height=200, width=800,
                            bar='단속장소',
                            value='교통량',
                            color='powderblue',
                            opacity=0.7
                            )

        with col311:
            st.markdown('###### 영업소별 교통량 Top2')
            plost.bar_chart(data=data_3_sort_top,
                            height=190, width=200,
                            bar='단속장소',
                            value='교통량',
                            direction='horizontal',
                            color='powderblue',
                            opacity=0.7
                            )

        data_3_sort_ill = data_3.sort_values(by='단속건수', ascending=False)
        data_3_sort_ill = data_3_sort_ill.iloc[0:10]
        data_3_sort_ill_top = data_3_sort_ill.iloc[0:2]

        col320, col321 = st.columns([0.3, 0.1])
        with col320:
            st.markdown('###### 영업소별 단속건수 비교(Top10)')
            plost.bar_chart(data=data_3_sort_ill,
                            height=110, width=65,
                            bar='단속장소',
                            value=['과적', '불법개조'],
                            group=True,
                            opacity=0.7
                            )

        with col321:
            st.markdown('###### 영업소별 단속건수 Top2')
            plost.bar_chart(data=data_3_sort_ill_top,
                            height=65, width=130,
                            bar='단속장소',
                            value=['과적', '불법개조'],
                            group=True,
                            direction='horizontal',
                            legend=None,
                            opacity=0.7
                            )

    # 3번째 행(일별/주별/월별 단속 건수)
    ## 선택된 일자의 data 추출
    fake['일별'] = pd.to_datetime(fake['일시'])
    day_list_df = fake[(fake['일별'] >= slider_date[0]) & (fake['일별'] <= slider_date[-1])]

    ## 선택된 주간의 data 추출
    fake['주별'] = fake['일별'].apply(lambda x: (x - min_date).days // 7 + 1)
    try:
        min_week = int(fake[fake['일별'] == slider_date[0]]['주별'].mean())
    except:
        min_week = (slider_date[0] - min_date).days // 7 + 1

    try:
        max_week = int(fake[fake['일별'] == slider_date[-1]]['주별'].mean())
    except:
        max_week = (slider_week[-1] - min_date).days // 7 + 1
    week_list_df = fake[(fake['주별'] >= min_week) & (fake['주별'] <= max_week)]

    ## 선택된 월의 data 추출
    fake['월별'] = fake['일별'].apply(lambda x: (x.year - min_date.year) * 12 + (x.month - min_date.month) + 1)
    try:
        min_month = int(fake[fake['일별'] == slider_date[0]]['월별'].mean())
    except:
        min_month = (slider_date[0].year - min_date.year) * 12 + (slider_date[0].month - min_date.month) + 1

    try:
        max_month = int(fake[fake['일별'] == slider_date[-1]]['월별'].mean())
    except:
        max_month = (slider_date[-1].year - min_date.year) * 12 + (slider_date[-1].month - min_date.month) + 1
    month_list_df = fake[(fake['월별'] >= min_month) & (fake['월별'] <= max_month)]

    st.markdown("###### 일별/주별/월별 단속 건수")
    group_day = day_list_df.groupby(by=['단속 영업소', '일별'], as_index=False)[['단속여부']].sum()
    group_day = group_day.rename(columns={'단속여부':'단속건수'})

    group_week = week_list_df.groupby(by=['단속 영업소', '주별'], as_index=False)[['단속여부']].sum()
    group_week = group_week.rename(columns={'단속여부':'단속건수'})

    group_month = month_list_df.groupby(by=['단속 영업소', '월별'], as_index=False)[['단속여부']].sum()
    group_month = group_month.rename(columns={'단속여부':'단속건수'})

    ## 영업소 입력하면 metric 및 일별/주별/월별 통계 출력
    col330, col331, col332 = st.columns([0.2, 0.2, 0.2])
    if text:
        with col330: 
            text_df = group_day[group_day['단속 영업소'] == text]
            text_max = text_df['단속건수'].max()
            text_min = text_df['단속건수'].min()
            delt = int(text_max - text_min)
            st.metric(label='일별 최대 단속건수', value=text_max, delta=delt, label_visibility="visible")
            plost.bar_chart(data=text_df,
                            height=320, width=360,
                            bar='일별',
                            value='단속건수',
                            color='paleturquoise',
                            opacity=1
                            )

        with col331:
            text_df = group_week[group_week['단속 영업소'] == text]
            text_max = text_df['단속건수'].max()
            text_min = text_df['단속건수'].min()
            delt = int(text_max - text_min)
            st.metric(label='주별 최대 단속건수', value=text_max, delta=delt, label_visibility="visible")
            plost.bar_chart(data=text_df,
                            height=320, width=360,
                            bar='주별',
                            value='단속건수',
                            color='lightblue',
                            opacity=0.7
                            )

        with col332:
            text_df = group_month[group_month['단속 영업소'] == text]
            text_max = text_df['단속건수'].max()
            text_min = text_df['단속건수'].min()
            delt = int(text_max - text_min)
            st.metric(label='월별 최대 단속건수', value=text_max, delta=delt, label_visibility="visible")
            plost.bar_chart(data=text_df,
                            height=320, width=360,
                            bar='월별',
                            value='단속건수',
                            color='lightslategrey',
                            opacity=0.7
                            )
    else:
        st.write('None')


    # 4번째 행
    col340, col341 = st.columns([0.2, 0.2])

    ## 연도별 단속건수 추이
    with col340:
        data_year = data.groupby(by='연도', as_index=False)[['교통량', '단속건수', '과적', '불법개조']].sum()
        data_year.rename(columns={'연도': 'year', '과적': 'overload', '불법개조': 'illegal'}, inplace=True)
        
        st.markdown('###### 연도별 단속건수 추이')

        fig, ax1 = plt.subplots()
        fig = plt.figure(figsize=(5,4))
        sns.set_style("darkgrid")
        sns.set_context("paper", font_scale = .7, rc={'lines.linewidth': 1,
                                                      'patch.linewidth': 0.5,
                                                      'ytick.major.width': 0.3,
                                                      'axes.labelsize': 10,
                                                      })
        ax1 = sns.lineplot(data=data_year, x='year', y='overload', label='overload', marker='o', color='lightblue')
        ax1.set_ylim(75000, 220000)
        plt.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2 = sns.lineplot(data=data_year, x='year', y='illegal', label='illegal', marker='o', color='lightpink')
        ax2.set_ylim(4000, 12000)
        plt.legend()

        sns.despine(top=True, right=False)
        plt.xticks(np.arange(2018, 2023, step=1))
        st.pyplot(fig)

    ## 차종별 단속건수(nivo_chart)
    with col341:
        st.markdown('###### 차종별 단속건수')
        with elements("nivo_charts"):

            DATA = [ 
                { "차종": "3종차량", "illegal": 4500, "overload(X10)": 8009},
                { "차종": "4종차량", "illegal": 1107, "overload(X10)": 9002},
                { "차종": "5종차량", "illegal": 8302, "overload(X10)": 9006},
                { "차종": "6+7종차량", "illegal": 689, "overload(X10)": 1202},
            ]

            with mui.Box(sx={"height": 385}):
                nivo.Radar(
                    data=DATA,
                    keys=['illegal', 'overload(X10)'],
                    indexBy=['차종'],
                    valueFormat=">-.2f",
                    margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                    borderColor={ "from": "color",
                                 'modifiers': [
                                        ['darker', .6],
                                        ['opacity', .5]
                                    ]
                                },
                    gridLabelOffset=36,
                    dotSize=10,
                    dotColor={ "theme": "background" },
                    dotBorderWidth=2,
                    colors={ 'scheme': 'pastel1' },
                    blendMode="multiply",
                    motionConfig="wobbly",
                    legends=[
                        {
                            "anchor": "top-left",
                            "direction": "column",
                            "translateX": -50,
                            "translateY": -40,
                            "itemWidth": 80,
                            "itemHeight": 20,
                            "itemTextColor": "#999",
                            "symbolSize": 12,
                            "symbolShape": "circle",
                            "effects": [
                                {
                                    "on": "hover",
                                    "style": {
                                        "itemTextColor": "#000"
                                    }
                                }
                            ]
                        }
                    ],
                    theme={
                        "background": "#FFFFFF",
                         "textColor": "#31333F",
                        "tooltip": {
                            "container": {
                                "background": "#FFFFFF",
                                "color": "#31333F",
                            }
                        }
                    }
                )