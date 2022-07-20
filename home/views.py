from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Create your views here.
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from pathlib import Path

df = pd.read_csv('D:/진행중업무/데이터 분석_기획/데이터 분석/data/0627_login.csv')
def home(req):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = BASE_DIR.replace('\\', '/')

    testdf = pd.read_csv(str(BASE_DIR)+'/static/datafiles/2022-05-23 22년 형성평가 풀이.csv')
    print(str(BASE_DIR)+'/static/datafiles/2022-05-23 22년 형성평가 풀이.csv')
    #print(BASE_DIR)
    chart_1 = SubjectAssessmentTransformDF(testdf, '수학6', 9062, 1).to_html()
    chart_2 = SubjectAssessmentTransformDF(testdf, '수학6', 9062, 2).to_html()
    chart_3 = SubjectAssessmentTransformDF(testdf, '수학6', 9062, 3).to_html()
    chart_4 = SubjectAssessmentTransformDF(testdf, '수학6', 9062, 4).to_html()
    chart_5 = SubjectAssessmentTransformDF(testdf, '수학6', 9062, 5).to_html()

    context = {'chart_1': chart_1, 'chart_2': chart_2, 'chart_3': chart_3, 'chart_4': chart_4, 'chart_5': chart_5}

    return render(req, 'home/welcome.html', context)


def SubjectAssessmentUnitGraph(data, subject, unit):

    plt.rcParams['hatch.linewidth'] = 0.5  # * 그래프의 빗금 선 굵기 설정

    # plus: 정답, minus: 오답
    # (1-2)xo : 1번 틀리고 2번 맞음 , (1-2)xx: 1번 틀리고 2번 틀림
    # plus_cols 칼럼 값들은 그래프 상에서 양수 값에 bar그래프를 그림
    # minus_cols 칼럼 값들은 그래프 상에서 음수 값에 bar그래프를 그림
    plus_cols = ['right_1', '(1-2)xo_cnt', '(2-3)xo_cnt', '(3-4)xo_cnt', '(4-5)xo_cnt']
    minus_cols = ['wrong_1', '(1-2)xx_cnt', '(2-3)xx_cnt', '(3-4)xx_cnt', '(4-5)xx_cnt']

    # 정/오답자 중 해설을 본 학생수도 그리기 위한 칼럼명
    plus_comm_cols = ['none', '(1-2)cmt_o_cnt', '(2-3)cmt_o_cnt', '(3-4)cmt_o_cnt', '(4-5)cmt_o_cnt']
    minus_comm_cols = ['none', '(1-2)cmt_x_cnt', '(2-3)cmt_x_cnt', '(3-4)cmt_x_cnt', '(4-5)cmt_x_cnt']

    N = 5  # 형성평가 문항 개수

    data = data.loc[data['회차코드'] == unit]  # 회차 정보 가져오기
    data['none'] = 0  # 그래프를 위한 칼럼으로 사용하지 않음, 0으로 초기화

    fig, axs = plt.subplots(1, 5, figsize=(35, 6))

    #ind = np.arange(1, N + 1)  # 하나의 그래프 안에서의 x위치 (1st, 2nd, 3rd, 4th, 5th) 형태로 5개 필요
    ind = ['1st', '2nd', ' 3rd', '4th', '5th']
    width = 0.35  # bar 1개당 넓이 설정
    line_width = width / 2  # 선그래프 시작과 끝점 설정을 위한 넓이 1/2 길이 설정
    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=5, subplot_titles=("1문항", "2문항", "3문항", "4문항", "5문항")
    )
    #str_title = subject+' '+unit+'회차'
    for i in range(N):

        # 문항별 데이터 가져오기
        tmp_data = data[data['문항번호'] == i + 1]

        # 시도횟수(1st~5th)별 cnt1(정답자수), cnt2(오답자수)
        cnt1 = tmp_data.loc[:, plus_cols].values.reshape(-1)
        cnt2 = -1 * tmp_data.loc[:, minus_cols].values.reshape(-1)

        # 시도횟수(1st~5th)별 cnt3(정답자 중 해설조회 학생수), cnt4(오답자 중 해설조회 학생수)
        cnt3 = tmp_data.loc[:, plus_comm_cols].values.reshape(-1)
        cnt4 = -1 * tmp_data.loc[:, minus_comm_cols].values.reshape(-1)

        # Bar 그래프 그리기 (시도횟수별 정/오답자 수)
        p1 = axs[i].bar(ind, cnt1, width, label='Right_Count', color='blue', alpha=0.7)
        p2 = axs[i].bar(ind, cnt2, width, label='Wrong_Count', color='red', alpha=0.7)

        # Bar 그래프 그리기 (시도횟수별 정/오답자 중 해설조회 학생수)
        axs[i].bar(ind, cnt3, width, label='Right_Comm_Count', color='blue', alpha=0.5, hatch='\\\\\\')
        axs[i].bar(ind, cnt4, width, label='Wrong_Comm_Count', color='red', alpha=0.5, hatch='\\\\\\')

        fig.add_trace(go.Bar(x=ind, y=cnt1, marker_color='#3370ff'), row=1, col=i+1)
        fig.add_trace(go.Bar(x=ind, y=cnt2, marker_color='#ee0d5f'), row=1, col=i+1)
        fig.add_trace(go.Bar(x=ind, y=cnt3, marker_color='#3370ee', marker_pattern_shape="x"), row=1, col=i+1)
        fig.add_trace(go.Bar(x=ind, y=cnt4, marker_color='#ee0d5f', marker_pattern_shape="x"), row=1, col=i+1)
        fig.update_xaxes(title_text="try counts", row=1, col=i+1)
        fig.update_layout(
            barmode='overlay',
            title_text=subject+' '+str(unit)+'회차',
            yaxis=dict(
                title_text='user counts'
            ),
           # showlegend=False
        )

    return fig

def f(x):
    d = {}
    d['cnt_1'] = x['try_1'].count()
    d['right_1'] = len(x[x['yn_1'] == 1])
    d['wrong_1'] = len(x[x['yn_1'] == 0])
    d['mean_1'] = np.round(x['yn_1'].mean(), 2)
    for i in range(1, 5):
        d['cnt_{}'.format(i + 1)] = x['try_{}'.format(i + 1)].count()
        d['right_{}'.format(i + 1)] = len(x[x['yn_{}'.format(i + 1)] == 1])
        d['wrong_{}'.format(i + 1)] = len(x[x['yn_{}'.format(i + 1)] == 0])
        d['mean_{}'.format(i + 1)] = np.round(x['yn_{}'.format(i + 1)].mean(), 2)

        d['({0}-{1})oo_cnt'.format(i, i + 1)] = x['({0}-{1})oo'.format(i, i + 1)].sum()
        d['({0}-{1})ox_cnt'.format(i, i + 1)] = x['({0}-{1})ox'.format(i, i + 1)].sum()
        d['({0}-{1})xo_cnt'.format(i, i + 1)] = x['({0}-{1})xo'.format(i, i + 1)].sum()
        d['({0}-{1})xx_cnt'.format(i, i + 1)] = x['({0}-{1})xx'.format(i, i + 1)].sum()
        d['({0}-{1})cmt_o_cnt'.format(i, i + 1)] = x['({0}-{1})cmt_o'.format(i, i + 1)].sum()
        d['({0}-{1})cmt_x_cnt'.format(i, i + 1)] = x['({0}-{1})cmt_x'.format(i, i + 1)].sum()

    return pd.Series(d)


def SubjectAssessmentTransformDF(data, subject, code, units):

    COL_INFO = ['id', '학년코드', '교과군코드', '교과군', '과목코드', '과목', '단원코드', '단원', '회차코드', '회차', '문항번호']
    COL_INFO_NUM = ['id', '학년코드', '교과군코드', '과목코드', '단원코드', '회차코드', '문항번호']
    COL_EXAM = ['시도횟수', '정답여부', '해설보기여부']

    subject_df = data.loc[(data['과목'] == subject) & (data['과목코드'] == code)]
    subject_df = subject_df[COL_INFO_NUM + COL_EXAM]

    try_df = []
    comm_df = []

    # 시도횟수(1~5) 별 칼럼(try_1~5 , yn_1~5, comm_1~5 ) 만들기
    for i in range(1, 6):
        try_df.append(subject_df[subject_df['시도횟수'] == i].rename(columns={'시도횟수': 'try_{0}'.format(i),
                                                                          '정답여부': 'yn_{0}'.format(i),
                                                                          '해설보기여부': 'comm_{0}'.format(i)}))
    # 시도횟수 = 1
    merge_df = try_df[0]

    # 시도횟수 = 2~5 merge 수행
    # 시도횟수 1~2, 2~3, 3~4, 4~5 (oo, ox, xo, xx) 카운트
    for i in range(1, 5):
        merge_df = pd.merge(merge_df, try_df[i], on=COL_INFO_NUM, how='left')

        # O(정답) - O(정답)
        merge_df.loc[
            (merge_df['yn_{0}'.format(i)] == 1) & (merge_df['yn_{0}'.format(i + 1)] == 1), '({0}-{1})oo'.format(i,
                                                                                                                i + 1)] = 1
        # O(정답) - X(오답)
        merge_df.loc[
            (merge_df['yn_{0}'.format(i)] == 1) & (merge_df['yn_{0}'.format(i + 1)] == 0), '({0}-{1})ox'.format(i,
                                                                                                                i + 1)] = 1
        # X(오답) - O(정답)
        merge_df.loc[
            (merge_df['yn_{0}'.format(i)] == 0) & (merge_df['yn_{0}'.format(i + 1)] == 1), '({0}-{1})xo'.format(i,
                                                                                                                i + 1)] = 1
        # X(오답) - X(오답)
        merge_df.loc[
            (merge_df['yn_{0}'.format(i)] == 0) & (merge_df['yn_{0}'.format(i + 1)] == 0), '({0}-{1})xx'.format(i,
                                                                                                                i + 1)] = 1

        # X(오답) - O(정답) - 해설 보았는지
        merge_df.loc[(merge_df['yn_{0}'.format(i)] == 0) & (merge_df['yn_{0}'.format(i + 1)] == 1) & (
                    merge_df['comm_{0}'.format(i)] == 'Y'), '({0}-{1})cmt_o'.format(i, i + 1)] = 1
        # X(오답) - X(오답) - 해설 보았는지
        merge_df.loc[(merge_df['yn_{0}'.format(i)] == 0) & (merge_df['yn_{0}'.format(i + 1)] == 0) & (
                    merge_df['comm_{0}'.format(i)] == 'Y'), '({0}-{1})cmt_x'.format(i, i + 1)] = 1

    df = merge_df.groupby(COL_INFO_NUM[2:], as_index=False).apply(f)

    return SubjectAssessmentUnitGraph(df, subject, units)
