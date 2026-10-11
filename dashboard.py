"""CMP Process QE — read-only evidence dashboard (Streamlit).

Outputs are from PHM 2016 CMP public data except the clearly separated R2R tab,
which contains entirely simulated results. Never describe simulation as fab gains.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.dashboard_data import read_sources, relative_mae_gain, r2r_relative_gain

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CMP Process QE | 製程品質分析', page_icon='🧪', layout='wide')

@st.cache_data(show_spinner='載入並驗證既有分析結果…')
def get_data():
    return read_sources(ROOT)

st.title('CMP 製程品質監控與異常分析')
st.caption('公開 PHM 2016 CMP 衍生分析結果 + **獨立人工 R2R 模擬**｜唯讀成果展示，不重新訓練模型')
try:
    d = get_data()
except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
    st.error('找不到完整分析結果，或輸出檔案無法通過一致性檢查。請勿使用錯誤資料繪圖。')
    st.code(str(exc))
    st.info('請確認 Step 04A、04B、05C、05E、06D、07A、07C 已完成。詳細所需檔案見 README_STEP08A.md。')
    st.stop()

p_spc, p_fdc, p_vm, p_r2r, p_scope = st.tabs(['01｜SPC 統計監控', '02｜FDC 與人工回查', '03｜虛擬量測 VM', '04｜R2R 模擬', '資料來源與限制'])

with p_spc:
    st.subheader('Stage A｜示範性 I Chart，不是工廠正式 SPC')
    st.warning('圖中的 CL／UCL／LCL 由不穩定且具時間相關性的 Training 資料估計；超界不等於不良或設備故障。')
    group = st.selectbox('設備代碼組', ['Codes 1-3','Codes 4-6'], key='spc_group')
    data = d.spc.loc[d.spc.code_group == group].sort_values('order_within_group')
    meta = d.spc_summary.loc[d.spc_summary.code_group == group].iloc[0]
    ar = d.ar1.loc[d.ar1.code_group == group].iloc[0]
    a,b,c = st.columns(3)
    a.metric('示範 I Chart 後段超界', int(meta.monitor_outside_3sigma_count))
    b.metric('AR(1) 殘差後段超界', int(ar.AR1_monitor_flags))
    c.metric('後段監控筆數', int(meta.monitor_rows_later_40pct))
    fig=go.Figure()
    for zone,col in [('reference','#5793c5'),('monitor','#e9a247')]:
        ds=data[data.zone==zone]
        fig.add_trace(go.Scatter(x=ds.order_within_group,y=ds.AVG_REMOVAL_RATE,mode='markers',name=('前 60% 參考' if zone=='reference' else '後 40%'),marker=dict(color=col,size=5)))
    lo=float(meta.reference_LCL_provisional);hi=float(meta.reference_UCL_provisional);center=float(meta.reference_mean_CL_provisional)
    for val,name,dash,color in [(lo,'示範 LCL','dash','#b84141'),(center,'示範中心線','solid','#444444'),(hi,'示範 UCL','dash','#b84141')]:
        fig.add_hline(y=val,line_color=color,line_dash=dash,annotation_text=name,annotation_position='right')
    flagged=data.loc[data.outside_provisional_3sigma.astype(bool)]
    fig.add_trace(go.Scatter(x=flagged.order_within_group,y=flagged.AVG_REMOVAL_RATE,mode='markers',name='示範超界',marker=dict(symbol='x',color='red',size=9)))
    fig.update_layout(xaxis_title='依首次感測時間排序的順序（非已驗證產線順序）',yaxis_title='MRR（發布資料尺度）',height=500)
    st.plotly_chart(fig,use_container_width=True)
    with st.expander('查看 I Chart 與 AR(1) 的摘要資料'):
        st.dataframe(pd.concat([d.spc_summary.loc[d.spc_summary.code_group==group].reset_index(drop=True), d.ar1.loc[d.ar1.code_group==group].reset_index(drop=True)],axis=1).loc[:,lambda df: ~df.columns.duplicated()],use_container_width=True)

with p_fdc:
    st.subheader('感測數值偏差 → 品質對照 → 人工回查')
    st.warning('11 片是探索性的待查候選，不是 11 次已確認故障；MRR Q05–Q95 不是合格規格。')
    a,b,c=st.columns(3)
    a.metric('感測偏差候選晶圓',len(d.fdc_cases))
    b.metric('觸發的感測項目',len(d.fdc_signals))
    c.metric('候選 MRR 超出描述性區間',int(d.fdc_cases.outside_reference_MRR_q05_q95_descriptive.fillna(False).astype(bool).sum()))
    bar = px.bar(d.fdc_summary,x='MACHINE_DATA',y=['later_any_family','later_two_or_more_families'],barmode='group',labels={'MACHINE_DATA':'MACHINE_DATA 匿名代碼','value':'晶圓－代碼筆數','variable':'判別'},title='Training 後段的感測數值偏離數量')
    st.plotly_chart(bar,use_container_width=True)
    st.caption('以下保留個案原始識別碼，僅供公開資料的分析追溯；不涉及真實人員資訊。')
    ids=d.fdc_cases.WAFER_ID.astype(str).tolist()
    selected=st.selectbox('選擇要人工回查的晶圓',ids,index=ids.index('33494120') if '33494120' in ids else 0)
    case=d.fdc_cases[d.fdc_cases.WAFER_ID.astype(str)==selected].iloc[0]
    sig=d.fdc_signals[d.fdc_signals.WAFER_ID.astype(str)==selected].copy()
    st.write(f"**Code {int(case.MACHINE_DATA)}｜MRR = {case.measured_MRR:.3f}｜參考 MRR Q05–Q95 = {case.reference_MRR_q05:.2f}–{case.reference_MRR_q95:.2f}**")
    view=sig[['signal','family','direction','observed_median_nonzero','reference_q01','reference_q99']].rename(columns={'signal':'訊號','family':'類別','direction':'偏離方向','observed_median_nonzero':'非零值中位數','reference_q01':'參考 Q01','reference_q99':'參考 Q99'})
    st.dataframe(view,use_container_width=True,hide_index=True)
    with st.expander('查看 11 片完整候選名單'):
        st.dataframe(d.fdc_cases,use_container_width=True,hide_index=True)
    st.info('建議回查：先核對感測記錄與量測時點，再比對設備設定、耗材及保養歷史，交由製程／設備工程師確認。沒有實際執行停機或製程調整。')

with p_vm:
    st.subheader('Virtual Metrology｜內部 Test（一次性凍結）')
    st.info('使用完整研磨過程的感測特徵，於加工完成後、正式量測前預測 MRR；不是加工過程中的即時預警。')
    group=st.selectbox('選擇設備代碼組', ['Codes 1-3','Codes 4-6'],key='vm_group')
    m=d.vm_metrics[d.vm_metrics.group==group]
    base=m[m.model=='training_group_median'].iloc[0];hgb=m[m.model=='HGB_absolute_error'].iloc[0]
    a,b,c=st.columns(3)
    a.metric('HGB Test MAE',f'{hgb.all_mae:.3f}',delta=f'{relative_mae_gain(d,group):.1f}% 比基準低',delta_color='normal')
    b.metric('HGB Test RMSE',f'{hgb.all_rmse:.3f}')
    c.metric('HGB Test R²',f'{hgb.all_r2:.3f}')
    pred=d.vm_predictions[d.vm_predictions.code_family==group]
    fig=px.scatter(pred,x='actual_MRR',y='predicted_MRR',color='model',labels={'actual_MRR':'實際 MRR（發布尺度）','predicted_MRR':'預測 MRR','model':'模型'},title='最終內部 Test｜Baseline vs HGB')
    limlo=float(min(pred.actual_MRR.min(),pred.predicted_MRR.min()));limhi=float(max(pred.actual_MRR.max(),pred.predicted_MRR.max()))
    fig.add_trace(go.Scatter(x=[limlo,limhi],y=[limlo,limhi],mode='lines',line=dict(color='gray',dash='dash'),name='理想預測'))
    fig.update_layout(height=470)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(m[['model','test_rows','all_mae','all_rmse','all_r2']],hide_index=True,use_container_width=True)
    st.warning('這是從 PHM 原始 Training 內部切出的 Test，不是官方競賽外部 Test；曾出現的極端 MRR 在這次 Test 未重現，不能聲稱模型能處理該情況。')

with p_r2r:
    st.subheader('EWMA R2R｜完全獨立的人工模擬')
    st.error('**非真實 PHM／Micron 工廠控制。** 目標 75、設定值與增益皆是假設數值；請勿拿此處改善百分比宣稱產線改善。')
    a,b,c,dcol=st.columns(4)
    a.metric('無控制 MAE',f"{d.r2r_comparison.loc[d.r2r_comparison.metric=='MAE_from_target','open_loop'].iloc[0]:.3f}")
    b.metric('EWMA MAE',f"{d.r2r_comparison.loc[d.r2r_comparison.metric=='MAE_from_target','closed_loop'].iloc[0]:.3f}")
    c.metric('模擬 MAE 降低',f'{r2r_relative_gain(d):.1f}%')
    dcol.metric('設定值實際變動次數',int(d.r2r_comparison.loc[d.r2r_comparison.metric=='setpoint_change_count','closed_loop'].iloc[0]))
    end=st.slider('顯示模擬前 N 輪',min_value=10,max_value=120,value=120,step=1)
    left=d.open_runs.head(end);right=d.closed_runs.head(end)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=left.run,y=left.measured_MRR_simulated,name='固定設定值',mode='lines'))
    fig.add_trace(go.Scatter(x=right.run,y=right.measured_MRR_simulated,name='EWMA 回授',mode='lines'))
    fig.add_hline(y=75,line_color='black',line_dash='dash',annotation_text='假設目標 75')
    fig.add_vline(x=61,line_color='gray',line_dash='dot')
    fig.update_layout(title='相同人工漂移與雜訊下的 MRR',xaxis_title='模擬輪次',yaxis_title='模擬 MRR（任意尺度）',height=470)
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=left.run,y=left.setpoint_applied,name='無控制設定值'))
    fig2.add_trace(go.Scatter(x=right.run,y=right.setpoint_applied,name='EWMA 設定值'))
    for bound in [42,58]:fig2.add_hline(y=bound,line_color='gray',line_dash='dot')
    fig2.update_layout(title='設定值調整（假設範圍 42～58）',xaxis_title='模擬輪次',yaxis_title='假設設定值',height=360)
    st.plotly_chart(fig2,use_container_width=True)
    st.dataframe(d.r2r_comparison,hide_index=True,use_container_width=True)
    st.caption('控制器只使用上一輪已完成的量測；沒有取得模擬器隱藏的漂移、真實增益與雜訊。回授能降低偏差，但增加 116 次設定值變動。')

with p_scope:
    st.subheader('資料來源與證據等級')
    st.markdown('''
| 模組 | 實際使用的資料 | 不能宣稱 |
|---|---|---|
| SPC | PHM 2016 CMP Training Stage A、分代碼 MRR | 工廠已核准管制界限、良率、Cpk |
| FDC | PHM 感測特徵、凍結候選、MRR 對照 | 已確認故障、故障類型、根因 |
| VM | PHM Training 內部分割、一次性 Test | 官方外部 Test、即時預測或真實上線 |
| R2R | 獨立人工 120 輪製程模擬 | PHM／Micron 實際控制改善 |
''')
    st.caption('本儀表板是唯讀呈現。修改分析程式或真實輸出後，應先重新驗證；不可為求更漂亮的內部 Test 或 R2R 模擬成績重選模型／參數。')
    st.success('已核對輸出：SPC、FDC、VM 與 R2R 報告筆數、VM 指標及 R2R 配對條件一致。')
