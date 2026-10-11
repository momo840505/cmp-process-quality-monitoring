"""Public, aggregate-only showcase for the PHM CMP research + separate R2R simulation.

This app intentionally does not load raw PHM data, per-wafer features, or models.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.public_demo_data import load_public_data

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='CMP｜製程品質分析作品',page_icon='🔬',layout='wide')

@st.cache_data(show_spinner='核對公開展示資料…')
def get_data():
    return load_public_data(ROOT)

try:
    summary,sim=get_data()
except (KeyError,ValueError,FileNotFoundError,TypeError) as e:
    st.error('公開展示資料缺失或驗證失敗。請先檢查 public_data 檔案。')
    st.code(str(e));st.stop()

st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:2.4rem;max-width:1180px;}
h1{font-size:2.0rem !important;letter-spacing:.01em;}
h2{font-size:1.4rem !important;}
div[data-testid="stMetric"]{border:1px solid #d9e2ee;border-radius:12px;padding:12px 16px;}
div[data-testid="stTabs"] button{font-weight:600;}
</style>""",unsafe_allow_html=True)

st.title('半導體 CMP｜製程品質監控與分析')
st.caption('PHM 2016 CMP 公開研究資料的彙總成果 ＋ 獨立的人工 R2R 模擬｜作品展示版')
st.info('**公開版的資料範圍：** 只有彙總指標、已產生的分析圖片及人工模擬資料。沒有公開逐片晶圓資料、正式製程界限或已部署工廠控制。')

t1,t2,t3,t4,t5=st.tabs(['01｜SPC 統計監控','02｜FDC 感測回查','03｜品質預測 VM','04｜R2R 人工模擬','資料來源與限制'])

with t1:
    st.subheader('先釐清製程群組，再比較監控方法')
    group=st.selectbox('選擇匿名代碼組',['Codes 1-3','Codes 4-6'],key='pub_spc')
    s=summary['spc'][group]
    a,b,c=st.columns(3)
    a.metric('I Chart 示範後段超界',f"{s['i_chart_flags']} / {s['monitor_n']}")
    b.metric('AR(1) 殘差示範超界',f"{s['ar1_flags']} / {s['monitor_n']}")
    c.metric('參考段 Lag-1 相關',f"{s['reference_lag1']:.3f}")
    st.image(str(ROOT/'docs/figures'/('spc_codes_1_3.png' if group=='Codes 1-3' else 'spc_codes_4_6.png')),use_container_width=True)
    st.warning('這張圖片來自實際分析輸出，但 CL／UCL／LCL 只是探索性示範，不是工廠核准的 SPC 界限。紅色叉號不等於不良品；AR(1) 警報較少也不代表故障偵測較準。')
    st.caption('圖片為先前固定的 Training Stage A 分析結果；公開展示不載入原始逐片資料。')

with t2:
    st.subheader('感測訊號偏差 → 需要回查的候選')
    f=summary['fdc']
    a,b,c=st.columns(3)
    a.metric('待查晶圓',f['distinct_candidate_wafers'])
    b.metric('觸發感測數值',f['sensor_trigger_records'])
    c.metric('候選超出 MRR 描述性區間',f['candidate_mrr_outside_descriptive_interval'])
    x=pd.DataFrame(f['by_code']);x['代碼']='Code '+x.code.astype(str)
    bar=pd.DataFrame({'匿名代碼':x['代碼'],'至少一類偏差':x['any_family'],'至少兩類同時偏差':x['two_or_more']})
    fig=px.bar(bar,x='匿名代碼',y=['至少一類偏差','至少兩類同時偏差'],barmode='group',labels={'value':'晶圓－代碼筆數','variable':'篩選條件'})
    fig.update_layout(height=380,legend_title_text='')
    st.plotly_chart(fig,use_container_width=True)
    with st.expander('查看各代碼彙總（不含逐片資料）'):
        view=x[['代碼','screening_n','usable_features','any_family','two_or_more']].rename(columns={'screening_n':'後段檢查筆數','usable_features':'可用的數值特徵','any_family':'至少一類偏差','two_or_more':'兩類以上偏差'})
        st.dataframe(view,use_container_width=True,hide_index=True)
        sig=pd.DataFrame([{'訊號':k,'觸發筆數':v} for k,v in f['signal_type_counts'].items()])
        st.dataframe(sig,use_container_width=True,hide_index=True)
    st.warning('這 11 片是探索性的感測偏差候選，不是已證實故障；MRR Q05–Q95 也不是製程良品規格。完整逐片人工回查請在本機 dashboard.py 檢視。')

with t3:
    st.subheader('Virtual Metrology｜製程結束後、正式量測前預測 MRR')
    group=st.selectbox('選擇匿名代碼組',['Codes 1-3','Codes 4-6'],key='pub_vm')
    v=summary['vm'][group]
    a,b,c=st.columns(3)
    a.metric('最終內部 Test MAE',f"{v['model_mae']:.3f}",f"比中位數基準降低 {v['improvement_pct']:.1f}%")
    b.metric('最終內部 Test RMSE',f"{v['model_rmse']:.3f}")
    c.metric('最終內部 Test R²',f"{v['model_r2']:.3f}")
    tab=pd.DataFrame({'方法':['訓練組中位數基準','HGB（66 個感測特徵）'],'MAE':[v['baseline_mae'],v['model_mae']],'RMSE':[v['baseline_rmse'],v['model_rmse']]})
    fig=px.bar(tab,x='方法',y=['MAE','RMSE'],barmode='group',title=f'{group}｜內部 Test {v["test_n"]} 片（誤差越小越好）')
    fig.update_layout(height=340)
    st.plotly_chart(fig,use_container_width=True)
    with st.expander('查看固定的最終 Test 散點圖'):
        st.image(str(ROOT/'docs/figures'/('vm_codes_1_3.png' if group=='Codes 1-3' else 'vm_codes_4_6.png')),use_container_width=True)
    st.warning('這是原始 PHM Training 的一次性內部 Test，不是競賽外部 Test 或真實上線成效。極端 MRR 在此 Test 未出現，不能宣稱模型能處理那些情況。')

with t4:
    st.subheader('R2R｜固定設定 vs EWMA 延遲回授（純人工模擬）')
    r=summary['r2r']
    a,b,c,d=st.columns(4)
    a.metric('固定設定 MAE',f"{r['mae_open']:.3f}")
    b.metric('EWMA 回授 MAE',f"{r['mae_ewma']:.3f}")
    c.metric('模擬 MAE 降低',f"{r['reduction_pct']:.1f}%")
    d.metric('設定值變動次數',str(r['adjustment_count']))
    count=st.slider('顯示前幾輪人工模擬',min_value=10,max_value=120,value=120,step=1)
    frame=sim.head(count)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=frame.run,y=frame.open_mrr,mode='lines',name='固定設定'))
    fig.add_trace(go.Scatter(x=frame.run,y=frame.ewma_mrr,mode='lines',name='EWMA 回授'))
    fig.add_hline(y=r['target'],line_dash='dash',annotation_text='假設 MRR 目標 75')
    fig.add_vline(x=61,line_dash='dot',line_color='gray')
    fig.update_layout(height=420,xaxis_title='人工模擬輪次',yaxis_title='模擬量測 MRR（假設尺度）')
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=frame.run,y=frame.open_setting,name='固定設定'))
    fig2.add_trace(go.Scatter(x=frame.run,y=frame.ewma_setting,name='EWMA 設定'))
    for bound in [42,58]:fig2.add_hline(y=bound,line_dash='dot',line_color='gray')
    fig2.update_layout(height=300,xaxis_title='人工模擬輪次',yaxis_title='假設設定值')
    with st.expander('查看設定值調整過程及代價'):
        st.plotly_chart(fig2,use_container_width=True)
        st.write(f"最後 30 輪平均偏差：固定設定 **{r['last30_bias_open']:+.3f}**；EWMA **{r['last30_bias_ewma']:+.3f}**。總設定變動幅度：**{r['adjustment_movement']:.2f}**。")
    st.error('這是獨立設計的 120 輪人工模擬，不使用 PHM／Micron 真實設備設定或回授。74.6% 絕不能當成真實產線改善率。')

with t5:
    st.subheader('使用範圍與證據等級')
    st.markdown("""| 模組 | 展示資料 | 不能推論 |
|---|---|---|
| SPC | PHM Training Stage A 的探索性研究彙總和既有圖 | 已核准管制界限、不良率 |
| FDC | 凍結的感測偏差候選彙總 | 已確認故障、故障根因 |
| VM | PHM Training 內部一次性 Test 評估彙總 | 外部競賽分數、工廠上線 |
| R2R | 120 輪獨立人工模擬 | 真實設備控制改善 |""")
    st.write('**公開版與本機版的差別：** 公開版可線上查看五個頁籤、切換群組及操作模擬滑桿；本機 `dashboard.py` 才可在合法取得資料後深入逐片調查。')
    st.caption('所有彙總結果均由先前固定輸出匯出；沒有在 Streamlit 啟動過程重訓模型、重新評估內部 Test 或調整 R2R 參數。')
