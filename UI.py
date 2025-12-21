import gradio as gr # type: ignore
import joblib # type: ignore
import pandas as pd
import sklearn # type: ignore
from sklearn.preprocessing import StandardScaler
# Define the paths where the scaler and model were saved in Google Drive
model_path = 'best_random_forest_binned_model.joblib'
scaler_b_path = 'scaler_b.joblib'

# Load the saved scaler and model
try:
    loaded_scaler_b = joblib.load(scaler_b_path)
    loaded_model = joblib.load(model_path)
    print("Scaler and model loaded successfully!")
except FileNotFoundError:
    print("Error: Make sure the files exist in your Google Drive and the paths are correct.")
except Exception as e:
    print(f"An error occurred while loading: {e}")
feature_names = [
        "ROA (B) before interest and depreciation after tax_binned",
        "Operating Gross Margin_binned",
        "Persistent EPS in the Last Four Seasons_binned",
        "Gross Profit to Sales_binned",
        "Cash / Total Assets_binned",
        "Debt Ratio %_binned",
        "Net Worth / Assets_binned",
        "Liability to Equity_binned",
        "Cash Flow Rate_binned",
        "Cash Flow Per Share_binned",
        "CFO to Assets_binned",
        "Cash Flow to Equity_binned",
        "Cash Flow to Liabilities_binned",
        "After-tax Net Profit Growth Rate_binned"
    ]
loaded_scaler_b.feature_names_in_ = feature_names

def predict(
    roa_b,
    Operating_Gross_Margin,
    Persistent_EPS,
    Gross_Profit_to_Sales,
    Cash_Total_Assets,
    Debt_Ratio,
    Net_Worth_Assets,
    Liability_to_Equity,
    Cash_Flow_Rate,
    Cash_Flow_Per_Share,
    CFO_to_Assets,
    Cash_Flow_To_Equity,
    Cash_Flow_To_Liabilities,
    After_tax_Net_Profit_Growth_Rate,
): 
    # binning each value to specific bin
    roa_bins = [-float('inf'),
               0.491782215322019, 0.520370469511216,
               0.533004978853258, 0.543069757481664,
               0.552277959205525, 0.5628138551314312,
               0.575940896193586, 0.5937148669629,
               0.620589967343005,
               float('inf')]
    ogm_bins = [
        -float('inf'),
        0.596513354184984,
        0.599208694273483,
        0.6014961299528674,
        0.6036696983237,
        0.605997492036495,
        0.608678418541634,
        0.6119185920811776,
        0.6161893368310298,
        0.6231525389527092,
        +float('inf')
    ]
    p_eps_bins = [
        -float('inf'),
        0.201096719296587, 0.211213009359932,
        0.216507516308972, 0.22030821594024744,
        0.22454382149948, 0.228987425545996,
        0.234660111562825, 0.243641864422804,
        0.25878793608773737,
        +float('inf')
    ]
    g_p_t_s = [
        -float('inf'),
        0.5965136790102759,
        0.5992048725443971,
        0.6014962824757836,
        0.6036702425196689,
        0.605998288167218,
        0.6086789984721334,
        0.6119166914368688,
        0.6161859517112064,
        0.6231522699388202,
        +float('inf')
        ]
    c_t_a = [
        -float('inf'),
        0.014580203321604642,
        0.02702683314934822,
        0.039568104652166564,
        0.055711704518551126,
        0.0748874639354301,
        0.09996370921794173,
        0.1359244947428953,
        0.19292890354797385,
        0.2995855588101138,
        +float('inf')
        ]
    d_r = [
        -float('inf'),
        0.04405478126795057,
        0.06369380536324333,
        0.0809928349003839,
        0.0973788433050156,
        0.111406717658796,
        0.12554342896876963,
        0.140732230855277,
        0.158236841309671,
        0.1839344559663814,
        +float('inf')
    ]
    n_w_a = [
        -float('inf'),
        0.8160655440336186,
        0.841763158690329,
        0.859267769144723,
        0.8744565710312304,
        0.888593282341204,
        0.902621156694984,
        0.919007165099616,
        0.9363061946367566,
        0.9559452187320492,
        +float('inf')
    ]
    l_t_e = [
        -float('inf'),
        0.2759386445397634,
        0.2765891025082422,
        0.27727822701517557,
        0.278028438295994,
        0.278777583629637,
        0.2796492406168916,
        0.2807757791156158,
        0.2823648898030066,
        0.2855732416167916,
        +float('inf')
    ]
    c_f_r = [
        -float('inf'),
        0.4578750744403342,
        0.4606164917843034,
        0.462197730303717,
        0.463610209439454,
        0.465079724549793,
        0.466876357749763,
        0.4694248592972834,
        0.473023158283217,
        0.4800479773198128,
        +float('inf')
    ]
    c_f_p_s = [
        -float('inf'),
        0.4578750744403342,
        0.4606164917843034,
        0.462197730303717,
        0.463610209439454,
        0.465079724549793,
        0.466876357749763,
        0.4694248592972834,
        0.473023158283217,
        0.4800479773198128,
        +float('inf')
    ]
    c_f_o_a = [
        -float('inf'),
        0.311593690316192, 0.3162976586263,
        0.318773431421094, 0.320577208743015,
        0.322487090613284, 0.324538445214685,
        0.3271415434674968, 0.33052981537808623,
        0.337023413737002, +float('inf')
    ]
    cfo_t_a = [
        -float('inf'),
        0.5300506444742535,
        0.5572796604178494,
        0.5719639545181976,
        0.582457292394041,
        0.593266274083544,
        0.6040416429440366,
        0.6175917808831816,
        0.6335248286323122,
        0.6585020168824591,
        +float('inf')
    ]
    c_f_t_e = [
        -float('inf'),
        0.3094124816866718,
        0.3121787351231764,
        0.3135669617517418,
        0.31437364015497105,
        0.314952752072916,
        0.3157434345953432,
        0.3168407701630426,
        0.3187618623151664,
        0.3227742365828652,
        +float('inf')
    ]
    c_f_t_l = [
        -float('inf'),
        0.4502340075417116, 0.45568710337828,
        0.4580273020729182, 0.45904313701068117,
        0.459750137932885, 0.4608706941941624,
        0.462795997322695, 0.46630625926473523,
        +float('inf')
    ]
    a_t_n_p_g_r = [
        -float('inf'),
        0.6888918742926496,
        0.6892155963186279,
        0.689313185551231,
        0.6893817940537679,
        0.689438526343149,
        0.6894940294617798,
        0.6895859460425204,
        0.689722972612691,
        0.6900140379378645,
        +float('inf')
    ]
    # binning number
    labels = list(range(10))
    # binning the data
    def bin_data(data, bins):
        for j in range(len(bins)-1):
            if data >= bins[j] and data < bins[j+1]:
                return j
        return len(bins)-1 # return the last bin if data is larger than the last bin

    roa_b = bin_data(roa_b, roa_bins)
    Operating_Gross_Margin = bin_data(Operating_Gross_Margin, ogm_bins)
    Persistent_EPS = bin_data(Persistent_EPS, p_eps_bins)
    Gross_Profit_to_Sales = bin_data(Gross_Profit_to_Sales, g_p_t_s)
    Cash_Total_Assets = bin_data(Cash_Total_Assets, c_t_a)
    Debt_Ratio = bin_data(Debt_Ratio, d_r)
    Net_Worth_Assets = bin_data(Net_Worth_Assets, n_w_a)
    liability_to_Equity = bin_data(Liability_to_Equity, l_t_e)
    cash_Flow_Rate = bin_data(Cash_Flow_Rate, c_f_r)
    cash_Flow_Per_Share = bin_data(Cash_Flow_Per_Share, c_f_p_s)
    CFO_to_Assets = bin_data(CFO_to_Assets, c_f_o_a)
    cash_Flow_To_Equity = bin_data(Cash_Flow_To_Equity, cfo_t_a)
    cash_Flow_To_Liabilities = bin_data(Cash_Flow_To_Liabilities, c_f_t_l)
    After_tax_Net_Profit_Growth_Rate = bin_data(After_tax_Net_Profit_Growth_Rate, a_t_n_p_g_r)

    feature_names = [
        "ROA (B) before interest and depreciation after tax_binned",
        "Operating Gross Margin_binned",
        "Persistent EPS in the Last Four Seasons_binned",
        "Gross Profit to Sales_binned",
        "Cash / Total Assets_binned",
        "Debt Ratio %_binned",
        "Net Worth / Assets_binned",
        "Liability to Equity_binned",
        "Cash Flow Rate_binned",
        "Cash Flow Per Share_binned",
        "CFO to Assets_binned",
        "Cash Flow to Equity_binned",
        "Cash Flow to Liabilities_binned",
        "After-tax Net Profit Growth Rate_binned"
    ]
    input_data = {
        'ROA (B) before interest and depreciation after tax_binned': [roa_b],
        'Operating Gross Margin_binned': [Operating_Gross_Margin],
        'Persistent EPS in the Last Four Seasons_binned': [Persistent_EPS],
        'Gross Profit to Sales_binned': [Gross_Profit_to_Sales],
        'Cash / Total Assets_binned': [Cash_Total_Assets],
        'Debt Ratio %_binned': [Debt_Ratio],
        'Net Worth / Assets_binned': [Net_Worth_Assets],
        'Liability to Equity_binned': [Liability_to_Equity],
        'Cash Flow Rate_binned': [Cash_Flow_Rate],
        'Cash Flow Per Share_binned': [Cash_Flow_Per_Share],
        'CFO to Assets_binned': [CFO_to_Assets],
        'Cash Flow to Equity_binned': [Cash_Flow_To_Equity],
        'Cash Flow to Liabilities_binned': [Cash_Flow_To_Liabilities],
        'After-tax Net Profit Growth Rate_binned': [After_tax_Net_Profit_Growth_Rate],
    }
    input_data = pd.DataFrame(input_data, columns=feature_names)
    input_data = loaded_scaler_b.transform(input_data)
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        prediction = "Bankrupt"
    else:
        prediction = "Not Bankrupt"
    return prediction

    

with gr.Blocks() as demo:
    gr.Markdown("Company Bankrupt prediction")
    with gr.Row():
        roa_b = gr.Number(value=0.01, label="ROA (B) before interest and depreciation after tax", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Operating_Gross_Margin = gr.Number(value=0.01, label="Operating Gross Margin", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Persistent_EPS = gr.Number(value=0.01, label="Persistent EPS in the Last Four Seasons", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Gross_Profit_to_Sales = gr.Number(value=0.01, label="Gross Profit to Sales", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Cash_Total_Assets = gr.Number(value=0.001, label="Cash / Total Assets", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Debt_Ratio = gr.Number(value=0.01, label="Debt Ratio %" ,minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Net_Worth_Assets = gr.Number(value=0.001, label="Net Worth / Assets" ,minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Liability_to_Equity = gr.Number(value=0.001, label="Liability to Equity", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Cash_Flow_Rate = gr.Number(value=0.001, label="Cash Flow Rate", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Cash_Flow_Per_Share = gr.Number(value=0.001, label="Cash Flow Per Share", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        CFO_to_Assets = gr.Number(value=0.001, label="CFO to Assets", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Cash_Flow_To_Equity = gr.Number(value=0.001, label="Cash Flow to Equity", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        Cash_Flow_To_Liabilities = gr.Number(value=0.001, label="Cash Flow to Liabilities", minimum=0.0, maximum=1.0, min_width=100)
    with gr.Row():
        After_tax_Net_Profit_Growth_Rate = gr.Number(value=0.001, label="After-tax Net Profit Growth Rate", minimum=0.0, maximum=1.0, min_width=100)
    
    with gr.Row():
        predict_btn = gr.Button("Predict")
        output = gr.Textbox()
    predict_btn.click(
        fn=predict,
        inputs=[
            roa_b,
            Operating_Gross_Margin,
            Persistent_EPS,
            Gross_Profit_to_Sales,
            Cash_Total_Assets,
            Debt_Ratio,
            Net_Worth_Assets,
            Liability_to_Equity,
            Cash_Flow_Rate,
            Cash_Flow_Per_Share,
            CFO_to_Assets,
            Cash_Flow_To_Equity,
            Cash_Flow_To_Liabilities,
            After_tax_Net_Profit_Growth_Rate,
        ],
        outputs=output,
    )



demo.launch()