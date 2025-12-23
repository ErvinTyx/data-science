import gradio as gr
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load models and scalers
try:
    # Neural Network model
    nn_model = load_model('bankruptcy_predictor_final.h5')
    nn_scaler = joblib.load('scaler_1.pkl')
    print("Neural Network model loaded successfully!")
except Exception as e:
    print(f"Error loading Neural Network model: {e}")
    nn_model = None

try:
    # Random Forest model
    rf_model = joblib.load('best_random_forest_binned_model.joblib')
    rf_scaler = joblib.load('scaler_b.joblib')
    
    # Set feature names for RF scaler
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
    rf_scaler.feature_names_in_ = feature_names
    print("Random Forest model loaded successfully!")
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    rf_model = None

# Binning definitions for Random Forest
def get_bins():
    return {
        'roa': [-float('inf'), 0.491782215322019, 0.520370469511216, 0.533004978853258, 
                0.543069757481664, 0.552277959205525, 0.5628138551314312, 0.575940896193586, 
                0.5937148669629, 0.620589967343005, float('inf')],
        'ogm': [-float('inf'), 0.596513354184984, 0.599208694273483, 0.6014961299528674, 
                0.6036696983237, 0.605997492036495, 0.608678418541634, 0.6119185920811776, 
                0.6161893368310298, 0.6231525389527092, float('inf')],
        'p_eps': [-float('inf'), 0.201096719296587, 0.211213009359932, 0.216507516308972, 
                  0.22030821594024744, 0.22454382149948, 0.228987425545996, 0.234660111562825, 
                  0.243641864422804, 0.25878793608773737, float('inf')],
        'g_p_t_s': [-float('inf'), 0.5965136790102759, 0.5992048725443971, 0.6014962824757836, 
                    0.6036702425196689, 0.605998288167218, 0.6086789984721334, 0.6119166914368688, 
                    0.6161859517112064, 0.6231522699388202, float('inf')],
        'c_t_a': [-float('inf'), 0.014580203321604642, 0.02702683314934822, 0.039568104652166564, 
                  0.055711704518551126, 0.0748874639354301, 0.09996370921794173, 0.1359244947428953, 
                  0.19292890354797385, 0.2995855588101138, float('inf')],
        'd_r': [-float('inf'), 0.04405478126795057, 0.06369380536324333, 0.0809928349003839, 
                0.0973788433050156, 0.111406717658796, 0.12554342896876963, 0.140732230855277, 
                0.158236841309671, 0.1839344559663814, float('inf')],
        'n_w_a': [-float('inf'), 0.8160655440336186, 0.841763158690329, 0.859267769144723, 
                  0.8744565710312304, 0.888593282341204, 0.902621156694984, 0.919007165099616, 
                  0.9363061946367566, 0.9559452187320492, float('inf')],
        'l_t_e': [-float('inf'), 0.2759386445397634, 0.2765891025082422, 0.27727822701517557, 
                  0.278028438295994, 0.278777583629637, 0.2796492406168916, 0.2807757791156158, 
                  0.2823648898030066, 0.2855732416167916, float('inf')],
        'c_f_r': [-float('inf'), 0.4578750744403342, 0.4606164917843034, 0.462197730303717, 
                  0.463610209439454, 0.465079724549793, 0.466876357749763, 0.4694248592972834, 
                  0.473023158283217, 0.4800479773198128, float('inf')],
        'c_f_p_s': [-float('inf'), 0.4578750744403342, 0.4606164917843034, 0.462197730303717, 
                    0.463610209439454, 0.465079724549793, 0.466876357749763, 0.4694248592972834, 
                    0.473023158283217, 0.4800479773198128, float('inf')],
        'c_f_o_a': [-float('inf'), 0.311593690316192, 0.3162976586263, 0.318773431421094, 
                    0.320577208743015, 0.322487090613284, 0.324538445214685, 0.3271415434674968, 
                    0.33052981537808623, 0.337023413737002, float('inf')],
        'cfo_t_a': [-float('inf'), 0.5300506444742535, 0.5572796604178494, 0.5719639545181976, 
                    0.582457292394041, 0.593266274083544, 0.6040416429440366, 0.6175917808831816, 
                    0.6335248286323122, 0.6585020168824591, float('inf')],
        'c_f_t_e': [-float('inf'), 0.3094124816866718, 0.3121787351231764, 0.3135669617517418, 
                    0.31437364015497105, 0.314952752072916, 0.3157434345953432, 0.3168407701630426, 
                    0.3187618623151664, 0.3227742365828652, float('inf')],
        'c_f_t_l': [-float('inf'), 0.4502340075417116, 0.45568710337828, 0.4580273020729182, 
                    0.45904313701068117, 0.459750137932885, 0.4608706941941624, 0.462795997322695, 
                    0.46630625926473523, float('inf')],
        'a_t_n_p_g_r': [-float('inf'), 0.6888918742926496, 0.6892155963186279, 0.689313185551231, 
                        0.6893817940537679, 0.689438526343149, 0.6894940294617798, 0.6895859460425204, 
                        0.689722972612691, 0.6900140379378645, float('inf')]
    }

def bin_data(data, bins):
    for j in range(len(bins)-1):
        if data >= bins[j] and data < bins[j+1]:
            return j
    return len(bins)-2

def predict_bankruptcy(model_choice, roa_b, Operating_Gross_Margin, Persistent_EPS, 
                       Gross_Profit_to_Sales, Cash_Total_Assets, Debt_Ratio,
                       Net_Worth_Assets, Liability_to_Equity, Cash_Flow_Rate,
                       Cash_Flow_Per_Share, CFO_to_Assets, Cash_Flow_To_Equity,
                       Cash_Flow_To_Liabilities, After_tax_Net_Profit_Growth_Rate,
                       threshold):
    
    if model_choice == "Neural Network":
        if nn_model is None:
            return None, "<div style='padding: 20px; color: red;'>Neural Network model not loaded!</div>"
        
        # Prepare input for NN
        input_data = np.array([[
            roa_b, Operating_Gross_Margin, Persistent_EPS, Gross_Profit_to_Sales,
            Cash_Total_Assets, Debt_Ratio, Net_Worth_Assets, Liability_to_Equity,
            Cash_Flow_Rate, Cash_Flow_Per_Share, CFO_to_Assets, Cash_Flow_To_Equity,
            Cash_Flow_To_Liabilities, After_tax_Net_Profit_Growth_Rate
        ]])
        
        # Scale and predict
        input_scaled = nn_scaler.transform(input_data)
        probability = nn_model.predict(input_scaled, verbose=0)[0][0]
        
        # Determine result
        is_risky = probability > threshold
        
        # Format output
        result_html = f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {"#1A1818" if is_risky else "#0E3149"}; 
                    border-left: 5px solid {"#DC2626" if is_risky else "#10B981"};'>
            <h3>{"⚠️ HIGH BANKRUPTCY RISK" if is_risky else "✅ LOW BANKRUPTCY RISK"}</h3>
            <p><strong>Model:</strong> Neural Network</p>
            <p><strong>Probability:</strong> {probability:.2%}</p>
            <p><strong>Threshold:</strong> {threshold:.0%}</p>
            <p><strong>Recommendation:</strong> {"Immediate financial review required" if is_risky else "Continue regular monitoring"}</p>
        </div>
        """
        
        return probability, result_html
    
    else:  # Random Forest
        if rf_model is None:
            return None, "<div style='padding: 20px; color: red;'>Random Forest model not loaded!</div>"
        
        # Bin the data
        bins = get_bins()
        binned_values = {
            'ROA (B) before interest and depreciation after tax_binned': bin_data(roa_b, bins['roa']),
            'Operating Gross Margin_binned': bin_data(Operating_Gross_Margin, bins['ogm']),
            'Persistent EPS in the Last Four Seasons_binned': bin_data(Persistent_EPS, bins['p_eps']),
            'Gross Profit to Sales_binned': bin_data(Gross_Profit_to_Sales, bins['g_p_t_s']),
            'Cash / Total Assets_binned': bin_data(Cash_Total_Assets, bins['c_t_a']),
            'Debt Ratio %_binned': bin_data(Debt_Ratio, bins['d_r']),
            'Net Worth / Assets_binned': bin_data(Net_Worth_Assets, bins['n_w_a']),
            'Liability to Equity_binned': bin_data(Liability_to_Equity, bins['l_t_e']),
            'Cash Flow Rate_binned': bin_data(Cash_Flow_Rate, bins['c_f_r']),
            'Cash Flow Per Share_binned': bin_data(Cash_Flow_Per_Share, bins['c_f_p_s']),
            'CFO to Assets_binned': bin_data(CFO_to_Assets, bins['c_f_o_a']),
            'Cash Flow to Equity_binned': bin_data(Cash_Flow_To_Equity, bins['cfo_t_a']),
            'Cash Flow to Liabilities_binned': bin_data(Cash_Flow_To_Liabilities, bins['c_f_t_l']),
            'After-tax Net Profit Growth Rate_binned': bin_data(After_tax_Net_Profit_Growth_Rate, bins['a_t_n_p_g_r'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([binned_values])
        
        # Scale and predict
        input_scaled = rf_scaler.transform(input_df)
        prediction = rf_model.predict(input_scaled)[0]
        
        # Get probability if available
        try:
            probabilities = rf_model.predict_proba(input_scaled)[0]
            probability = probabilities[1]  # Probability of bankruptcy
        except:
            probability = 1.0 if prediction == 1 else 0.0
        
        is_risky = prediction == 1
        
        # Format output
        result_html = f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {"#1A1818" if is_risky else "#0E3149"}; 
                    border-left: 5px solid {"#DC2626" if is_risky else "#10B981"};'>
            <h3>{"⚠️ HIGH BANKRUPTCY RISK" if is_risky else "✅ LOW BANKRUPTCY RISK"}</h3>
            <p><strong>Model:</strong> Random Forest (Binned)</p>
            <p><strong>Prediction:</strong> {"Bankrupt" if is_risky else "Not Bankrupt"}</p>
            <p><strong>Probability:</strong> {probability:.2%}</p>
            <p><strong>Recommendation:</strong> {"Immediate financial review required" if is_risky else "Continue regular monitoring"}</p>
        </div>
        """
        
        return probability, result_html

# Create Gradio interface
with gr.Blocks(title="Bankruptcy Risk Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 Bankruptcy Risk Predictor")
    gr.Markdown("Enter company financial ratios to assess bankruptcy risk using different ML models")
    
    with gr.Row():
        model_choice = gr.Radio(
            choices=["Neural Network", "Random Forest"],
            value="Neural Network",
            label="Select Model",
            info="Choose which model to use for prediction"
        )
        threshold = gr.Slider(
            0.05, 0.5, 
            value=0.15, 
            label="Risk Threshold (Neural Network only)", 
            info="Higher = more conservative"
        )
    
    with gr.Row():
        with gr.Column():
            roa_b = gr.Number(value=0.01, label="ROA (B) before interest and depreciation after tax")
            Operating_Gross_Margin = gr.Number(value=0.01, label="Operating Gross Margin")
            Persistent_EPS = gr.Number(value=0.01, label="Persistent EPS in the Last Four Seasons")
            Gross_Profit_to_Sales = gr.Number(value=0.01, label="Gross Profit to Sales")
            Cash_Total_Assets = gr.Number(value=0.001, label="Cash / Total Assets")
            Debt_Ratio = gr.Number(value=0.01, label="Debt Ratio %")
            Net_Worth_Assets = gr.Number(value=0.001, label="Net Worth / Assets")
        
        with gr.Column():
            Liability_to_Equity = gr.Number(value=0.001, label="Liability to Equity")
            Cash_Flow_Rate = gr.Number(value=0.001, label="Cash Flow Rate")
            Cash_Flow_Per_Share = gr.Number(value=0.001, label="Cash Flow Per Share")
            CFO_to_Assets = gr.Number(value=0.001, label="CFO to Assets")
            Cash_Flow_To_Equity = gr.Number(value=0.001, label="Cash Flow to Equity")
            Cash_Flow_To_Liabilities = gr.Number(value=0.001, label="Cash Flow to Liabilities")
            After_tax_Net_Profit_Growth_Rate = gr.Number(value=0.001, label="After-tax Net Profit Growth Rate")
    
    predict_btn = gr.Button("🔍 Predict Bankruptcy Risk", variant="primary", size="lg")
    
    with gr.Row():
        probability_output = gr.Number(label="Bankruptcy Probability")
        html_output = gr.HTML()
    
    predict_btn.click(
        fn=predict_bankruptcy,
        inputs=[
            model_choice, roa_b, Operating_Gross_Margin, Persistent_EPS, Gross_Profit_to_Sales,
            Cash_Total_Assets, Debt_Ratio, Net_Worth_Assets, Liability_to_Equity,
            Cash_Flow_Rate, Cash_Flow_Per_Share, CFO_to_Assets, Cash_Flow_To_Equity,
            Cash_Flow_To_Liabilities, After_tax_Net_Profit_Growth_Rate, threshold
        ],
        outputs=[probability_output, html_output]
    )

if __name__ == "__main__":
    demo.launch()