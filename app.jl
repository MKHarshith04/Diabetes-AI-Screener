# app.jl - The Clinical Web Interface
using Genie
import Genie.Router: route
import Genie.Renderer.Html: html
import Genie.Requests: postpayload
using MLJ, DataFrames, MLJDecisionTreeInterface, DecisionTree
using Base64  # <--- Make sure this line is here!
import MLJ: predict, pdf

println("--- Starting Server Boot Sequence ---")

# 1. LOAD THE AI'S BRAIN
const model_url = "https://github.com/MKHarshith04/Diabetes-AI-Screener/releases/download/v1.0/diabetes_rf_model.jls"
const model_path = joinpath(@__DIR__, "diabetes_rf_model.jls")

# If the cloud server doesn't have the file, download it directly from your GitHub Release!
if !isfile(model_path)
    println("☁️ Downloading heavy AI model from GitHub Releases...")
    download(model_url, model_path)
end

println("Loading AI Model from: ", model_path)
const mach_forest = machine(model_path)
println("--- Model Loaded Successfully! ---")

# 2. DEFINE THE HTML FORM (We removed the hardcoded 'value' and 'selected' tags)
const form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes AI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; margin: 0; padding: 40px 0; }
        .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 400px; }
        h2 { color: #d35400; text-align: center; margin-bottom: 20px;}
        label { font-weight: bold; margin-top: 10px; display: block; color: #333;}
        input, select { width: 100%; padding: 10px; margin-top: 5px; margin-bottom: 15px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
        button { width: 100%; background-color: #d35400; color: white; padding: 12px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; font-weight: bold; transition: background 0.3s;}
        button:hover { background-color: #e67e22; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px; }
        .high-risk { background-color: #fadbd8; color: #c0392b; border: 1px solid #e74c3c; }
        .low-risk { background-color: #d4efdf; color: #27ae60; border: 1px solid #2ecc71; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🩺 Clinical AI Screener</h2>
        <form action="/predict" method="POST">
            
            <label>Patient BMI</label>
            <input type="number" step="0.1" name="bmi" required value="{{BMI_VAL}}">

            <label>Age Category</label>
            <select name="age">
                <option value="1" {{AGE_1}}>18 to 24</option>
                <option value="2" {{AGE_2}}>25 to 29</option>
                <option value="3" {{AGE_3}}>30 to 34</option>
                <option value="4" {{AGE_4}}>35 to 39</option>
                <option value="5" {{AGE_5}}>40 to 44</option>
                <option value="6" {{AGE_6}}>45 to 49</option>
                <option value="7" {{AGE_7}}>50 to 54</option>
                <option value="8" {{AGE_8}}>55 to 59</option>
                <option value="9" {{AGE_9}}>60 to 64</option>
                <option value="10" {{AGE_10}}>65 to 69</option>
                <option value="11" {{AGE_11}}>70 to 74</option>
                <option value="12" {{AGE_12}}>75 to 79</option>
                <option value="13" {{AGE_13}}>80 or older</option>
            </select>

            <label>High Blood Pressure?</label>
            <select name="highbp">
                <option value="0.0" {{BP_0}}>No</option>
                <option value="1.0" {{BP_1}}>Yes</option>
            </select>

            <label>General Health (1:Excellent - 5:Poor)</label>
            <select name="genhlth">
                <option value="1.0" {{GH_1}}>1 - Excellent</option>
                <option value="2.0" {{GH_2}}>2 - Very Good</option>
                <option value="3.0" {{GH_3}}>3 - Good</option>
                <option value="4.0" {{GH_4}}>4 - Fair</option>
                <option value="5.0" {{GH_5}}>5 - Poor</option>
            </select>

            <button type="submit">Run AI Prediction</button>
        </form>
    </div>
</body>
</html>
"""

# 3. ROUTE 1: The Homepage (Injects default values for the first visit)
route("/") do
    page = replace(form_html, 
        "{{BMI_VAL}}" => "25.0",
        "{{AGE_9}}" => "selected",
        "{{BP_0}}" => "selected",
        "{{GH_3}}" => "selected",
        r"{{[A-Z0-9_]+}}" => "" # Clears out the other placeholders
    )
    html(page)
end

# 4. ROUTE 2: The Prediction Engine (Injects the USER'S values back into the form)
route("/predict", method = POST) do
    # 4a. Grab the user inputs from the web form as strings first (for the HTML replacement)
    raw_bmi = postpayload(:bmi)
    raw_age = postpayload(:age)
    raw_bp = postpayload(:highbp)
    raw_gh = postpayload(:genhlth)

    # Convert to numbers for the AI
    user_bmi = parse(Float64, raw_bmi)
    user_age = parse(Float64, raw_age)
    user_highbp = parse(Float64, raw_bp)
    user_genhlth = parse(Float64, raw_gh)

    # 4b. Create the DataFrame
    patient_data = DataFrame(
        HighBP = [user_highbp], HighChol = [0.0], CholCheck = [1.0], BMI = [user_bmi],
        Smoker = [0.0], Stroke = [0.0], HeartDiseaseorAttack = [0.0], PhysActivity = [1.0],
        Fruits = [1.0], Veggies = [1.0], HvyAlcoholConsump = [0.0], AnyHealthcare = [1.0],
        NoDocbcCost = [0.0], GenHlth = [user_genhlth], MentHlth = [0.0], PhysHlth = [0.0],
        DiffWalk = [0.0], Sex = [0.0], Age = [user_age], Education = [6.0], Income = [8.0]
    )

    # 4c. Run the AI Prediction
    y_probs = predict(mach_forest, patient_data)
    unhealthy_prob = pdf(y_probs[1], levels(y_probs[1])[2]) 
    is_diabetic = unhealthy_prob >= 0.30
    prob_percent = round(unhealthy_prob * 100, digits=1)

    # 4d. Generate Result HTML with Explainable AI (XAI)
    if is_diabetic
        # Load the feature importance image and convert it to text for the web
        img_path = joinpath(@__DIR__, "Feature_Importance.png")
        if isfile(img_path)
            img_b64 = base64encode(read(img_path))
            xai_dashboard = """
                <hr style="border: 0; border-top: 1px solid #e74c3c; margin: 20px 0;">
                <p style="color: #c0392b; font-size: 15px; margin-bottom: 5px;"><b>🔍 AI Decision Logic</b></p>
                <p style="color: #333; font-size: 13px; font-weight: normal; margin-top: 0;">Top clinical factors this AI model uses to evaluate risk:</p>
                <img src="data:image/png;base64,$img_b64" style="width: 100%; border-radius: 5px; background: white; padding: 5px; box-sizing: border-box;">
            """
        else
            xai_dashboard = "<br><small>(Feature Importance graph not found)</small>"
        end

        result_box = "<div class='result high-risk'>⚠️ HIGH RISK<br>AI Confidence: $prob_percent% $xai_dashboard</div>"
    else
        result_box = "<div class='result low-risk'>✅ LOW RISK<br>AI Confidence: $prob_percent%</div>"
    end

    # 4e. Re-inject the user's values back into the form so it "remembers" them
    age_tag = "{{AGE_$(Int(user_age))}}"
    bp_tag = "{{BP_$(Int(user_highbp))}}"
    gh_tag = "{{GH_$(Int(user_genhlth))}}"

    page_with_result = replace(form_html, 
        "{{BMI_VAL}}" => raw_bmi,
        age_tag => "selected",
        bp_tag => "selected",
        gh_tag => "selected",
        r"{{[A-Z0-9_]+}}" => "" # Clears out any unused placeholders
    )
    
    # Finally, add the result box to the bottom
    final_page = replace(page_with_result, "</form>" => "</form>" * result_box)

    html(final_page)
end

# 5. Start the server
Genie.up(8000, async=false)