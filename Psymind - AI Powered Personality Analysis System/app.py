from flask import Flask, request, jsonify, send_file, redirect, url_for, flash, make_response, session, g
from models.emotion_detector import classify_emotion
from models.personality_model import predict_ocean_traits
from models.distortion_model import detect_distortions
from models.gpt_utils import generate_summary_with_openrouter
from models.gpt_role_suggestion import get_role_suggestion_openrouter
from utils.cv_parser import extract_text_from_file, extract_info
from utils.db_handler import init_db, save_applicant, get_all_applicants, delete_applicant, export_applicants_csv, delete_all, get_applicant_by_id, DB_PATH
from models.chat_agent import run_agent_response
from models.advisor import get_advisor_chain
from authlib.integrations.flask_client import OAuth
from flask import render_template
from collections import Counter
import secrets
import json
import requests
import traceback
import pandas as pd
import os
import sqlite3


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_dev_key")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":  # 💡 You can replace with DB check
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid login credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("admin_logged_in", None)
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))

@app.route("/")
def home():
    linkedin_data = session.pop('linkedin_data', None)
    
    return render_template('index1.html', linkedin_data=linkedin_data or {})

@app.route("/admin")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("login"))

    df = get_all_applicants()

    # Emotion data
    emotion_counts = Counter(df["emotion"].dropna())

    # Personality - average of Big Five
    personality_scores = {"Openness": 0, "Conscientiousness": 0, "Extraversion": 0, "Agreeableness": 0, "Neuroticism": 0}
    total = 0
    for val in df["personality"].dropna():
        try:
            parsed = json.loads(val.replace("'", "\""))
            for k in personality_scores:
                personality_scores[k] += parsed.get(k, 0)
            total += 1
        except:
            continue
    if total > 0:
        personality_avg = {k: round(v / total, 2) for k, v in personality_scores.items()}
    else:
        personality_avg = {}

    # Distortion counts
    distortions = []
    for val in df["distortion"].dropna():
        try:
            parsed = json.loads(val.replace("'", "\""))
            distortions.extend(parsed if isinstance(parsed, list) else [])
        except:
            continue
    distortion_counts = dict(Counter(distortions))

    return render_template("admin.html",
                           applicants=df.to_dict(orient="records"),
                           emotion_data=emotion_counts,
                           personality_data=personality_avg,
                           distortion_data=distortion_counts)

@app.route("/delete/<int:applicant_id>", methods=["POST"])
def delete(applicant_id):
    delete_applicant(applicant_id)
    flash("Submission deleted successfully.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/delete_all", methods=["POST"])
def delete_all_applicants():
    delete_all()
    flash("All submissions deleted successfully.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/export")
def export_csv():
    csv_data = export_applicants_csv()
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=applicants.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route("/role_suggestions", methods=["GET"])
def role_suggestions_page():
    applicants_df = get_all_applicants()

    # Convert to list of dicts for Jinja compatibility
    applicants = applicants_df.to_dict(orient="records")
    return render_template("role_suggestions.html", applicants=applicants)

@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text input provided"}), 400

    text = data["text"]
    name = data.get("name", "Anonymous")
    email = data.get("email", "Not provided")

    # Analysis
    
    emotions = classify_emotion(text)
    personality = predict_ocean_traits(text)
    distortions = detect_distortions(text)
    summary = generate_summary_with_openrouter(text, personality, emotions, distortions)

    session['analysis_summary'] = summary
    session['analysis_personality'] = personality
    session['analysis_emotion'] = emotions
    session['analysis_distortion'] = distortions
    session['chat_history'] = []
    
    # Save to DB
    save_applicant(name, email, text, str(emotions), str(personality), str(distortions), summary)

    return jsonify({
        "name": name,
        "email": email,
        "personality": personality,
        "emotions": emotions,
        "distortions": distortions,
        "summary": summary
    })


@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    if "cv" not in request.files:
        return jsonify({"error": "No CV file uploaded"}), 400

    uploaded_file = request.files["cv"]
    text_data = extract_text_from_file(uploaded_file)
    info = extract_info(text_data)
    text = info["text"]
    name = info["name"]
    email = info["email"]

    # Analysis
    emotions = classify_emotion(text)
    personality = predict_ocean_traits(text)
    distortions = detect_distortions(text)
    summary = generate_summary_with_openrouter(text, personality, emotions, distortions)

    summary = generate_summary_with_openrouter(text, personality, emotions, distortions)

    session['analysis_summary'] = summary
    session['analysis_personality'] = personality
    session['analysis_emotion'] = emotions
    session['analysis_distortion'] = distortions

    session['chat_history'] = []
    # Save to DB
    save_applicant(name, email, text, str(emotions), str(personality), str(distortions), summary)

    return jsonify({
    "name": name,
    "email": email,
    "personality": personality,
    "emotions": emotions,  
    "distortions": distortions,  
    "summary": summary  
})


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question")

    # Ensure there's a base summary to work from
    base_summary = session.get("analysis_summary")
    if not base_summary:
        return jsonify({"error": "No analysis summary found. Please submit a profile first."}), 400

    # Load or initialize chat history
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Add the new user question to history
    session['chat_history'].append({
        "role": "user",
        "content": user_input
    })
    summary = session.get("analysis_summary", "")
    personality = session.get("analysis_personality", "")
    emotion = session.get("analysis_emotion", "")
    distortion = session.get("analysis_distortion", "")
    # Send to Claude via OpenRouter
    response_text = run_agent_response(
        user_question= user_input,
        chat_history=session['chat_history'],
        summary=summary,
        personality=personality,
        emotion=emotion,
        distortion=distortion
    )

    # Append assistant response to history
    session['chat_history'].append({
        "role": "assistant",
        "content": response_text
    })

    # Return the assistant's reply and history
    return jsonify({"response": response_text, "history": session['chat_history']})

@app.route("/get_role_suggestion/<int:applicant_id>", methods=["POST"])
def get_role_suggestion(applicant_id):
    applicant = get_applicant_by_id(applicant_id)

    if not applicant:
        flash("Applicant not found.", "danger")
        return redirect(url_for("role_suggestions_page"))

    # Extract relevant fields
    personality = applicant["personality"]
    emotion = applicant["emotion"]
    distortion = applicant["distortion"]

    # Generate role suggestion from GPT
    role_suggestion = get_role_suggestion_openrouter(personality, emotion, distortion)

    return render_template("role_result.html", suggestion=role_suggestion, name=applicant["name"])

@app.before_request
def init_advisor():
    if "advisor_chain" not in g:
        g.advisor_chain = get_advisor_chain(api_key="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")

@app.route("/mental-health", methods=["GET", "POST"])
def mental_health_chat():
    response = ""
    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            chain = g.advisor_chain
            response = chain.run(user_input)

    return render_template("mental_health.html", response=response)

@app.route("/ask_advisor", methods=["POST"])
def ask_advisor():
    data = request.get_json()
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"response": "Please enter a message."}), 400

    try:
        chain = g.advisor_chain
        response = chain.run(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500


@app.route("/clear-session", methods=["POST"])
def clear_session():
    session.clear()
    return {"status": "success", "message": "Session cleared"}

oauth = OAuth(app)

linkedin = oauth.register(
    name='linkedin',
    client_id="78ecktgbifudqx",  # Hardcoded (replace with your actual ID)
    client_secret="WPL_AP1.6ITFP4EU4VV40o6O.sqnYgQ==",  # Hardcoded (replace with actual secret)
    access_token_url='https://www.linkedin.com/oauth/v2/accessToken',
    authorize_url='https://www.linkedin.com/oauth/v2/authorization',
    api_base_url='https://api.linkedin.com/v2/',
    jwks_uri='https://www.linkedin.com/oauth/openid/jwks',
    client_kwargs={
        'scope': 'openid profile email',
        'token_endpoint_auth_method': 'client_secret_post',
        'prompt': 'select_account',
        'metadata': {
            'issuer': 'https://www.linkedin.com',
            'authorization_endpoint': 'https://www.linkedin.com/oauth/v2/authorization',
            'token_endpoint': 'https://www.linkedin.com/oauth/v2/accessToken',
            'userinfo_endpoint': 'https://api.linkedin.com/v2/userinfo',
            'jwks_uri': 'https://www.linkedin.com/oauth/openid/jwks',
            'response_types_supported': ['code']
        }
    }
)

@app.route('/login/linkedin')
def login_linkedin():
    """Initiate LinkedIn OAuth flow"""
    try:
        # Generate a secure random state parameter
        state = secrets.token_hex(16)
        session['oauth_state'] = state
        
        print(f"Generated state: {state}")
        
        # Create the authorization URL with proper parameters
        redirect_uri = url_for('authorize_linkedin', _external=True)
        
        # Use the OAuth client to create authorization URL
        auth_response = linkedin.create_authorization_url(
            redirect_uri=redirect_uri,
            state=state,
        )
        
        # Extract the URL from the response (it returns a dict with 'url' key)
        if isinstance(auth_response, dict):
            authorization_url = auth_response.get('url')
            print(f"Authorization response: {auth_response}")
        else:
            authorization_url = auth_response
        
        print(f"Authorization URL: {authorization_url}")
        print(f"Redirect URI: {redirect_uri}")
        
        if not authorization_url:
            raise Exception("No authorization URL generated")
        
        return redirect(authorization_url)
        
    except Exception as e:
        print(f"Error in login_linkedin: {str(e)}")
        traceback.print_exc()
        return redirect(url_for('home') + '?error=login_failed')


@app.route('/authorize/linkedin')
def authorize_linkedin():
    """Handle LinkedIn OAuth callback"""
    try:
        print("=== LinkedIn OAuth Callback Debug ===")
        print(f"Full callback URL: {request.url}")
        print(f"Request args: {dict(request.args)}")
        
        # Check for error parameter first
        error = request.args.get('error')
        if error:
            error_description = request.args.get('error_description', 'No description')
            print(f"OAuth error received: {error} - {error_description}")
            return redirect(url_for('home') + f'?error=oauth_{error}')
        
        # Retrieve the stored state parameter
        expected_state = session.pop('oauth_state', None)
        received_state = request.args.get('state')
        
        print(f"Expected state: {expected_state}")
        print(f"Received state: {received_state}")
        
        if not expected_state:
            print("No state found in session")
            return redirect(url_for('home') + '?error=no_state')
            
        # Validate state parameter for CSRF protection
        if expected_state != received_state:
            print("State validation failed - possible CSRF attack")
            return redirect(url_for('home') + '?error=invalid_state')
        
        print("State validation passed!")
        
        # Get the authorization code
        code = request.args.get('code')
        if not code:
            print("No authorization code received")
            return redirect(url_for('home') + '?error=no_code')
        
        print(f"Authorization code received: {code[:20]}...")
        
        # FIXED: Manual token exchange with proper parameters
        try:
            print("Attempting manual token exchange...")
            
            # Construct the exact redirect URI used in the initial request
            redirect_uri = url_for('authorize_linkedin', _external=True)
            
            token_data = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': redirect_uri,
                'client_id': linkedin.client_id,
                'client_secret': linkedin.client_secret,
            }
            
            print(f"Token request data: {token_data}")
            print(f"Redirect URI being used: {redirect_uri}")
            
            # Make the token request
            response = requests.post(
                'https://www.linkedin.com/oauth/v2/accessToken',
                data=token_data,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
            )
            
            print(f"Token response status: {response.status_code}")
            print(f"Token response: {response.text}")
            
            if response.status_code == 200:
                token_info = response.json()
                access_token = token_info.get('access_token')
                
                if not access_token:
                    print("No access token in response")
                    return redirect(url_for('home') + '?error=no_access_token')
                
                print("✓ Token exchange succeeded!")
                print(f"Access token received: {access_token[:20]}...")
                
            else:
                print(f"Token exchange failed: HTTP {response.status_code}")
                print(f"Error response: {response.text}")
                
                # Parse the error response
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error_description', error_data.get('error', 'Unknown error'))
                    print(f"LinkedIn error: {error_msg}")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return redirect(url_for('home') + f'?error=token_failed&detail={error_msg}')
                
        except Exception as e:
            print(f"Exception during token exchange: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return redirect(url_for('home') + '?error=token_exception')
        
        # Get user profile information
        try:
            print("Fetching user profile...")
            
            # Use LinkedIn's userinfo endpoint (OpenID Connect)
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            # Try the OpenID Connect userinfo endpoint first
            profile_response = requests.get(
                'https://api.linkedin.com/v2/userinfo',
                headers=headers
            )
            
            print(f"Profile response status: {profile_response.status_code}")
            print(f"Profile response: {profile_response.text}")
            
            if profile_response.status_code == 200:
                profile = profile_response.json()
                print("Profile data received successfully")
                
            else:
                print(f"Failed to get profile: {profile_response.status_code}")
                print(f"Trying alternative endpoint...")
                
                # Try the v2/people endpoint as fallback
                profile_response = requests.get(
                    'https://api.linkedin.com/v2/people/~',
                    headers=headers
                )
                
                if profile_response.status_code == 200:
                    profile = profile_response.json()
                    print("Profile data received from people endpoint")
                else:
                    print(f"All profile endpoints failed: {profile_response.status_code}")
                    return redirect(url_for('home') + '?error=profile_failed')
            
        except Exception as e:
            print(f"Error getting profile: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return redirect(url_for('home') + '?error=profile_exception')
        
        # Extract user information
        try:
            # Handle different response formats
            if 'given_name' in profile and 'family_name' in profile:
                # OpenID Connect format
                name = f"{profile.get('given_name', '')} {profile.get('family_name', '')}".strip()
                email = profile.get('email', 'No email available')
            elif 'localizedFirstName' in profile and 'localizedLastName' in profile:
                # LinkedIn v2 API format
                name = f"{profile.get('localizedFirstName', '')} {profile.get('localizedLastName', '')}".strip()
                email = 'No email available'  # Need separate email API call
            else:
                # Fallback
                name = profile.get('name', 'LinkedIn User')
                email = profile.get('email', 'No email available')
            
            if not name or name.strip() == '':
                name = 'LinkedIn User'
            
            print(f"Extracted name: {name}")
            print(f"Extracted email: {email}")
            
        except Exception as e:
            print(f"Error extracting profile data: {str(e)}")
            name = 'LinkedIn User'
            email = 'No email available'
        
        # If email is not available, try to get it separately
        if email == 'No email available':
            try:
                print("Attempting to get email separately...")
                email_response = requests.get(
                    'https://api.linkedin.com/v2/emailAddress?q=members&projection=(elements*(handle~))',
                    headers=headers
                )
                
                if email_response.status_code == 200:
                    email_data = email_response.json()
                    elements = email_data.get('elements', [])
                    if elements and len(elements) > 0:
                        email_handle = elements[0].get('handle~', {})
                        email = email_handle.get('emailAddress', 'No email available')
                        print(f"Email retrieved: {email}")
                
            except Exception as e:
                print(f"Could not retrieve email: {str(e)}")
        
        # Prepare text data for analysis
        headline = ""  # You can get this with additional API calls if needed
        text_data = f"{name} - {headline}" if headline else name
        
        # Only proceed with analysis if we have meaningful data
        if len(text_data.strip()) < 2:
            text_data = f"LinkedIn user: {name}"
        
        print(f"Analyzing text data: {text_data}")
        
        # Run your analysis models (make sure these functions exist)
        emotion = classify_emotion(text_data)
        personality = predict_ocean_traits(text_data)
        distortion = detect_distortions(text_data)
        
        # Generate summary using your AI service
        summary = generate_summary_with_openrouter(
                text=text_data,  # First parameter: text
                personality=personality,  # Second parameter: personality
                emotions=emotion,  # Third parameter: emotions (note: renamed from 'emotion' to 'emotions')
                distortions=distortion  # Fourth parameter: distortions (note: renamed from 'distortion' to 'distortions')
            )
        try:
            emotion_json = json.dumps(emotion) if emotion else "{}"
            personality_json = json.dumps(personality) if personality else "{}"
            distortion_json = json.dumps(distortion) if distortion else "{}"
            
            print(f"Serialized data types:")
            print(f"  emotion_json: {type(emotion_json)} - {emotion_json}")
            print(f"  personality_json: {type(personality_json)} - {personality_json}")
            print(f"  distortion_json: {type(distortion_json)} - {distortion_json}")
            
        except Exception as e:
            print(f"Error serializing data: {e}")
            emotion_json = "{}"
            personality_json = "{}"
            distortion_json = "{}"
        # Save to database
        save_applicant(name, email, text_data, emotion_json, personality_json, distortion_json, summary)
        
        print(f"Successfully processed LinkedIn auth for user: {name}")
        
        return render_template("dashboard.html", 
                             name=name, 
                             email=email,
                             emotion=emotion, 
                             personality=personality,
                             distortion=distortion, 
                             summary=summary)
        
    except Exception as e:
        print(f"CRITICAL ERROR during LinkedIn auth: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return redirect(url_for('home') + '?error=authentication_failed')


# Additional helper route for debugging OAuth configuration
@app.route('/debug/oauth')
def debug_oauth():
    """Debug route to check OAuth configuration"""
    try:
        config_info = {
            'client_id': linkedin.client_id,
            'client_secret': linkedin.client_secret[:10] + '...' if linkedin.client_secret else None,
            'redirect_uri': url_for('authorize_linkedin', _external=True),
            'server_metadata_url': getattr(linkedin, 'server_metadata_url', 'Not set'),
            'client_kwargs': getattr(linkedin, 'client_kwargs', {}),
        }
        
        return f"<pre>LinkedIn OAuth Configuration:\n{json.dumps(config_info, indent=2)}</pre>"
    except Exception as e:
        return f"Error getting OAuth config: {str(e)}"


# Also add an error handling route to show detailed error messages
@app.route('/oauth/error')
def oauth_error():
    """Display OAuth error details"""
    error = request.args.get('error', 'Unknown error')
    detail = request.args.get('detail', 'No additional details')
    
    return f"""
    <h1>OAuth Authentication Error</h1>
    <p><strong>Error:</strong> {error}</p>
    <p><strong>Details:</strong> {detail}</p>
    <p><a href="{url_for('home')}">Return to Home</a></p>
    """

    

if __name__ == "__main__":
    init_db()
    app.run(debug=True)




