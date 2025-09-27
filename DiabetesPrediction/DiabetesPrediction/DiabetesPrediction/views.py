from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.utils import timezone
from django.conf import settings
from django.http import HttpResponse, JsonResponse
import os
import pickle
from datetime import datetime
from io import BytesIO
from .performance_utils import get_performance_data, generate_roc_data

# Optional: high-quality PDF generation with reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

try:
    import numpy as np
except Exception:
    np = None
try:
    import joblib
except Exception:
    joblib = None


# Cache for loaded artifacts per algorithm key
_ARTIFACT_CACHE = {}


def _first_existing(paths, names):
    for p in paths:
        for n in names:
            candidate = os.path.join(p, n)
            if os.path.exists(candidate):
                return candidate
    return None


def _load_artifacts_for_algo(algo_hint: str):
    """Load model/scaler based on selected algorithm with simple caching.

    algo_hint can be one of: 'svm', 'logreg', 'rf', 'knn', or None/'svm' default.
    Returns (model, scaler, error, meta_dict)
    """
    key = (algo_hint or 'svm').lower()
    if key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    base_dir = settings.BASE_DIR
    project_root = os.path.dirname(base_dir)
    search_dirs = [
        os.path.join(base_dir, 'prediction'),
        os.path.join(base_dir, 'prediction', 'ml_models'),
        os.path.join(project_root, 'prediction'),
        os.path.join(project_root, 'prediction', 'ml_models'),
        project_root,
        base_dir,
    ]

    generic_model_names = [
        'model.pkl', 'diabetes_model.pkl', 'model.joblib', 'diabetes_model.joblib', 'model.sav', 'diabetes_model.sav'
    ]

    algo_to_candidates = {
        'svm': ['svm.pkl', 'svc.pkl', 'svm.joblib', 'svc.joblib', 'diabetes_svm.pkl', 'diabetes_model_svm.pkl'],
        'logreg': ['logreg.pkl', 'logistic.pkl', 'logistic_regression.pkl', 'logreg.joblib', 'diabetes_logreg.pkl'],
        'rf': ['rf.pkl', 'random_forest.pkl', 'randomforest.pkl', 'rf.joblib', 'diabetes_rf.pkl'],
        'knn': ['knn.pkl', 'knn.joblib', 'diabetes_knn.pkl'],
        'svm_default': [],
    }

    scaler_generic = ['scaler.pkl', 'scaler.joblib', 'standard_scaler.pkl', 'standardscaler.pkl']
    scaler_algo = {
        'svm': ['scaler_svm.pkl', 'svm_scaler.pkl'],
        'logreg': ['scaler_logreg.pkl', 'logreg_scaler.pkl'],
        'rf': ['scaler_rf.pkl', 'rf_scaler.pkl'],  # RF usually doesn't need scaling but we'll support it
        'knn': ['scaler_knn.pkl', 'knn_scaler.pkl'],
        'svm_default': [],
    }

    prioritized_models = algo_to_candidates.get(key, []) + generic_model_names
    prioritized_scalers = scaler_algo.get(key, []) + scaler_generic

    model_path = _first_existing(search_dirs, prioritized_models)
    scaler_path = _first_existing(search_dirs, prioritized_scalers)

    model = None
    scaler = None
    error = None
    try:
        if not model_path:
            raise FileNotFoundError('No model file found. Place your model under the prediction/ or prediction/ml_models/ folder.')
        if joblib is not None:
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        if scaler_path:
            if joblib is not None:
                scaler = joblib.load(scaler_path)
            else:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
        # If model is a Pipeline, ignore external scaler to avoid double-scaling
        try:
            if hasattr(model, 'named_steps') or model.__class__.__name__.lower() == 'pipeline':
                scaler = None
        except Exception:
            pass
    except Exception as exc:
        error = str(exc)

    meta = {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'resolved_algo': key,
    }
    _ARTIFACT_CACHE[key] = (model, scaler, error, meta)
    return _ARTIFACT_CACHE[key]


def index(request):
    # Backward-compat: keep old name but behave as a public landing page
    if request.user.is_authenticated:
        return redirect('home')
    return render(request, 'landing.html')


@login_required(login_url='login')
def home(request):
    # Track home visits in session
    home_visits = request.session.get('home_visits', 0) + 1
    request.session['home_visits'] = home_visits

    # Dynamic greeting
    local_now = timezone.localtime()
    hour = local_now.hour
    if hour < 12:
        greeting = 'Good morning'
    elif hour < 17:
        greeting = 'Good afternoon'
    else:
        greeting = 'Good evening'

    # Membership info
    date_joined = request.user.date_joined
    days_member = (timezone.now() - date_joined).days

    # Rotating health tip of the day
    tips = [
        'Stay hydrated and aim for 6–8 glasses of water daily.',
        'Choose whole grains over refined carbs to stabilize blood sugar.',
        'Add a 20-minute walk after meals to improve glucose control.',
        'Include lean proteins and healthy fats to keep you full longer.',
        'Prioritize consistent sleep: 7–8 hours helps metabolic health.',
        'Track your numbers regularly to spot patterns early.',
    ]
    tip_of_day = tips[local_now.toordinal() % len(tips)]

    context = {
        'greeting': greeting,
        'tip_of_day': tip_of_day,
        'last_login': request.user.last_login,
        'date_joined': date_joined,
        'days_member': days_member,
        'home_visits': home_visits,
        'predict_visits': request.session.get('predict_visits', 0),
        'about_visits': request.session.get('about_visits', 0),
    }
    return render(request, 'home.html', context)


@login_required(login_url='login')
def predict(request):
    # Track predict page visits
    predict_visits = request.session.get('predict_visits', 0) + 1
    request.session['predict_visits'] = predict_visits
    return render(request, 'predict.html', { 'selected_algo': request.GET.get('algo', '') })


@login_required(login_url='login')
def result(request):
    context = {}
    selected_algo = request.GET.get('algo', 'svm')
    model, scaler, load_error, meta = _load_artifacts_for_algo(selected_algo)
    if model is None:
        context['result2'] = f"Model not available: {load_error or 'unknown error'}."
        context['selected_algo'] = selected_algo
        return render(request, 'predict.html', context)

    # Parse inputs with validation
    try:
        raw_features = [
            float(request.GET.get('n1', 0)),  # Pregnancies
            float(request.GET.get('n2', 0)),  # Glucose
            float(request.GET.get('n3', 0)),  # Blood Pressure
            float(request.GET.get('n4', 0)),  # Skin Thickness
            float(request.GET.get('n5', 0)),  # Insulin
            float(request.GET.get('n6', 0)),  # BMI
            float(request.GET.get('n7', 0)),  # Diabetes Pedigree
            float(request.GET.get('n8', 0)),  # Age
        ]
        
        # Validate ranges for better predictions
        if raw_features[1] < 50 or raw_features[1] > 300:  # Glucose
            context['result2'] = 'Glucose level should be between 50-300 mg/dL for accurate prediction.'
            context['selected_algo'] = selected_algo
            return render(request, 'predict.html', context)
        if raw_features[5] < 10 or raw_features[5] > 60:  # BMI
            context['result2'] = 'BMI should be between 10-60 for accurate prediction.'
            context['selected_algo'] = selected_algo
            return render(request, 'predict.html', context)
        if raw_features[7] < 1 or raw_features[7] > 120:  # Age
            context['result2'] = 'Age should be between 1-120 years for accurate prediction.'
            context['selected_algo'] = selected_algo
            return render(request, 'predict.html', context)
            
    except (ValueError, TypeError):
        context['result2'] = 'Invalid input. Please fill all fields with valid numbers.'
        context['selected_algo'] = selected_algo
        return render(request, 'predict.html', context)

    if np is None:
        context['result2'] = 'NumPy not installed. Please install dependencies.'
        context['selected_algo'] = selected_algo
        return render(request, 'predict.html', context)

    features = np.array([raw_features], dtype=float)
    try:
        if scaler is not None:
            features = scaler.transform(features)
        proba_text = ''
        threshold = 0.20  # Lower threshold to catch more diabetes cases
        if hasattr(model, 'predict_proba'):
            try:
                prob = float(model.predict_proba(features)[0][1])
                proba_text = f" (probability: {prob:.2%})"
                label = 1 if prob >= threshold else 0
            except Exception:
                # Fallback to predict if predict_proba fails
                y_pred = model.predict(features)
                label = int(y_pred[0]) if hasattr(y_pred, '__iter__') else int(y_pred)
        else:
            y_pred = model.predict(features)
            label = int(y_pred[0]) if hasattr(y_pred, '__iter__') else int(y_pred)
        # Enhanced result messages with risk levels and hospital recommendations
        if label == 1:
            if hasattr(model, 'predict_proba') and prob > 0.7:
                context['result2'] = f"HIGH RISK for diabetes{proba_text}. Immediate medical consultation recommended."
                context['show_hospitals'] = True
                context['risk_level'] = 'high'
            elif hasattr(model, 'predict_proba') and prob > 0.4:
                context['result2'] = f"MODERATE RISK for diabetes{proba_text}. Schedule a check-up with your doctor."
                context['show_hospitals'] = True
                context['risk_level'] = 'moderate'
            else:
                context['result2'] = f"MILD RISK for diabetes{proba_text}. Consider lifestyle changes and monitor regularly."
                context['show_hospitals'] = True
                context['risk_level'] = 'mild'
        else:
            context['result2'] = f"LOW RISK for diabetes{proba_text}. Continue maintaining a healthy lifestyle."
            context['show_hospitals'] = False
            context['risk_level'] = 'low'
        # Provide debug info only when explicitly requested and in DEBUG mode
        show_debug = getattr(settings, 'DEBUG', False) and str(request.GET.get('debug', '')).lower() in ('1', 'true', 'yes')
        if show_debug:
            context['debug_info'] = {
                'model_path': meta.get('model_path'),
                'scaler_path': meta.get('scaler_path'),
                'inputs_order': [
                    'Pregnancies','Glucose','BloodPressure','SkinThickness',
                    'Insulin','BMI','DiabetesPedigreeFunction','Age'
                ],
                'raw_features': raw_features,
                'scaled_features_preview': features.tolist(),
                'scaler_applied': scaler is not None,
                'threshold': threshold,
                'selected_algo': selected_algo,
            }
    except Exception as exc:
        context['result2'] = f"Prediction failed: {exc}"

    context['selected_algo'] = selected_algo
    return render(request, 'predict.html', context)


@login_required(login_url='login')
def about(request):
    # Track about page visits
    about_visits = request.session.get('about_visits', 0) + 1
    request.session['about_visits'] = about_visits
    return render(request, 'about.html')


@login_required(login_url='login')
def performance(request):
    """Standalone Performance Analysis page"""
    # Reuse same page; charts are rendered client-side
    return render(request, 'performance.html')

@login_required(login_url='login')
def performance_analysis_api(request):
    """API endpoint to get performance analysis data"""
    try:
        performance_data = get_performance_data()
        roc_data = generate_roc_data()
        
        return JsonResponse({
            'success': True,
            'performance_data': performance_data,
            'roc_data': roc_data
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required(login_url='login')
def nearby_hospitals_api(request):
    """API endpoint to get nearby hospitals based on user location or city"""
    try:
        # Sample hospitals data - in production, integrate with Google Places API or similar
        hospitals = [
            {
                'name': 'Apollo Hospital',
                'address': '154/11, Opp. IIM-B, Bannerghatta Road, Bangalore',
                'phone': '+91-80-2692-2222',
                'specialties': ['Endocrinology', 'Diabetes Care', 'Internal Medicine'],
                'rating': 4.3,
                'distance': '2.5 km',
                'emergency': True
            },
            {
                'name': 'Fortis Hospital',
                'address': '14, Cunningham Road, Bangalore',
                'phone': '+91-80-6621-4444',
                'specialties': ['Diabetology', 'Cardiology', 'General Medicine'],
                'rating': 4.1,
                'distance': '3.2 km',
                'emergency': True
            },
            {
                'name': 'Manipal Hospital',
                'address': '98, Rustom Bagh, Airport Road, Bangalore',
                'phone': '+91-80-2502-4444',
                'specialties': ['Diabetes & Endocrinology', 'Nephrology'],
                'rating': 4.2,
                'distance': '4.1 km',
                'emergency': True
            },
            {
                'name': 'Narayana Health City',
                'address': '258/A, Bommasandra Industrial Area, Bangalore',
                'phone': '+91-80-7122-2222',
                'specialties': ['Comprehensive Diabetes Care', 'Preventive Medicine'],
                'rating': 4.0,
                'distance': '5.8 km',
                'emergency': True
            },
            {
                'name': 'BGS Gleneagles Global Hospital',
                'address': '67, Uttarahalli Road, Kengeri, Bangalore',
                'phone': '+91-80-4969-9999',
                'specialties': ['Endocrinology', 'Diabetes Management'],
                'rating': 3.9,
                'distance': '6.2 km',
                'emergency': False
            }
        ]
        
        return JsonResponse({
            'success': True,
            'hospitals': hospitals
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required(login_url='login')
def download_result_pdf(request):
    """Generate a very simple PDF containing the last prediction result.

    For simplicity, reads values from query parameters again (same as on the result page).
    In production, you may want to persist the last result in session and read from there.
    """
    # Build a plain-text report; we will wrap it in a minimal PDF
    user = request.user.username
    algo = request.GET.get('algo', 'svm')
    result_text = request.GET.get('result_text', 'No result')
    input_labels = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    input_values = [
        request.GET.get('n1'), request.GET.get('n2'), request.GET.get('n3'), request.GET.get('n4'),
        request.GET.get('n5'), request.GET.get('n6'), request.GET.get('n7'), request.GET.get('n8'),
    ]

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    if _REPORTLAB_AVAILABLE:
        # Generate robust PDF with reportlab
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, title="Diabetes Prediction Report")
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Diabetes Prediction Report", styles['Title']))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Date: {now_str}", styles['Normal']))
        story.append(Paragraph(f"User: {user}", styles['Normal']))
        story.append(Paragraph(f"Algorithm: {algo.upper()}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Inputs", styles['Heading3']))

        table_data = [["Feature", "Value"]] + [[lbl, str(val)] for lbl, val in zip(input_labels, input_values)]
        table = Table(table_data, hAlign='LEFT', colWidths=[220, 220])
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Result: {result_text}", styles['Heading3']))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()
        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="diabetes_prediction.pdf"'
        return response
    else:
        # Fallback: ask to install reportlab and provide the data as plain text
        hint_lines = [
            "Install PDF dependency first: pip install reportlab",
            "",
            "Diabetes Prediction Report",
            f"Date: {now_str}",
            f"User: {user}",
            f"Algorithm: {algo.upper()}",
            "",
            "Inputs:",
        ]
        for lbl, val in zip(input_labels, input_values):
            hint_lines.append(f"  - {lbl}: {val}")
        hint_lines += ["", f"Result: {result_text}"]
        content = "\n".join(hint_lines)
        return HttpResponse(content, content_type='text/plain; charset=utf-8')
