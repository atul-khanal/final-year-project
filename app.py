from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import supervised
import unsupervised

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload and static directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '<div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">No file uploaded</div>'
    
    file = request.files['file']
    if file.filename == '':
        return '<div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">No selected file</div>'
    
    if not allowed_file(file.filename):
        return '<div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">Invalid file type</div>'
    
    model_type = request.form.get('model_type', 'supervised')
    if model_type not in ['supervised', 'unsupervised']:
        return '<div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">Invalid model type</div>'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Check if file doesn't exist, then save it
    if not os.path.exists(filepath):
        file.save(filepath)
        is_new_upload = True
    else:
        is_new_upload = False
    
    try:
        # Show processing status
        processing_message = f'''
        <div class="bg-yellow-50 p-6 rounded-lg shadow-md animate-pulse">
            <div class="flex items-center">
                <svg class="animate-spin h-5 w-5 mr-3 text-yellow-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <h3 class="text-lg font-semibold text-yellow-700">Processing {filename}...</h3>
            </div>
            <p class="mt-2 text-sm text-yellow-600">Please wait while we analyze your data.</p>
        </div>
        '''
        
        if model_type == 'supervised':
            metrics = supervised.process_data(filepath)
            image_path = 'supervised_confusion_matrix.png'
        else:
            metrics = unsupervised.process_data(filepath)
            image_path = 'unsupervised_confusion_matrix.png'

        # Check if DDoS was detected (high prediction rate of malicious traffic)
        ddos_threshold = 0.7  # You can adjust this threshold
        ddos_alert = ""
        if metrics.get("precision", 0) > ddos_threshold:
            ddos_alert = f'''
            <div class="bg-red-100 border-l-4 border-red-500 p-4 mb-4">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                    </div>
                    <div class="ml-3">
                        <p class="text-red-700 font-bold">DDoS Attack Detected!</p>
                        <p class="text-red-600 text-sm">High probability of malicious traffic detected in the analyzed data.</p>
                    </div>
                </div>
            </div>
            '''
        
        # Return HTML partial with results after processing is complete
        return f'''
        {ddos_alert}
        <div class="bg-white p-6 rounded-lg shadow-md">
            <div class="flex items-center mb-4">
                <svg class="h-6 w-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                <h3 class="text-lg font-semibold">Analysis Complete</h3>
            </div>
            
            <!-- Upload Status -->
            <div class="mb-4">
                <p class="text-sm {'text-green-600' if is_new_upload else 'text-blue-600'}">
                    {f"New file uploaded and processed: {filename}" if is_new_upload else f"Processed existing file: {filename}"}
                </p>
            </div>
            
            <!-- Metrics -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                {f"""
                <div class="bg-blue-50 p-4 rounded-lg">
                    <p class="text-sm text-blue-600 font-medium">Accuracy</p>
                    <p class="text-2xl font-bold">{metrics["accuracy"]:.6f}</p>
                </div>
                """ if "accuracy" in metrics else ""}
                <div class="bg-green-50 p-4 rounded-lg">
                    <p class="text-sm text-green-600 font-medium">Precision</p>
                    <p class="text-2xl font-bold">{metrics["precision"]:.6f}</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <p class="text-sm text-purple-600 font-medium">Recall</p>
                    <p class="text-2xl font-bold">{metrics["recall"]:.6f}</p>
                </div>
                <div class="bg-yellow-50 p-4 rounded-lg">
                    <p class="text-sm text-yellow-600 font-medium">F1 Score</p>
                    <p class="text-2xl font-bold">{metrics["f1"]:.6f}</p>
                </div>
            </div>

            <!-- Confusion Matrix Image -->
            <div class="mt-6">
                <h4 class="text-md font-semibold mb-2">Confusion Matrix</h4>
                <img src="/static/{image_path}" alt="Confusion Matrix" 
                     class="mx-auto max-w-full h-auto rounded-lg shadow-sm">
            </div>
        </div>
        '''
    
    except Exception as e:
        return f'<div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">Error processing file: {str(e)}</div>'
    
    finally:
        # Only remove the file if it was newly uploaded
        if is_new_upload and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True) 