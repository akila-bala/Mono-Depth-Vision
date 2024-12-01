from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'original_data\\test_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'your_secret_key'  # Required for flash messages

# Dummy user data for login validation
users = {'user': 'password'}

# Helper function to check allowed file extensions
def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# Decorator to enforce login on all routes except login itself
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash("Please log in to access this page.", 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
    return render_template('login.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Thank you for your message!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/upload_and_visualize', methods=['POST'])
@login_required
def upload_and_visualize():
    # Check if the post request has the files
    if 'image' not in request.files or 'csvfile' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    image_file = request.files['image']
    csv_file = request.files['csvfile']

    if image_file and allowed_file(image_file.filename, app.config['ALLOWED_EXTENSIONS']) and csv_file:
        image_filename = secure_filename(image_file.filename)
        csv_filename = secure_filename(csv_file.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        image_file.save(image_path)
        csv_file.save(csv_path)

        return redirect(url_for('visualize', image_filename=image_filename, csv_filename=csv_filename))
    else:
        flash('Invalid file type.', 'danger')
        return redirect(request.url)

# Visualization route
@app.route('/visualize', methods=['GET'])
def visualize():
    if login_required():
        return login_required()

    image_filename = request.args.get('image_filename')
    csv_filename = request.args.get('csv_filename')

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

    # Load the image
    background_image = Image.open(image_path)
    background_image_np = np.array(background_image)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Image dimensions
    img_height, img_width, _ = background_image_np.shape

    # Create figure
    fig = go.Figure()

    # Create surface for background image
    x_plane = np.linspace(0, img_width, img_width)
    y_plane = np.linspace(0, img_height, img_height)
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    z_plane = np.zeros_like(x_plane)

    # Add background image as surface
    fig.add_trace(go.Surface(
        x=x_plane,
        y=y_plane,
        z=z_plane,
        surfacecolor=background_image_np[:, :, 0],  # Use red channel of the image for surface color
        showscale=False,
        opacity=0.9
    ))

    # Plot the 3D points on top of the image
    for idx, row in df.iterrows():
        if idx >= 50:
            break

        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        z_actual = row['zloc']
        z_pred = row['zloc_pred']

        # Calculate the center of the bounding box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Add actual Z-location (green)
        fig.add_trace(go.Scatter3d(
            x=[x_center],
            y=[y_center],
            z=[z_actual],
            mode='markers',
            marker=dict(size=5, color='green'),
        ))

        # Add predicted Z-location (red)
        fig.add_trace(go.Scatter3d(
            x=[x_center],
            y=[y_center],
            z=[z_pred],
            mode='markers',
            marker=dict(size=5, color='red'),
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
            zaxis=dict(title="Z (Depth)"),
            aspectratio=dict(x=img_width/img_height, y=1, z=1)
        )
    )

    # Convert the figure to HTML
    graph_html = fig.to_html(full_html=False)

    return render_template('visualize.html', plot_html=graph_html)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
