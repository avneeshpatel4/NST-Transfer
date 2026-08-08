import os
import uuid
import base64
import io
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = bool(os.environ.get('VERCEL') or os.environ.get('VERCEL_ENV'))

# Render's entry-level instances are CPU-only and memory constrained.  Keeping
# inference images bounded prevents one request from exhausting the worker.
INFERENCE_SIZE = int(os.environ.get('INFERENCE_SIZE', '256'))
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', '10'))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-secret-in-production')
# Vercel functions have a read-only project directory.  ``/tmp`` is the only
# writable location and is intentionally treated as temporary storage.
app.config['UPLOAD_FOLDER'] = (
    os.path.join('/tmp', 'nst-uploads') if IS_VERCEL
    else os.path.join(BASE_DIR, 'static', 'uploads')
)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Avoid CPU oversubscription when Gunicorn runs in a small Render container.
if device.type == 'cpu':
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '2')))

encoder = VGGEncoder(os.path.join(BASE_DIR, 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth'),
        map_location=device
    )
)


encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    # ``Resize(size)`` only changes the shorter edge, so a panoramic upload can
    # still allocate a huge tensor. ``thumbnail`` bounds both dimensions while
    # retaining the aspect ratio.
    content_image.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE), Image.Resampling.LANCZOS)
    style_image.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE), Image.Resampling.LANCZOS)

    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def image_data_url(image):
    """Return a small inline JPEG so serverless result downloads are reliable."""
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    buffer = io.BytesIO()
    transforms.ToPILImage()(image).save(buffer, format='JPEG', quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    result_url = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST' and form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = f"{uuid.uuid4().hex}_{secure_filename(form.content.data.filename)}"
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = f"{uuid.uuid4().hex}_{secure_filename(form.style.data.filename)}"
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
                if IS_VERCEL:
                    result_url = image_data_url(stylized_image)
            except Exception as e:
                error = str(e)
    elif request.method == 'POST':
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, result_url=result_url, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)
