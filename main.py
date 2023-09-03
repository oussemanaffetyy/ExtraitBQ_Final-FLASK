from flask import Flask, request, send_file
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import predictions as pred
from flask import make_response
import csv
import io
import json
import pandas as pd

app = Flask(__name__)
app.secret_key = 'document_scanner_app'

docscan = utils.DocumentScan()
@app.route('/', methods=['GET', 'POST'])
def scandoc():
    if request.method == 'POST':
        files = request.files.getlist('image_name')
        results = {}
        
        for file in files:
            upload_image_path = settings.join_path(settings.MEDIA_DIR, file.filename)
            file.save(upload_image_path)
            print('Image saved in =', upload_image_path)
            
            wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, file.filename) 
            image = cv2.imread(wrap_image_filepath)
            image_bb, prediction_results = pred.getPredictions(image)
            
            bb_filename = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg') 
            cv2.imwrite(bb_filename, image_bb)
            
            results[file.filename] = prediction_results
        
        return render_template('predictions.html', results=results)

    return render_template('scanner.html')

@app.route('/prediction')
def prediction():
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, 'magic_color.jpg') 
    image = cv2.imread(wrap_image_filepath)
    image_bb, results = pred.getPredictions(image)
    
    bb_filename = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg') 
    cv2.imwrite(bb_filename, image_bb)
    
    return render_template('predictions.html', results=results)


@app.route('/transform',methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        #utils.save_image(magic_color,'magic_color.jpg')
        filename =  'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR,filename)
        cv2.imwrite(magic_image_path,magic_color)
        
        return 'sucess'
    except:
        return 'fail'




@app.route('/save_csv', methods=['POST'])
def save_csv():
    if request.method == 'POST':
        results = request.form.getlist('results[]')

        # Créer un DataFrame pandas pour stocker les résultats
        df = pd.DataFrame(columns=['ORG', 'DATE', 'MONEY', 'CARDINAL'])

        # Parcourir les résultats et ajouter les données au DataFrame
        for result in results:
            prediction = eval(result)  # Convertir la chaîne en dictionnaire Python

            org = ', '.join(prediction['ORG']) if 'ORG' in prediction else ''
            
            date = ', '.join(prediction['DATE']) if 'DATE' in prediction else ''
            
            money = ', '.join(prediction['MONEY']) if 'MONEY' in prediction else ''
            
            cardinal = ', '.join(prediction['CARDINAL']) if 'CARDINAL' in prediction else ''

            new_row = pd.DataFrame({'ORG': [org], 'DATE': [date], 'MONEY': [money], 'CARDINAL': [cardinal]})
            df = pd.concat([df, new_row], ignore_index=True)

        # Générer le fichier CSV à partir du DataFrame
        csv_text = df.to_csv(index=False, sep='\t')

        # Créer une réponse Flask avec les données CSV
        response = make_response(csv_text)

        # Spécifier les en-têtes de la réponse pour indiquer qu'il s'agit d'un fichier CSV
        response.headers['Content-Disposition'] = 'attachment; filename=results.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response

if __name__ == "__main__":
    app.run(debug=True)