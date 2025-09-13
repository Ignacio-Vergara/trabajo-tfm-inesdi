from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
import os
import time
import json
from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import datetime
import re

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Configurar autenticación básica
auth = HTTPBasicAuth()

# Configurar usuarios (en producción, usa una base de datos)
users = {
    "supervisor": generate_password_hash("telco123"),
    "admin": generate_password_hash("admin123")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# Configurar cliente de OpenAI con la nueva API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ASSISTANT_ID = "asst_Vsw5vw6TzjV1QYatry8dmrP7"  # Tu Assistant ID

# Cargar datasets y modelo
try:
    # Cargar el pipeline completo que ya incluye el preprocesador entrenado
    pipeline = joblib.load('modelo_lealtad.pkl')
    print("✅ Pipeline cargado correctamente")
    
    # Cargar datos de clientes
    df_clientes = pd.read_csv('Dataset_with_Loyalty_Fields.csv')
    print("✅ Datos de clientes cargados correctamente")
    
except Exception as e:
    print(f"❌ Error al cargar archivos: {str(e)}")
    exit(1)

# Extraer el preprocesador del pipeline
preprocessor = pipeline.named_steps['pre']

RESPUESTAS = {
    0: "🔴 Cliente con BAJA LEALTAD. Recomendación: Ofrecer promociones especiales y seguimiento cercano.",
    1: "🟡 Cliente con LEALTAD MEDIA. Sugerencia: Paquetes personalizados y beneficios adicionales.",
    2: "🟢 Cliente con ALTA LEALTAD. Acción: Programas de recompensas y solicitar referidos."
}

# Diccionario para almacenar threads de conversación por sesión
user_threads = {}

# Archivos Excel para almacenar interacciones
INTERACTIONS_FILE = 'interactions.xlsx'
LOYALTY_CHANGES_FILE = 'loyalty_changes.xlsx'

# Función personalizada de serialización para manejar tipos de numpy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)
    
def convert_numpy_types(obj):
    """Convertir tipos de numpy a tipos nativos de Python recursivamente"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # Para otros tipos de numpy
        return obj.item()
    else:
        return obj

# Configurar Flask para usar nuestro encoder personalizado
app.json_encoder = NumpyEncoder

def init_excel_files():
    """Inicializar archivos Excel si no existen"""
    if not os.path.exists(INTERACTIONS_FILE):
        df = pd.DataFrame(columns=[
            'session_id', 'customer_id', 'message', 'sender', 
            'timestamp', 'loyalty_level', 'loyalty_index', 'points_change'
        ])
        df.to_excel(INTERACTIONS_FILE, index=False)
    
    if not os.path.exists(LOYALTY_CHANGES_FILE):
        df = pd.DataFrame(columns=[
            'customer_id', 'old_level', 'new_level', 'old_index', 'new_index',
            'change_reason', 'timestamp'
        ])
        df.to_excel(LOYALTY_CHANGES_FILE, index=False)

init_excel_files()

def update_customer_loyalty(customer_id, points_change, reason):
    """Actualizar puntos de lealtad de un cliente en el CSV"""
    try:
        global df_clientes
        
        # Encontrar el índice del cliente
        customer_idx = df_clientes[df_clientes['customerID'] == customer_id].index
        
        if len(customer_idx) == 0:
            print(f"Cliente {customer_id} no encontrado")
            return None, None, None
        
        customer_idx = customer_idx[0]
        
        # Obtener valores actuales
        old_index = df_clientes.loc[customer_idx, 'loyalty_index']
        old_level = df_clientes.loc[customer_idx, 'loyalty_class']
        
        # Calcular nuevo puntaje (no menor a 0)
        new_index = max(0, old_index + points_change)
        
        # Actualizar el DataFrame
        df_clientes.loc[customer_idx, 'loyalty_index'] = new_index
        
        # Recalcular la clase de lealtad basado en los nuevos puntos
        if new_index < 20:
            new_level = 0  # Baja lealtad
        elif new_index < 40:
            new_level = 1  # Lealtad media
        else:
            new_level = 2  # Alta lealtad
            
        df_clientes.loc[customer_idx, 'loyalty_class'] = new_level
        
        # Guardar el CSV actualizado
        df_clientes.to_csv('Dataset_with_Loyalty_Fields.csv', index=False)
        
        # Registrar el cambio si hubo modificación
        if points_change != 0:
            save_loyalty_change(customer_id, old_level, new_level, old_index, new_index, reason)
        
        return new_index, new_level, old_level != new_level
        
    except Exception as e:
        print(f"Error actualizando lealtad: {str(e)}")
        return None, None, None

def calculate_points_from_message(message, sentiment):
    """Calcular puntos basados en el mensaje y sentimiento"""
    # Puntos base por interactuar
    points = 1
    
    # Detectar palabras clave positivas
    positive_keywords = ['gracias', 'excelente', 'genial', 'perfecto', 'ayuda', 'bueno', 'feliz', 'contento', 'satisfecho']
    negative_keywords = ['molesto', 'enojado', 'frustrado', 'terrible', 'horrible', 'pésimo', 'decepcionado', 'cancelar', 'queja']
    
    message_lower = message.lower()
    
    # Verificar palabras positivas
    positive_count = sum(1 for word in positive_keywords if word in message_lower)
    points += positive_count * 2
    
    # Verificar palabras negativas
    negative_count = sum(1 for word in negative_keywords if word in message_lower)
    points -= negative_count * 3
    
    # Ajustar basado en sentimiento (si está disponible)
    if sentiment == 'positive':
        points += 3
    elif sentiment == 'negative':
        points -= 2
    
    # Limitar el rango de puntos por interacción
    return max(-5, min(10, points))

def analyze_message_sentiment(message):
    """Análisis simple de sentimiento (puede mejorarse con NLP)"""
    message_lower = message.lower()
    
    positive_words = ['gracias', 'excelente', 'genial', 'perfecto', 'bueno', 'feliz', 'contento', 'satisfecho', 'ayuda', 'resuelto']
    negative_words = ['molesto', 'enojado', 'frustrado', 'terrible', 'horrible', 'pésimo', 'decepcionado', 'problema', 'error', 'mal']
    
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def save_interaction(session_id, customer_id, message, sender, loyalty_level, loyalty_index, points_change=0):
    """Guardar interacción en archivo Excel"""
    try:
        # Leer archivo existente
        try:
            df = pd.read_excel(INTERACTIONS_FILE)
        except:
            # Si el archivo no existe o está corrupto, crear uno nuevo
            df = pd.DataFrame(columns=[
                'session_id', 'customer_id', 'message', 'sender', 
                'timestamp', 'loyalty_level', 'loyalty_index', 'points_change'
            ])
        
        # Agregar nueva interacción
        new_row = {
            'session_id': session_id,
            'customer_id': customer_id,
            'message': message,
            'sender': sender,
            'timestamp': datetime.datetime.now().isoformat(),
            'loyalty_level': loyalty_level if loyalty_level is not None else 0,
            'loyalty_index': loyalty_index if loyalty_index is not None else 0,
            'points_change': points_change
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(INTERACTIONS_FILE, index=False)
        
    except Exception as e:
        print(f"Error guardando interacción: {str(e)}")

def save_loyalty_change(customer_id, old_level, new_level, old_index, new_index, change_reason):
    """Guardar cambio de lealtad en archivo Excel"""
    try:
        # Leer archivo existente
        df = pd.read_excel(LOYALTY_CHANGES_FILE)
        
        # Agregar nuevo cambio
        new_row = {
            'customer_id': customer_id,
            'old_level': old_level,
            'new_level': new_level,
            'old_index': old_index,
            'new_index': new_index,
            'change_reason': change_reason,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(LOYALTY_CHANGES_FILE, index=False)
        
    except Exception as e:
        print(f"Error guardando cambio de lealtad: {str(e)}")

def get_interaction_history(customer_id=None, limit=100):
    """Obtener historial de interacciones desde Excel"""
    try:
        df = pd.read_excel(INTERACTIONS_FILE)
        
        # Asegurarse de que las columnas numéricas tengan valores por defecto
        if 'loyalty_level' in df.columns:
            df['loyalty_level'] = df['loyalty_level'].fillna(0).astype(int)
        else:
            df['loyalty_level'] = 0
            
        if 'loyalty_index' in df.columns:
            df['loyalty_index'] = df['loyalty_index'].fillna(0).astype(int)
        else:
            df['loyalty_index'] = 0
            
        if 'points_change' in df.columns:
            df['points_change'] = df['points_change'].fillna(0).astype(int)
        else:
            df['points_change'] = 0
        
        if customer_id:
            df = df[df['customer_id'] == customer_id]
        
        df = df.sort_values('timestamp', ascending=False).head(limit)
        
        # Convertir a diccionario y asegurar tipos nativos de Python
        interactions = df.to_dict('records')
        return [convert_numpy_types(interaction) for interaction in interactions]
        
    except Exception as e:
        print(f"Error leyendo interacciones: {str(e)}")
        return []

def get_loyalty_changes(customer_id=None, limit=50):
    """Obtener cambios de lealtad desde Excel"""
    try:
        df = pd.read_excel(LOYALTY_CHANGES_FILE)
        
        if customer_id:
            df = df[df['customer_id'] == customer_id]
        
        df = df.sort_values('timestamp', ascending=False).head(limit)
        
        # Convertir a diccionario y asegurar tipos nativos de Python
        changes = df.to_dict('records')
        return [convert_numpy_types(change) for change in changes]
        
    except Exception as e:
        print(f"Error leyendo cambios de lealtad: {str(e)}")
        return []

def wait_for_run_completion(thread_id, run_id, timeout=30):
    """Esperar a que un run se complete con timeout"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        if run.status in ['completed', 'failed', 'cancelled', 'expired']:
            return run
        
        time.sleep(1)  # Esperar 1 segundo entre checks
    
    # Si llegamos aquí, timeout
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )
    return run

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/telco')
def telco_interface():
    return render_template('telco_interface.html')

@app.route('/chat')
def chat_interface():
    return render_template('chat_interface.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        customer_id = data.get('customerID', '').strip().upper()
        
        if not customer_id:
            return jsonify({'status': 'error', 'message': 'Por favor ingresa un customerID'}), 400
        
        # Buscar cliente en el dataset
        cliente = df_clientes[df_clientes['customerID'] == customer_id]
        
        if cliente.empty:
            return jsonify({'status': 'error', 'message': 'Cliente no encontrado'}), 404
        
        # Obtener solo las características usadas en el entrenamiento
        ignore_cols = ['customerID', 'Churn', 'loyalty_index', 'loyalty_class']
        feature_cols = [col for col in df_clientes.columns if col not in ignore_cols]
        
        # Preparar datos para predicción
        datos_cliente = cliente.iloc[0][feature_cols].to_dict()
        df = pd.DataFrame([datos_cliente])
        
        # Preprocesamiento
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        
        # Convertir SeniorCitizen a string (porque es categórico en el entrenamiento)
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        
        # Usar el pipeline completo para predecir
        prediccion = pipeline.predict(df)[0]
        
        # Obtener datos para el chatbot - asegurar conversión a tipos nativos de Python
        loyalty_index = int(cliente['loyalty_index'].iloc[0]) if 'loyalty_index' in cliente.columns else 0
        loyalty_class = int(cliente['loyalty_class'].iloc[0]) if 'loyalty_class' in cliente.columns else int(prediccion)
        
        # Convertir todos los valores numpy a tipos nativos de Python
        customer_data = {
            'customerID': str(customer_id),
            'tenure': int(df['tenure'].iloc[0]),
            'MonthlyCharges': float(df['MonthlyCharges'].iloc[0]),
            'TotalCharges': float(df['TotalCharges'].iloc[0]),
            'Contract': str(df['Contract'].iloc[0]),
            'Churn': str(cliente['Churn'].iloc[0]),
            'loyalty_index': int(loyalty_index),
            'loyalty_class': int(loyalty_class)
        }
        
        # Asegurar que todos los valores sean tipos nativos de Python
        customer_data = convert_numpy_types(customer_data)
        
        respuesta = {
            'status': 'success',
            'prediction': int(prediccion),
            'response': RESPUESTAS[int(prediccion)],
            'customer_data': customer_data
        }
        
        print(f"✅ Cliente {customer_id} analizado - Lealtad: {prediccion}")
        return jsonify(respuesta)

    except Exception as e:
        error_msg = f"Error al procesar la solicitud: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/start_chat', methods=['POST'])
def start_chat():
    """Iniciar una nueva conversación con el Assistant"""
    try:
        data = request.get_json()
        customer_id = data.get('customerID', '').strip().upper()
        session_id = data.get('sessionID', 'default')
        
        if not customer_id:
            return jsonify({'status': 'error', 'message': 'CustomerID requerido'}), 400
        
        # Obtener datos del cliente para el contexto
        cliente = df_clientes[df_clientes['customerID'] == customer_id]
        if cliente.empty:
            return jsonify({'status': 'error', 'message': 'Cliente no encontrado'}), 404
        
        # Obtener valores actuales de lealtad
        loyalty_index = int(cliente['loyalty_index'].iloc[0]) if 'loyalty_index' in cliente.columns else 0
        loyalty_class = int(cliente['loyalty_class'].iloc[0]) if 'loyalty_class' in cliente.columns else 0
        
        # Crear contexto inicial para el Assistant
        contexto_cliente = f"""
        INFORMACIÓN DEL CLIENTE:
        - CustomerID: {customer_id}
        - Nivel de lealtad: {loyalty_class} ({RESPUESTAS[loyalty_class].split('.')[0]})
        - Puntos de lealtad: {loyalty_index}
        - Antigüedad: {int(cliente['tenure'].iloc[0])} meses
        - Plan mensual: ${float(cliente['MonthlyCharges'].iloc[0])}
        - Cargo total: ${float(cliente['TotalCharges'].iloc[0])}
        - Tipo de contrato: {cliente['Contract'].iloc[0]}
        - Estado de bajas: {cliente['Churn'].iloc[0]}
        
        Este es un cliente de TelcoPlus. Adapta tu conversación según su nivel de lealtad.
        """
        
        # Crear un nuevo thread para esta conversación
        thread = client.beta.threads.create()
        
        # Agregar el contexto del cliente como primer mensaje
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=contexto_cliente
        )
        
        # Guardar el thread en la sesión
        user_threads[session_id] = {
            'thread_id': thread.id,
            'customer_id': customer_id,
            'loyalty_level': loyalty_class,
            'loyalty_index': loyalty_index,
            'active_run': None
        }
        
        # Crear un run para procesar el mensaje inicial
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )
        
        # Esperar a que el run se complete
        run = wait_for_run_completion(thread.id, run.id)
        
        # Obtener la respuesta del assistant
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            # El último mensaje es la respuesta del assistant
            assistant_response = messages.data[0].content[0].text.value
            
            # Guardar interacción inicial del bot
            save_interaction(
                session_id, 
                customer_id, 
                assistant_response, 
                'bot', 
                loyalty_class,
                loyalty_index
            )
            
            return jsonify({
                'status': 'success',
                'response': assistant_response,
                'thread_id': thread.id,
                'loyalty_level': loyalty_class,
                'loyalty_index': loyalty_index,
                'session_id': session_id
            })
        else:
            error_msg = f'Error en el assistant: {run.status}'
            if run.last_error:
                error_msg += f" - {run.last_error.message}"
            return jsonify({
                'status': 'error', 
                'message': error_msg
            }), 500
            
    except Exception as e:
        error_msg = f"Error al iniciar chat: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/chat_message', methods=['POST'])
def chat_message():
    """Enviar un mensaje al Assistant y obtener respuesta"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('sessionID', 'default')
        
        if not message:
            return jsonify({'status': 'error', 'message': 'Mensaje vacío'}), 400
        
        if session_id not in user_threads:
            return jsonify({'status': 'error', 'message': 'Sesión no encontrada. Inicia una nueva conversación.'}), 404
        
        thread_info = user_threads[session_id]
        thread_id = thread_info['thread_id']
        customer_id = thread_info['customer_id']
        
        # Analizar sentimiento del mensaje
        sentiment = analyze_message_sentiment(message)
        
        # Calcular puntos basados en el mensaje
        points_change = calculate_points_from_message(message, sentiment)
        
        # Actualizar lealtad del cliente
        new_index, new_level, level_changed = update_customer_loyalty(
            customer_id, 
            points_change, 
            f"Interacción en chat: {message[:50]}..."
        )
        
        if new_index is not None:
            # Actualizar información en la sesión
            thread_info['loyalty_index'] = new_index
            thread_info['loyalty_level'] = new_level
            user_threads[session_id] = thread_info
        
        # Guardar interacción del usuario
        save_interaction(
            session_id, 
            customer_id, 
            message, 
            'user', 
            new_level if new_level is not None else thread_info['loyalty_level'],
            new_index if new_index is not None else thread_info['loyalty_index'],
            points_change
        )
        
        # Verificar si hay un run activo
        if thread_info.get('active_run'):
            try:
                active_run = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=thread_info['active_run']
                )
                if active_run.status in ['queued', 'in_progress']:
                    return jsonify({
                        'status': 'error', 
                        'message': 'Por favor espera a que termine la respuesta anterior.'
                    }), 429
            except:
                thread_info['active_run'] = None
        
        # Agregar el mensaje del usuario al thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Crear un run para procesar el mensaje
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # Guardar el ID del run activo
        thread_info['active_run'] = run.id
        user_threads[session_id] = thread_info
        
        # Esperar a que el run se complete
        run = wait_for_run_completion(thread_id, run.id)
        
        # Limpiar el run activo
        thread_info['active_run'] = None
        user_threads[session_id] = thread_info
        
        # Obtener la respuesta del assistant
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            assistant_response = messages.data[0].content[0].text.value
            
            # Guardar interacción del bot
            save_interaction(
                session_id, 
                customer_id, 
                assistant_response, 
                'bot', 
                new_level if new_level is not None else thread_info['loyalty_level'],
                new_index if new_index is not None else thread_info['loyalty_index']
            )
            
            # Preparar respuesta con conversión explícita a tipos nativos
            loyalty_level_val = new_level if new_level is not None else thread_info['loyalty_level']
            loyalty_index_val = new_index if new_index is not None else thread_info['loyalty_index']
            
            response_data = {
                'status': 'success',
                'response': str(assistant_response),
                'loyalty_level': int(loyalty_level_val),
                'loyalty_index': int(loyalty_index_val),
                'points_change': int(points_change)
            }
            
            # Debuggear serialización antes de enviar
            debug_json_serialization(response_data)
            
            return jsonify(response_data)
        else:
            error_msg = f'Error en el assistant: {run.status}'
            if run.last_error:
                error_msg += f" - {run.last_error.message}"
            
            # Asegurar que el error message sea string
            error_response = {
                'status': 'error', 
                'message': str(error_msg)
            }
            return jsonify(error_response), 500
            
    except Exception as e:
        error_msg = f"Error en chat: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Asegurar que el error message sea string
        error_response = {
            'status': 'error', 
            'message': str(error_msg)
        }
        return jsonify(error_response), 500

# Endpoints para la vista de supervisor (se mantienen igual)
@app.route('/supervisor')
@auth.login_required
def supervisor_dashboard():
    return render_template('supervisor_dashboard.html')

@app.route('/api/supervisor/interactions')
@auth.login_required
def get_supervisor_interactions():
    customer_id = request.args.get('customer_id')
    limit = int(request.args.get('limit', 100))
    
    interactions = get_interaction_history(customer_id, limit)
    return jsonify({'status': 'success', 'data': interactions})

@app.route('/api/supervisor/loyalty-changes')
@auth.login_required
def get_supervisor_loyalty_changes():
    customer_id = request.args.get('customer_id')
    limit = int(request.args.get('limit', 50))
    
    changes = get_loyalty_changes(customer_id, limit)
    return jsonify({'status': 'success', 'data': changes})

@app.route('/api/supervisor/customers')
@auth.login_required
def get_supervisor_customers():
    try:
        # Obtener lista de clientes únicos que han interactuado
        df = pd.read_excel(INTERACTIONS_FILE)
        customers = df['customer_id'].unique().tolist()
        customers = [c for c in customers if pd.notna(c)]  # Filtrar valores NaN
        
        return jsonify({'status': 'success', 'data': sorted(customers)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/supervisor/stats')
@auth.login_required
def get_supervisor_stats():
    try:
        # Estadísticas de interacciones
        df_interactions = pd.read_excel(INTERACTIONS_FILE)
        total_interactions = len(df_interactions)
        unique_customers = df_interactions['customer_id'].nunique()
        
        # Estadísticas de cambios de lealtad
        try:
            df_changes = pd.read_excel(LOYALTY_CHANGES_FILE)
            total_loyalty_changes = len(df_changes)
        except:
            total_loyalty_changes = 0
        
        # Distribución de niveles de lealtad
        loyalty_distribution = {}
        if not df_interactions.empty:
            loyalty_counts = df_interactions['loyalty_level'].value_counts()
            for level, count in loyalty_counts.items():
                if pd.notna(level):
                    loyalty_distribution[int(level)] = int(count)
        
        # Asegurar que todos los valores sean tipos nativos
        stats_data = {
            'total_interactions': int(total_interactions),
            'unique_customers': int(unique_customers),
            'total_loyalty_changes': int(total_loyalty_changes),
            'loyalty_distribution': convert_numpy_types(loyalty_distribution)
        }
        
        return jsonify({
            'status': 'success',
            'data': stats_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def debug_json_serialization(obj, path='root'):
    """Función para debuggear problemas de serialización JSON con más detalle"""
    try:
        json.dumps(obj, cls=NumpyEncoder)
        return True
    except TypeError as e:
        print(f"🚨 Error de serialización en {path}: {e}")
        print(f"   Tipo: {type(obj)}")
        print(f"   Valor: {obj}")
        
        # Si es un diccionario, verificar cada key
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    json.dumps(value, cls=NumpyEncoder)
                except TypeError:
                    print(f"   → Key problemática: '{key}' con valor: {value} (tipo: {type(value)})")
                    debug_json_serialization(value, f"{path}.{key}")
        
        # Si es una lista, verificar cada item
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                try:
                    json.dumps(item, cls=NumpyEncoder)
                except TypeError:
                    print(f"   → Ítem problemático en índice {i}: {item} (tipo: {type(item)})")
                    debug_json_serialization(item, f"{path}[{i}]")
        
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=True)