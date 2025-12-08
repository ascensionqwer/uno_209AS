from flask import Flask, render_template, jsonify, request
import sys
import os
import random

# Add parent directory to path to import game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui.game_manager import GameManager

app = Flask(__name__)
game_manager = GameManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_game', methods=['POST'])
def start_game():
    data = request.json
    opponent_type = data.get('opponent_type', 'offline')
    game_mode = data.get('game_mode', 'mini')
    seed = data.get('seed')
    
    if seed is None:
        seed = random.randint(0, 10000)
        
    game_manager.start_new_game(opponent_type, seed, game_mode)
    return jsonify(game_manager.get_game_state())

@app.route('/api/play_card', methods=['POST'])
def play_card():
    data = request.json
    card_index = data.get('card_index')
    
    success, message = game_manager.player_play_card(card_index)
    
    response = game_manager.get_game_state()
    response['success'] = success
    response['message'] = message
    return jsonify(response)

@app.route('/api/draw_card', methods=['POST'])
def draw_card():
    success, message = game_manager.player_draw_card()
    
    response = game_manager.get_game_state()
    response['success'] = success
    response['message'] = message
    return jsonify(response)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    move_info = game_manager.ai_make_move()
    
    response = game_manager.get_game_state()
    response['ai_move'] = move_info
    return jsonify(response)

@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    return jsonify(game_manager.get_game_state())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
