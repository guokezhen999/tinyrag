from flask import Blueprint, request, jsonify, render_template, Response
import json

from sympy.polys.polyconfig import query

from . import agent_dialog_service

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """ 根 URL 路由，用于渲染主前端页面。 """
    return render_template('index.html')

@bp.route('/api/start_dialog', methods=['POST'])
def start_dialog():
    config_data = request.get_json()
    print(config_data)

    if not config_data:
        error_msg = json.dumps({'error': '请求体为空，需要提供对话配置'})
        return Response(error_msg, status=400, mimetype='application/json')

    try:
        stream_generator = agent_dialog_service.stream_dialog(config_data)
        return Response(stream_generator, mimetype='text/event-stream')
    except Exception as e:
        print(f"[ERROR] /api/start_dialog: {e}")
        error_msg = json.dumps({'error': f'启动对话失败: {e}'})
        return Response(error_msg, status=500, mimetype='application/json')