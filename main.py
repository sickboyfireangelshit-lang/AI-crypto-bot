from flask import Flask, render_template_string, jsonify
from core.bot import AITradingBot
import threading

app = Flask(__name__)
app.config.from_object('config.Config')

bot = AITradingBot()
threading.Thread(target=bot.run, daemon=True).start()

DASHBOARD = """
<h1>AI Crypto Bot Live 🔥</h1>
<h2>Status: Running</h2>
<div id="status">Loading...</div>
<script>
setInterval(() => fetch('/status').then(r => r.json()).then(d => document.getElementById('status').innerText = JSON.stringify(d)), 5000);
</script>
"""

@app.route('/')
def home():
    return render_template_string(DASHBOARD)

@app.route('/status')
def status():
    return jsonify(bot.get_status())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
