from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_chat_echo():
    resp = client.post('/api/chat', json={'message': 'hello'})
    assert resp.status_code == 200
    assert 'echo' in resp.json()