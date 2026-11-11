import os
import sys
import json

os.environ['MPLCONFIGDIR'] = r'C:\SignLanguageDetection3\.mplconfig'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

from app import app

client = app.test_client()

print('--- Testing GET / ---')
res = client.get('/')
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
print('Status:', res.status_code, 'HTML size:', len(res.data))

print('--- Testing GET /api/model_info ---')
res = client.get('/api/model_info')
assert res.status_code == 200
data = res.get_json()
print('Architecture:', data['architecture'], 'Classes count:', data['classes_count'])
assert data['classes_count'] == 36

print('--- Testing GET /api/samples ---')
res = client.get('/api/samples')
assert res.status_code == 200
data = res.get_json()
print('Total samples available:', len(data['samples']))
assert len(data['samples']) > 0

print('--- Testing POST /api/detect_image ---')
first_sample = data['samples'][0]['url']
res = client.post('/api/detect_image', json={
    'sample_path': first_sample,
    'confidence': 0.25,
    'iou': 0.45
})
assert res.status_code == 200
det_res = res.get_json()
print('Image detection status:', det_res['status'])
print('Detections count:', det_res['count'])
print('Latency:', det_res['latency_ms'], 'ms')
if det_res['detections']:
    print('First detection:', det_res['detections'][0]['name'], f"{det_res['detections'][0]['confidence_pct']}%")

print('--- Testing POST /api/detect_frame (Real-time frame simulation) ---')
# Test with base64 from the first detection annotated image
sample_b64 = det_res['original_image']
res = client.post('/api/detect_frame', json={
    'frame': sample_b64,
    'confidence': 0.25,
    'iou': 0.45
})
assert res.status_code == 200
frame_res = res.get_json()
print('Frame detection status:', frame_res['status'])
print('Frame latency:', frame_res['latency_ms'], 'ms')
print('Frame detections count:', frame_res['count'])

print('\n=============================================')
print('>>> ALL END-TO-END TESTS PASSED 100%! <<<')
print('=============================================')
