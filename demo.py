from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import time
import pickle


register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
tic = time.time()
# please prepare an image with person
for i in range(0,1):
    results = inference_topdown(model, 'demo.jpg')
    print(i)
toc = time.time()

elapsed = toc-tic 
print(f"Mean time of {elapsed/10}")
print("Stop")
# Pickling the object

# Saving the pickled object to a file
with open('results.pkl', 'wb') as file:
    pickle.dump(results, file)