# https://docs.ultralytics.com/modes/predict/#plotting-results

from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\user\PycharmProjects\pythonProject4\runs\detect\train\weights\best.pt')

# Run inference on 'bus.jpg'
results = model(r'C:\Users\user\PycharmProjects\pythonProject4\Fish-44\test\images\FishDataset22_png.rf.db11b2a5d8d8a1e31fe1414981840ead.jpg')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('InferImage/results.jpg')  # save image
