# googly-eyes
Funny eye segmentation project

## UI Demo
![Quick Streamlit demo](demo-googly-eyes.gif)

## Run
```
cd googly-eyes
docker compose up -d
```

- go to http://localhost:8501/ for Streamlit UI
- go to http://localhost:9090/docs to access API documentation of the backend service.


## System Limitations
1. This repository has not been run through static checks like pylint or black. For a production logic I would probably configure pre-commit hooks to make these checks out of the box.
1. There are inefficiencies all over:
    - The images have not been optimized, they are probably very large.
    - The docker images have many inefficiencies when building, to improve this I would add some reusable layers to save space and time.
    - No tests.
    - No model versioning. In this case since the model is over 100MB, it is not even included in this repository.
    - No CICD.
    - The inference is reshaping the images to 512x512 pixels, which means that images with a lot of people, or with faces too zoomed out will yield worse results. This is by desing, to keep the latencies low. Currently the whole operations takes <50ms.


## Model Summary
The underlying logic to replace the eyes by googly eyes consists on:
1. A pre-trained deeplabv3 model fine tuned with the [Mut1ny's dataset](https://store.mut1ny.com/product/face-head-segmentation-dataset-community-edition). Inspired by this [blog post](https://medium.com/technovators/semantic-segmentation-using-deeplabv3-ce68621e139e).
2. Then, a bit of post processing to mask the pixels corresponding to eyes.
3. After that, using cv2.contours, we can identify the clusters which normally map one to one to the eye.
4. And finally, some postprocessing to overlap a googly eye on top of the regular boring eye.

## Model Limitations
In summary, considering that this was a time restricted challenge, the model has not received the attention it would get for a real use case. To enumerate the limitations of this demo:

1. The first model has been taken. I could imagine it would be possible to further improve the results of the model by adjusting the preprocessing and the image selecting, and potentially growing the dataset with additional cases like people with glasses, or rooms with more people.
1. Exploring other metrics instead of f1-measure might be a good option, like Dice Coefficient, or Jaccard Index, especially for images that have small faces. 
1. The postprocessing has some flaws, like the eyes are not of the same size, or sometimes even some eyes are not detected. In a weird case, I also saw two googly eyes, one on top of each other. Many of these issues should be trivially solved by investing more time in the preprocessing.
