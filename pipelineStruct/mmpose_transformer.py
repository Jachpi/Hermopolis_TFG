from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(pose3d='human3d')
result_generator = inferencer('videos/clip.mp4', show=False)
result = next(result_generator)

# Ver estructura del resultado
pred = result['predictions'][0][0]
print(type(pred))
print(pred.keys() if hasattr(pred, 'keys') else dir(pred))
print(len(pred['keypoints']))
print(pred['keypoints'])
