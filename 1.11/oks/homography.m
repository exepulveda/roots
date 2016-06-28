function [tform]=homography(moving, fixed)

[optimiser, metric]=imregconfig('multimodal');

tform = imregtform(moving,fixed,'Similarity',optimiser,metric);

return

