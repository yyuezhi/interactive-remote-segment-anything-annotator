# How to use Segmentation Server

# Interactive Image Annotation Tool

This repository contains a Python tool for interactive **mask annotation** of image with support for:
- Freehand painting / erasing
- Point-based interactive segmentation (connected to a backend, e.g. SAM2 server)
- Saving multiple formats of masks and overlays

It is ideally suitable for senarios where you could only run SAM2 remotely without monitor but you want to do SAM2W anotation interactively locally.

<img src="./asset/demo.gif" width="600"/>

## Install
1. Setup SAM2 at https://github.com/facebookresearch/sam2/tree/main
2. Establish SSH link using: ```ssh -L 8003:localhost:8003 [username]@[servername]  (change your own port)```
3. Put backend.py at server and run 
```python backend.py --host 0.0.0.0 --port 8003``` (You may need to change port)
4. At local end, ```python frontend.py``` 

## Controls

- **`w`** → Previous image  
- **`s`** → Save current image annotations and move to next image  
- **`Control + C`** → Exit the program  

- **`a`** → Decrement current label (clamped, no wraparound)  
- **`d`** → Increment current label (clamped, no wraparound)  

- **`z`** → Zoom mode  
  - Scroll wheel = zoom in/out (centered on cursor)  
  - Left-drag rectangle = zoom to region  

- **`x`** → Point mode  
  - Left click = positive point (foreground)  
  - Right click = negative point (background)  
  - Sends clicks to backend and rewrites mask for current label  

- **`q`** → Paint mode  
  - Left click-drag = paint current label  
  - Right click-drag = erase current label  

- **`e`** → Clear the current label mask for this image  
- **`c`** → Reset current image to full size and clear **all labels + points**  
- **`r`** → Save annotations for **all images**  

## Saving


At the top of the script, edit the dictionary:
```
SAVE_OPTIONS = {
    "image_mask":  True,  # save image with only labeled pixels (background black)
    "overlay":     True,  # save image + transparent overlay of all labels
    "label_single":True,  # save one binary PNG per label (if not empty)
    "label_multi": True,  # save a colorized label map
}
```

## License

MIT License. Free for academic and commercial use.