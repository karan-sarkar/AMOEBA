



from PIL import Image
import os
from pathlib import Path


### converted directory

if __name__ == '__main__':
    path = os.path.join('/data/bdd/cyclegan/day_night_scenario', 'converted')
    filenames = os.listdir(path)
    destination = os.path.join('/data/bdd/cyclegan/day_night_scenario', 'converted2')
    for filename in filenames:
        complete = os.path.join(path, filename)
        print(f'filename is: {complete}')
        filename_no_extension = Path(complete).stem
        ### we need to take out the .png
        destination_complete = os.path.join(destination, filename_no_extension+'.jpg')
        print(f"saving to: {destination_complete}")
        im1 = Image.open(complete)
        im1.save(destination_complete)

