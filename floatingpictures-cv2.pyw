#!/usr/bin/env python
# pylint: disable=abstract-method,no-member
"""
Floating pictures screensaver written to mimic the original Apple TV screensaver.

Created By: Lebonez
"""
import sys
import os
import time
import io
import logging
import queue
import configparser

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
from threading import Thread

import cv2
import numpy as np
import pyglet


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FloatingPictures(pyglet.window.Window):
    """
    Build pyglet window as FloatingPictures screen saver to manage pictures as sprites.
    Allows for significant configuration compared to original Apple TV screensaver.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, images_path=os.getcwd(), fullscreen=True, debug=False, speed=240.0,
                 max_picture_speed=0.5, min_picture_speed=0.0625, max_picture_width=0.4,
                 min_picture_width=0.15, max_images=40, subpixel_movement=True,
                 update_image_interval=2, shadow_darkness=100, shadow_size=7,
                 num_variations=20):
        super().__init__(fullscreen=fullscreen)
        self.debug = debug
        self.min_picture_width = min_picture_width
        self.max_picture_width = max_picture_width
        self.min_picture_speed = min_picture_speed
        self.max_picture_speed = max_picture_speed
        self.max_images = max_images
        self.subpixel_movement = subpixel_movement
        self.update_image_interval = update_image_interval
        self.shadow_darkness = shadow_darkness
        self.shadow_size = shadow_size
        # Speed is technically the inverse of the frame rate.
        self.speed = 1 / speed
        self.scales = np.linspace(min_picture_width, max_picture_width, num_variations)
        # Make sure smaller pictures occur a bit more often to fill the screen.
        self.probability = np.array([1 / i**(1/4) for i in range(1, num_variations+1)])
        self.probability = self.probability / self.probability.sum()
        self.batch = pyglet.graphics.Batch()
        self.images_path = images_path
        self.image_sprites = []
        self.image_queue = queue.Queue(maxsize=10)
        self.previous_sprite = []
        self.image_paths = []

    def start(self):
        """
        Handle the image queue thread and build the clock scheduler and graphics config
        then finish with app run.
        """
        thread = Thread(target=self.queue_image)
        thread.daemon = True
        thread.start()
        pyglet.clock.schedule_interval(self.update_image, self.update_image_interval)
        pyglet.clock.schedule_interval(self.update_pan, self.speed)
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA,
            pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        self.set_mouse_visible(False)
        pyglet.app.run()

    def on_draw(self):
        """
        Clear window and draw the batch of sprites.
        """
        self.clear()
        self.batch.draw()

    @staticmethod
    def get_image_paths(images_path):
        """
        Recursively grab matching images in the user specified folder.
        """
        paths = []
        for root, _, files in os.walk(images_path, topdown=True):
            for file in sorted(files):
                path = os.path.abspath(os.path.join(root, file))
                paths.append(path)
        return paths

    def update_pan(self, _):
        """
        Move sprites up by their z component. 
        Alternatively if set to false move in defined single pixel rather than subpixel movement.
        Handles deletion of sprites after they have floated to top of screen.
        """
        actual_sprites = []
        for image_sprite in self.image_sprites:
            if self.subpixel_movement:
                image_sprite[0].y += image_sprite[0].z
            else:
                image_sprite[1] += image_sprite[0].z
                if image_sprite[1] >= 1:
                    image_sprite[0].y += 1
                    image_sprite[1] = 0
            if image_sprite[0].y > self.height:
                image_sprite[0].delete()
            else:
                actual_sprites.append(image_sprite)
        self.image_sprites = actual_sprites

    def queue_image(self):
        """
        Take the list of images and prep them to be used as sprites.
        This function is daemon threaded to reduce the computationally expensive
        process of interpolating images and creating shadows.
        """
        self.image_paths = self.get_image_paths(self.images_path)
        np.random.shuffle(self.image_paths)
        while True:
            if len(self.image_paths) == 0:
                self.image_paths = self.get_image_paths(self.images_path)
                np.random.shuffle(self.image_paths)
            if not self.image_queue.full():
                image_path = self.image_paths.pop()
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Skipping {image_path}...")
                    continue
                choice = np.random.choice(self.scales, p=self.probability)
                # Portrait mode scale on height instead.
                if img.shape[0] > img.shape[1]:
                    height = int(self.width * choice)
                    width = int(np.ceil(img.shape[1] * height / img.shape[0]))
                else:
                    width = int(self.width * choice)
                    height = int(np.ceil(img.shape[0] * width / img.shape[1]))
                frame = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                frame = self.image_shadow(frame)
                is_success, buffer = cv2.imencode(".png", frame)
                if is_success:
                    io_buf = io.BytesIO(buffer)
                    img = pyglet.image.load('.png', file=io_buf)
                    self.image_queue.put([img, choice])
            else:
                time.sleep(1)

    def image_shadow(self, frame):
        """
        Expand image by shadow configuration options and add alpha and a simple shadow to image.
        """
        # Invert shadow darkness so lower values result is less shadow
        shadow_darkness = 255 - self.shadow_darkness
        # get alpha from picture which is just an array of 255s
        alpha = frame[:,:,3]
        # Increase size of array on the top and bottom
        # For some unknown and annoying reason we must construct the shadow oversized...
        alpha = np.pad(alpha,
            ((self.shadow_size, self.shadow_size), (self.shadow_size, self.shadow_size)),
            'linear_ramp', end_values=shadow_darkness)
        # Move corners to correct shadow pointing top left
        alpha[-self.shadow_size*2 - self.shadow_size:-self.shadow_size*2,-self.shadow_size:] \
            = alpha[-self.shadow_size:,-self.shadow_size:]
        alpha[:self.shadow_size,self.shadow_size*2:self.shadow_size*2 + self.shadow_size] \
            = alpha[:self.shadow_size,:self.shadow_size]
        # Reduce shadow by minimum value to scale it to one prevents it from being too dark.
        alpha[:self.shadow_size,:] -= np.min(alpha[:self.shadow_size,:])
        alpha[self.shadow_size:,-self.shadow_size:] -= \
            np.min(alpha[self.shadow_size:,-self.shadow_size:])
        # Remove shadow from corners to make perception of top right facing shadow.
        alpha[-self.shadow_size:,:] = 0
        alpha[:,:self.shadow_size] = 0
        alpha[:self.shadow_size,:self.shadow_size*2] = 0
        alpha[-self.shadow_size*2:,-self.shadow_size:] = 0
        # Add zero rgb to top and bottom for shadow
        frame =  np.pad(frame, ((self.shadow_size, self.shadow_size),
            (self.shadow_size, self.shadow_size), (0, 0)), 'constant')
        # Apply linear ramped alpha to actual photo.
        frame[:,:,3] = alpha
        return frame


    def update_image(self, _):
        """
        Check if we can create a new image. This is the meat of the process.
        Takes images from the threaded queue creates the sprite and figures out where and if
        it can fit in the negative y space below the screen.
        It selects a random location where the image will fit or puts the current image back at
        front of queue then waits for next update interval.
        """
        if not self.image_queue.empty() and len(self.image_sprites) < self.max_images:
            img, choice = self.image_queue.get_nowait()
            sprite = pyglet.sprite.Sprite(img, batch=self.batch)
            sprite.image = img
            potential_xs = np.ones(self.width)
            for ysprite in self.image_sprites:
                if ysprite[0].y < 0:
                    potential_xs[ysprite[0].x:ysprite[0].x + ysprite[0].width] = 0
            sections = np.where(np.diff(np.pad(potential_xs, 1)) != 0)[0].reshape(-1,2)-[0,1]
            potentials = []
            for section in sections:
                start, end = section
                if end - start < sprite.width:
                    continue
                potentials.extend(list(range(start, end - sprite.width)))
            if not potentials:
                self.image_queue.put([img, choice])
                sprite.delete()
                return
            sprite.x = np.random.choice(potentials)
            sprite.y = -sprite.height - 1
            sprite.z = (((choice - self.min_picture_width) *
                (self.max_picture_speed - self.min_picture_speed)) /
                (self.max_picture_width - self.min_picture_width)) + self.min_picture_speed
            self.previous_sprite = [sprite.x, sprite.y, sprite.height, sprite.width]
            self.image_sprites.append([sprite, 0])

    def on_mouse_motion(self, x, y, dx, dy):
        """
        Check for mouse motion if not in debug mode and close window.
        """
        if not self.debug:
            if x - dx > 5 or y - dy > 5:
                self.dispatch_event('on_close')

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Check if mouse was pressed if not in debug mode and close window.
        """
        if not self.debug:
            self.dispatch_event('on_close')

    def on_key_press(self, symbol, modifiers):
        """
        Close window on any key press unless in debug where escape only closes the window.
        """
        if not self.debug:
            self.dispatch_event('on_close')
        else:
            super().on_key_press(symbol, modifiers)


class FloatingPicturesConfig:
    """
    Handle a fairly naive config process for the windows screensaver option to use.
    Uses tkinet so it isn't pretty but it allows modification of all options.
    Many options scale on framerate and screen resolution/size so choose carefully.
    """

    def __init__(self):
        self.window = None
        self.image_paths = None
        self.config_options = [
            {'name': 'images_path',
            'description': f'Path to images to recusively get pictures (default {os.getcwd()}):',
            'default': os.getcwd(),
            'input_type': ['button', self.update_images_path],
            'type': str},
            {'name': 'max_images',
            'description': 'Max pictures to queue and prepare in ram (default 40):',
            'default': 40,
            'type': int},
            {'name': 'max_picture_speed',
            'description': 'Max picture speed in pixels moved per frame (default 0.5):',
            'default': 0.5,
            'type': float},
            {'name': 'max_picture_width',
            'description': 'Max picture width in fraction of window width (default 0.4):',
            'default': 0.4,
            'type': float},
            {'name': 'min_picture_speed',
            'description': 'Min picture speed in pixels moved per frame (default 0.0625):',
            'default': 0.0625,
            'type': float},
            {'name': 'min_picture_width',
            'description': 'Min picture speed in fraction of window width (default 0.15):',
            'default': 0.15,
            'type': float},
            {'name': 'update_image_interval',
            'description': 'Interval in seconds to spawn new pictures if they fit (default 2):',
            'default': 2,
            'type': int},
            {'name': 'shadow_darkness',
            'description': 'Higher values increase shadow darkness max 255 min 0 (default 100):',
            'default': 100,
            'type': int},
            {'name': 'shadow_size',
            'description': 'Shadow width in pixels (default 7):',
            'default': 7,
            'type': int},
            {'name': 'speed',
            'description': 'How many frames per second (default 240):',
            'default': 240.0,
            'type': float},
            {'name': 'subpixel_movement',
            'description': 'Set to 1 to turn on and 0 to turn off subpixel movement (default 1):',
            'default': True,
            'type': bool},
            {'name': 'num_variations',
            'description': 'Number of pictures variations of size and speed (default 10):',
            'default': 10,
            'type': int},
        ]
        self.config_file = self.get_config()

    def get_config(self):
        """
        Get config from ini file if it exists. If not skip to adding and setting defaults for now.
        """
        config_file = configparser.ConfigParser()
        config_file.read(
            f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/floating_pictures.ini')
        for config_option in self.config_options:
            if config_option['name'] not in config_file['DEFAULT']:
                config_file.set('DEFAULT', config_option['name'], str(config_option['default']))
            config_option['value'] = config_option['type'](
                config_file.get('DEFAULT', config_option['name']))
        with open(f'{os.path.dirname(os.path.abspath(
                sys.argv[0]))}/floating_pictures.ini', 'w+', encoding='utf-8') as fh:
            config_file.write(fh)
        return config_file

    def update_config(self):
        """
        Function that handles if any setttings were changed in the tkinet menu.
        Presents an error if a setting isn't correct.
        Note that it is pretty niave and won't check for proper values but only checks type.
        """
        error_window = None
        row = 0
        for config_option in self.config_options:
            try:
                if 'input_type' in config_option:
                    config_option['value'] = config_option['type'](config_option['var'].get())
                else:
                    config_option['value'] = config_option['type'](config_option['input'].get())
                self.config_file.set('DEFAULT', config_option['name'], str(config_option['value']))
            except ValueError:
                if error_window is None:
                    error_window = tk.Tk()
                    error_window.title("Floating Pictures Settings Error")
                    error_window.eval('tk::PlaceWindow . center')
                    ttk.Style(error_window).theme_use()
                config_option['info'] = tk.Label(error_window,
                    text=
                    f'{config_option['name']} must be of type {config_option['type'].__name__}.')
                config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
                row += 1
        if error_window is not None:
            button_ok = ttk.Button(error_window, text="OK", command=error_window.destroy)
            button_ok.grid(row=row, column=0, sticky="esn", padx=5, pady=5)
            error_window.mainloop()
            return
        with open(f'{os.path.dirname(os.path.abspath(
                sys.argv[0]))}/floating_pictures.ini', 'w+', encoding='utf-8') as fh:
            self.config_file.write(fh)
        self.window.destroy()

    def update_images_path(self):
        """
        Special option that shows and select directory window provided by windows.
        """
        for config_option in self.config_options:
            if config_option['name'] == 'images_path':
                config_option['var'].set(askdirectory())
                break

    def menu(self):
        """
        Show the menu that windows screensaver uses.
        Just takes config options and presents them graphically and modifiable with
        update_config function.
        """
        self.window = tk.Tk()
        self.window.title("Floating Pictures Settings")
        ttk.Style(self.window).theme_use()
        row = 0
        for config_option in self.config_options:
            config_option['info'] = tk.Label(self.window, text=config_option['description'])
            config_option['var'] = tk.StringVar(self.window, config_option['value'])
            if 'input_type' in config_option and 'button' == config_option['input_type'][0]:
                config_option['input'] = ttk.Button(self.window, textvariable=config_option['var'],
                    command=config_option['input_type'][1])
                config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
                config_option['input'].grid(row=row, column=1, sticky="esn", padx=2, pady=2)
            else:
                config_option['input'] = ttk.Entry(self.window, textvariable=config_option['var'])
                config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
                config_option['input'].grid(row=row, column=1, sticky="esn", padx=2, pady=2)
            row += 1
        button_save = ttk.Button(self.window, text="Save", command=self.update_config)
        button_cancel = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        button_save.grid(row=row, column=0, sticky="esn", padx=5, pady=5)
        button_cancel.grid(row=row, column=1, sticky="esn", padx=5, pady=5)
        self.window.mainloop()


def usage():
    """
    argparse wasn't working properly with windows slash options so wrote a naive version.
    """
    print(f'''
usage: {sys.argv[0]} [/s|-s] [/c|-c] [/d|-d] [/h|-h]

positional arguments:
  /s|-s                 Start the screensaver in fullscreen mode.
  /c|-c                 Show the configuration settings dialog box.
  /d|-d                 Start the screensaver in windowed mode.
                        Press <escape> to exit debug mode.
  /h|-h                 Show this help message and exit
''')


def main():
    """
    Main function that handles arguments without argparse due to issues
    with windows slash options.
    """
    fpc = FloatingPicturesConfig()

    if len(sys.argv) == 2:
        arg = sys.argv[1].lower()
        if ':' in arg:
            arg = arg.split(':')[0]
        if ' ' in arg:
            arg = arg.split(' ')[0]
    else:
        usage()
        sys.exit()
    if arg in ['-s', '/s']:
        fp = FloatingPictures(**{conf['name']: conf['value'] for conf in fpc.config_options})
        fp.start()
    elif arg in ['-c', '/c']:
        fpc.menu()
        sys.exit()
    elif arg in ['-d', '/d']:
        fp = FloatingPictures(fullscreen=False, debug=True, **{conf['name']: conf['value'] for conf in fpc.config_options})
        fp.start()
    elif arg in ['-h', '/h']:
        usage()
        sys.exit()
    else:
        sys.exit()


if __name__ == '__main__':
    main()
