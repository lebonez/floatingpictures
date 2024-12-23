import os
import sys
import math
import time
import weakref
import logging
import random
import configparser
import queue
import ctypes

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
from threading import Thread

from PIL import Image, ImageOps, ImageFilter
import numpy as np
import pyglet
from pyglet.math import Mat4, Vec3
from pillow_heif import register_heif_opener


register_heif_opener()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_image_paths(images_path):
    """
    Recursively grab matching images in the user specified folder.
    """
    return [os.path.abspath(os.path.join(root, file))
            for root, _, files in os.walk(images_path, topdown=True)
            for file in sorted(files)]


class FloatingPictures:
    def __init__(self, window, images_path=os.getcwd(), debug=False,
            spin_speed=2, number_image_variations=10, image_scale=0.4, image_frequency=1, spin_frequency=60):
        self.debug = debug
        self.spin_speed = spin_speed
        self.number_image_variations = number_image_variations
        self.image_scale = image_scale
        self.spin_frequency = spin_frequency
        self.z_near = 1
        self.z_far = 10001
        self.z_shift = 2000
        self.initial_position = Vec3(0.0, 0.0, self.z_far + self.z_shift)
        self.position = self.initial_position
        self.up = Vec3(0.0, 1.0, 0.0)
        self.fov = 45
        self.images_path = images_path
        self.image_queue = queue.Queue(maxsize=10)
        self.sprites = []
        self.spin = False
        self.time = 0
        self.next_image = None
        self.next_x = None
        self.next_z = None
        self.prime = 0.0
        self._window = weakref.proxy(window)
        self._window.push_handlers(self)
        self.start_image_thread()
        pyglet.clock.schedule_interval(self.update_image, image_frequency)
        pyglet.clock.schedule_interval(self.do_spin, self.spin_frequency)
        self._window.set_mouse_visible(False)
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

    def on_close(self):
        self._window.close()

    def on_draw(self):
        self._window.clear()
        self.sprites.sort(key=lambda sprite: abs(self.position[2] - sprite.z), reverse=True)
        for sprite in self.sprites:
            sprite.draw()

    def on_resize(self, width, height):
        self._window.viewport = (0, 0, width, height)
        self._window.projection = Mat4.perspective_projection(
            self._window.aspect_ratio, z_near=self.z_near, z_far=self.z_far + self.z_shift, fov=self.fov)
        self.y_scaler = math.tan(self.fov * 0.5 * (math.pi / 180))
        self.max_z = self.initial_position[2] - math.ceil(self._window.height * 0.5 / self.y_scaler)
        self.target = Vec3(0.0, 0.0, int((self.z_far + (self.z_shift / 2))))
        self._window.view = Mat4.look_at(self.initial_position, self.target, self.up)
        self.potential_locations = self.get_potential_locations()
        return pyglet.event.EVENT_HANDLED

    def on_refresh(self, dt):
        if self.prime < self.spin_frequency:
            self.prime += dt
            if self.prime >= self.spin_frequency:
                self.prime = self.spin_frequency
        if self.spin:
            self.handle_spin(dt)
        else:
            self.update_sprites()

    def start_image_thread(self):
        thread = Thread(target=self.queue_image)
        thread.daemon = True
        thread.start()

    def handle_spin(self, dt):
        self.time -= self.spin_speed * dt * math.sin(self.time * 1/2) ** 2 + dt
        time_factor = (1 - math.cos(self.spin_speed * self.time)) / 2
        max_radius = (self.initial_position[2] - self.target[2] + 1.5 * self._window.width)
        min_radius = (self.initial_position[2] - self.target[2])
        if -math.pi / self.spin_speed <= self.time <= 0 or -2 * math.pi <= self.time <= -2 * math.pi + math.pi / self.spin_speed:
            radius = min_radius + (max_radius - min_radius) * time_factor
        else:
            radius = min_radius + (max_radius - min_radius)
        x = radius * math.sin(self.time)
        z = radius * math.cos(self.time) + self.target[2]

        self.position = Vec3(x, 0, z)
        
        if self.time < -2 * math.pi:
            self.position = self.initial_position
            self.spin = False
            self.time = 0

        self._window.view = Mat4.look_at(self.position, self.target, self.up)

    def update_sprites(self):
        sprites_to_remove = []
        for sprite in self.sprites:
            if not sprite.x or not sprite.z:
                logger.error(f'Skipping sprite due to bad attribute: {sprite.x} {sprite.z}')
                sprites_to_remove.append(sprite)
                continue
            sprite.y += 1
            if sprite.y > (self.initial_position[2] - sprite.z) * self.y_scaler * 2.0:
                sprites_to_remove.append(sprite)
        for sprite in sprites_to_remove:
            sprite.delete()
            self.sprites.remove(sprite)

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.debug and (x - dx > 5 or y - dy > 5):
            self._window.dispatch_event('on_close')

    def on_mouse_press(self, x, y, button, modifiers):
        if not self.debug:
            self._window.dispatch_event('on_close')

    def on_key_press(self, symbol, modifiers):
        if not self.debug:
            self._window.dispatch_event('on_close')
        else:
            self._window.on_key_press(symbol, modifiers)

    def do_spin(self, _):
        self.spin = True

    def update_image(self, _):
        if not self.spin:
            self.calc_next_image()
            if self.next_image and self.next_x and self.next_z:
                sprite = pyglet.sprite.Sprite(self.next_image)
                sprite.x = self.next_x
                sprite.y = self.next_y
                sprite.z = self.next_z
                self.sprites.append(sprite)
                self.next_image = None
                self.next_x = None
                self.next_z = None

    def calc_next_image(self):
        available_zs = set(self.potential_locations.keys())
        for csprite in self.sprites:
            if csprite.z not in self.potential_locations:
                continue
            cmax_y = self.potential_locations[csprite.z]['max_y']
            if csprite.y <= -cmax_y - cmax_y * self.prime / self.spin_frequency and csprite.z in available_zs:
                available_zs.remove(csprite.z)
        if not available_zs:
            self.next_z = None
            return
        self.next_z = random.choice(list(available_zs))
        max_y = self.potential_locations[self.next_z]['max_y']
        max_x = self.potential_locations[self.next_z]['max_x']
        if not self.next_image and not self.image_queue.empty():
            self.next_image = self.image_queue.get_nowait()
        self.next_y = int(-max_y - max_y * self.prime / self.spin_frequency - self.next_image.height)
        existing_intervals = self.get_existing_intervals(max_x)
        non_overlapping_intervals = self.get_non_overlapping_intervals(existing_intervals, max_x)
        if not non_overlapping_intervals:
            self.next_x = None
            return
        chosen_interval = random.choice(non_overlapping_intervals)
        self.next_x = random.randint(chosen_interval[0], chosen_interval[1])

    def get_existing_intervals(self, max_x):
        existing_intervals = []
        for csprite in self.sprites:
            if csprite.z not in self.potential_locations:
                continue
            cmax_y = self.potential_locations[csprite.z]['max_y']
            cmax_x = self.potential_locations[csprite.z]['max_x']
            if csprite.y <= -cmax_y - cmax_y * self.prime / self.spin_frequency:
                start = int((csprite.x + cmax_x) * max_x / cmax_x) - max_x
                end = int((csprite.x + csprite.width + cmax_x) * max_x / cmax_x) - max_x
                existing_intervals.append((start, end))
        existing_intervals.sort()
        return existing_intervals

    def get_non_overlapping_intervals(self, existing_intervals, max_x):
        non_overlapping_intervals = []
        current_start = -max_x
        for start, end in existing_intervals:
            if current_start + self.next_image.width < start:
                non_overlapping_intervals.append((current_start, start - self.next_image.width))
            current_start = max(current_start, end)
        if current_start + self.next_image.width < max_x:
            non_overlapping_intervals.append((current_start, max_x - self.next_image.width))
        return non_overlapping_intervals

    def get_potential_locations(self):
        potential_locations = {}
        potential_zs = np.linspace(self.z_far, self.max_z, self.number_image_variations)
        for potential_z in potential_zs:
            max_y = (self.initial_position[2] - potential_z) * self.y_scaler
            max_x = int(self._window.aspect_ratio * max_y)
            max_y = int(max_y)
            potential_locations[int(potential_z)] = {'max_x': max_x, 'max_y': max_y}
        return potential_locations

    def queue_image(self):
        ctypes.windll.ole32.CoInitialize(None)
        while True:
            if not hasattr(self, 'image_paths') or not self.image_paths:
                self.image_paths = get_image_paths(self.images_path)
                random.shuffle(self.image_paths)
                if not self.image_paths:
                    logger.error(f'No images found in: {self.images_path}')
                    os._exit(1)

            if not self.image_queue.full():
                image_path = self.image_paths.pop()
                try:
                    img = Image.open(image_path)
                    img = ImageOps.exif_transpose(img)
                except (IOError, FileNotFoundError):
                    logger.warning(f"Skipping {image_path}...")
                    continue

                width, height = self._get_scaled_dimensions(img)
                img = img.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")

                combined = self._add_blurred_shadow(img)
                combined = combined.transpose(Image.FLIP_TOP_BOTTOM)

                raw_data = combined.tobytes()
                img_data = pyglet.image.ImageData(combined.width, combined.height, 'RGBA', raw_data)
                self.image_queue.put(img_data)
            else:
                time.sleep(0.5)

    def _get_scaled_dimensions(self, img):
        if img.height > img.width:
            height = int(self._window.width * self.image_scale)
            width = int(math.ceil(img.width * height / img.height))
        else:
            width = int(self._window.width * self.image_scale)
            height = int(math.ceil(img.height * width / img.width))
        if img.height < height or img.width < width:
            width = img.width
            height = img.height
        return width, height

    def _add_blurred_shadow(self, img, padding_size=2):
        shadow = Image.new('RGBA', (img.width + 2 * padding_size, img.height + 2 * padding_size), (0, 0, 0, 0))
        shadow.paste(img, (padding_size, padding_size))
        shadow = shadow.filter(ImageFilter.GaussianBlur(padding_size))

        combined = Image.new('RGBA', shadow.size)
        combined.paste(shadow, (0, 0))
        combined.paste(img, (padding_size, padding_size), img)

        return combined


class FloatingPicturesConfig:
    def __init__(self):
        self.window = None
        self.image_paths = None
        self.config_options = [
            {'name': 'images_path',
            'description': f'Path to images to recursively get pictures (default {os.getcwd()}):',
            'default': os.getcwd(),
            'input_type': ['button', self.update_images_path],
            'type': str},
            {'name': 'spin_speed',
            'description': f'How fast the spin animation spins (default 2):',
            'default': 2,
            'type': int},
            {'name': 'image_frequency',
            'description': f'How often to spawn a picture in seconds (default 1):',
            'default': 1,
            'type': int},
            {'name': 'spin_frequency',
            'description': f'How often in seconds the spin animation occurs (default 60):',
            'default': 60,
            'type': int},
            {'name': 'number_image_variations',
            'description': f'Number of image variations (default 10):',
            'default': 10,
            'type': int},
            {'name': 'image_scale',
            'description': f'Scaling image size as a fraction of screen width (default 0.4):',
            'default': 0.4,
            'type': float},
        ]
        self.config_file = self.get_config()

    def get_config(self):
        config_file = configparser.ConfigParser()
        config_file.read(f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/3dfloatingpictures.ini')
        for config_option in self.config_options:
            if config_option['name'] not in config_file['DEFAULT']:
                config_file.set('DEFAULT', config_option['name'], str(config_option['default']))
            config_option['value'] = config_option['type'](config_file.get('DEFAULT', config_option['name']))
        with open(f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/3dfloatingpictures.ini', 'w+', encoding='utf-8') as fh:
            config_file.write(fh)
        return config_file

    def update_config(self):
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
                error_window = self.show_error(error_window, config_option, row)
                row += 1
        if error_window is not None:
            self.show_error_window(error_window, row)
        else:
            self.save_config()
            self.window.destroy()

    def show_error(self, error_window, config_option, row):
        if error_window is None:
            error_window = tk.Tk()
            error_window.title("Floating Pictures Settings Error")
            error_window.eval('tk::PlaceWindow . center')
            ttk.Style(error_window).theme_use()
        config_option['info'] = tk.Label(error_window, text=f'{config_option["name"]} must be of type {config_option["type"].__name__}.')
        config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
        return error_window

    def show_error_window(self, error_window, row):
        button_ok = ttk.Button(error_window, text="OK", command=error_window.destroy)
        button_ok.grid(row=row, column=0, sticky="esn", padx=5, pady=5)
        error_window.mainloop()

    def save_config(self):
        with open(f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/3dfloatingpictures.ini', 'w+', encoding='utf-8') as fh:
            self.config_file.write(fh)

    def update_images_path(self):
        for config_option in self.config_options:
            if config_option['name'] == 'images_path':
                config_option['var'].set(askdirectory())
                break

    def menu(self):
        self.window = tk.Tk()
        self.window.title("Floating Pictures Settings")
        ttk.Style(self.window).theme_use()
        row = 0
        for config_option in self.config_options:
            config_option['info'] = tk.Label(self.window, text=config_option['description'])
            config_option['var'] = tk.StringVar(self.window, config_option['value'])
            if 'input_type' in config_option and 'button' == config_option['input_type'][0]:
                config_option['input'] = ttk.Button(self.window, textvariable=config_option['var'], command=config_option['input_type'][1])
                config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
                config_option['input'].grid(row=row, column=1, sticky="esn", padx=2, pady=2)
            else:
                config_option['input'] = ttk.Entry(self.window, textvariable=config_option['var'])
                config_option['info'].grid(row=row, column=0, sticky='wsn', padx=2, pady=2)
                config_option['input'].grid(row=row, column=1, sticky="esn", padx=2, pady=2)
            row += 1
        self.add_menu_buttons(row)

    def add_menu_buttons(self, row):
        button_save = ttk.Button(self.window, text="Save", command=self.update_config)
        button_cancel = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        button_save.grid(row=row, column=0, sticky="esn", padx=5, pady=5)
        button_cancel.grid(row=row, column=1, sticky="esn", padx=5, pady=5)
        self.window.mainloop()

def usage():
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
    fpc = FloatingPicturesConfig()
    if len(sys.argv) == 2:
        arg = sys.argv[1].lower().split(':')[0].split(' ')[0]
    else:
        usage()
        sys.exit()
    if arg in ['-s', '/s']:
        run_screensaver(fullscreen=True, config=fpc.config_options)
    elif arg in ['-c', '/c']:
        fpc.menu()
        sys.exit()
    elif arg in ['-d', '/d']:
        run_screensaver(fullscreen=False, config=fpc.config_options)
    elif arg in ['-h', '/h']:
        usage()
        sys.exit()
    else:
        sys.exit()

def run_screensaver(fullscreen, config):
    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    create_window(screens[0], fullscreen, config)


def create_window(screen, fullscreen, config):
    window = pyglet.window.Window(fullscreen=fullscreen, resizable=not fullscreen, screen=screen)
    floating_pictures = FloatingPictures(window, debug=not fullscreen, **{conf['name']: conf['value'] for conf in config})
    try:
        pyglet.app.run()
    except AttributeError:
        pass


if __name__ == '__main__':
    main()
