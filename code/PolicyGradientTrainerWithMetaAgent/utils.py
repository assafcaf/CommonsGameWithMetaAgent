import os


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    if not os.path.isdir(path):
        os.mkdir(path)
    frames[0].save(os.path.join(path, filename), save_all=True, append_images=frames, duration=15)
