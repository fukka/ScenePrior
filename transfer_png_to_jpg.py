import os
import glob
from PIL import Image


def convert_png_to_jpg(input_path, output_path, quality=85):
    """
    Converts a PNG image to JPG format and saves it.

    Args:
        input_path (str): The path to the input PNG file.
        output_path (str): The path to save the output JPG file.
        quality (int, optional): The quality of the JPG image (0-100).
                                 Higher values mean better quality and larger file size.
                                 Defaults to 85.
    """
    try:
        img = Image.open(input_path)

        # Convert RGBA (with transparency) to RGB if present, as JPG does not support alpha channel.
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img.save(output_path, 'JPEG', quality=quality)
        print(f"Successfully converted '{input_path}' to '{output_path}'")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    input_dir = r'/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_3_teacher_vtt_checkpoint_0424_slide_outs1x_v'
    all_images = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    for image in all_images:
        convert_png_to_jpg(image, image.replace('.png', '.jpg'))
    # convert_png_to_jpg('image_with_transparency.png', 'converted_image.jpg', quality=75)
