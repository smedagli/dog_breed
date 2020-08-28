"""
Will print whether or not there is a dog in the input image file.
"""
import argparse
from dog_breed.detectors import detectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Will print whether or not there is a dog in the input image file.')

    parser.add_argument('-f', '--file', help='Path of the image', required=True)
    args = parser.parse_args()
    dog = detectors.is_dog(args.file)
    if dog:
        print("The image represents a dog")
    else:
        print("No dog detected in the image")
