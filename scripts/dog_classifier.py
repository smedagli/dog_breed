import argparse

from dog_breed.models import dog_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will print the dog breed (or the breed to which the human in the picture resembles the most.)')

    parser.add_argument('-f', '--file', help='Path of the image', default= 'images/Curly-coated_retriever_03896.jpg')
    args = parser.parse_args()
    image_file = args.file

    text = dog_classifier.get_output_message(image_file)
    breed = dog_classifier.get_breed(image_file)
    if text == "Not a dog nor a person was found in the image":
        raise NotImplemented
    else:
        print('\n')
        print(text)
        print(breed)
