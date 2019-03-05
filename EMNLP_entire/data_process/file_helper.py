import re


def get_file_index(filename):
    start_index = re.search('alignment/', filename).span()
    end_index = re.search('.txt', filename).span()

    return filename[start_index[1]:end_index[0]]

def get_file_index_stft(filename):
    start_index = re.search('/', filename).span()
    end_index = re.search('.wav', filename).span()
    return filename[start_index[1]:end_index[0]]


def get_file_index_test(filename):
    start_index = re.search('alignment-test/', filename).span()
    end_index = re.search('.txt', filename).span()

    return filename[start_index[1]:end_index[0]]
