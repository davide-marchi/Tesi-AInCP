import pandas as pd


def elaborate_magnitude(operation_type, magnitude_D, magnitude_ND):

    elaborated_magnitude = []

    if operation_type == 'concat':
        elaborated_magnitude = pd.concat([magnitude_D, magnitude_ND], ignore_index=True)
    elif operation_type == 'difference':
        elaborated_magnitude = magnitude_D - magnitude_ND
    elif operation_type == 'ai':
        elaborated_magnitude = (((magnitude_D - magnitude_ND) / (magnitude_D + magnitude_ND)) * 100).fillna(0)
    else: 
        print('operation type non supportata.')
        exit(1)


    return elaborated_magnitude