import numpy as np

class AddressIndexMatrix:
    def __init__(self, np_int_type = np.uint8, participants = None, external_address_list = None):
        if external_address_list is None and participants is None:
            raise TypeError('both participants and externalAddressList cannot be None')
        if external_address_list is not None and participants is not None:
            raise TypeError('both participants and externalAddressList cannot be defined')

        if external_address_list is None:
            n = len(participants)

            self._address_to_idx = {
                p.address: i
                for i, p in enumerate(participants)
            }
        else:
            self._address_to_idx = external_address_list
            n = len(external_address_list)
        self.np_int_type = np_int_type
        self._idx_to_address = {i: p for i, p in enumerate(self._address_to_idx)}
        self._matrix = np.zeros((n, n), dtype=self.np_int_type)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            if isinstance(key, int):
                return self._matrix[key]

            return self._matrix[self._address_to_idx[key]]

        giver_address_or_index, receiver_address_or_index = key
        if not isinstance(giver_address_or_index, type(receiver_address_or_index)) :
            raise TypeError('giver_address_or_index & receiver_address_or_index must be same type')
        if isinstance(giver_address_or_index, int):
            return self._matrix[giver_address_or_index][receiver_address_or_index]

        return self._matrix[self._address_to_idx[giver_address_or_index]][self._address_to_idx[receiver_address_or_index]]

    def __setitem__(self, key, value):
        giver_address_or_index, receiver_address_or_index = key
        if not isinstance(giver_address_or_index, type(receiver_address_or_index)) :
            raise TypeError('giver_address_or_index & receiver_address_or_index must be same type')

        if isinstance(giver_address_or_index, int):
            self._matrix[giver_address_or_index][receiver_address_or_index] = value
        self._matrix[self._address_to_idx[giver_address_or_index]][self._address_to_idx[receiver_address_or_index]] = min(value, np.iinfo(self.np_int_type).max)

    def get_user_address(self, index: int):
        return self._idx_to_address[index]