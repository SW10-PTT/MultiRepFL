import numpy as np

class AddressIndexList:
    def __init__(self, participants = None, external_address_list = None):
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

        self._idx_to_address = {i: p for i, p in enumerate(self._address_to_idx)}
        self._list = np.zeros(n, dtype=np.uint16)


    def __getitem__(self, address_or_index):
        if isinstance(address_or_index, int):
            return self._list[address_or_index]

        return self._list[self._address_to_idx[address_or_index]]

    def __setitem__(self, giver_address_or_index, value):
        if isinstance(giver_address_or_index, int):
            self._list[giver_address_or_index] = value

        self._list[self._address_to_idx[giver_address_or_index]] = min(value, np.iinfo(np.uint16).max)

    def get_user_address(self, index: int):
        return self._list[index]