import numpy as np

class AddressIndexList:
    def __init__(self, np_int_type = np.uint16, participants = None, external_address_list = None, id_to_label = None):
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
            self._id_to_idx = {
                p.id: i
                for i, p in enumerate(participants)
            }
            if id_to_label is None:
                id_to_label = {p.id: p.display_label() for p in participants if hasattr(p, "display_label")}
        else:
            self._address_to_idx = external_address_list
            n = len(external_address_list)

        self.np_int_type = np_int_type
        self._idx_to_id = {i: p for i, p in enumerate(self._address_to_idx)}
        self._id_to_label = id_to_label or {}
        self._list = np.zeros(n, dtype=np_int_type)


    def __getitem__(self, address_or_index):
        if isinstance(address_or_index, int):
            return self._list[address_or_index]

        return self._list[self._address_to_idx[address_or_index]]

    def __setitem__(self, giver_address_or_index, value):
        if isinstance(giver_address_or_index, int):
            self._list[giver_address_or_index] = value

        self._list[self._address_to_idx[giver_address_or_index]] = min(value, np.iinfo(self.np_int_type).max)

    def _label(self, i: int) -> str:
        user_id = self._idx_to_id[i]
        name = self._id_to_label.get(user_id)
        if name:
            return name
        return str(user_id)[-6:]

    def _full_label(self, i: int) -> str:
        user_id = self._idx_to_id[i]
        name = self._id_to_label.get(user_id)
        if name:
            return f"{name} ({user_id})"
        return str(user_id)

    def __str__(self):
        rows = [f"{self._full_label(i)}: {int(self._list[i]):>12,}" for i in range(len(self._idx_to_id))]
        return "\n".join(rows)
    
    def to_csv_cell(self):
        items = [
            f"({self._full_label(i)}, {int(self._list[i])})"
            for i in range(len(self._idx_to_id))
        ]

        return f"[{", ".join(items)}]"

    def get_as_normal_int(self, key=None):
        if key is None:
            return self._list.tolist()

        return self[key].item()

    def get_user_address(self, index: int):
        return self._list[index]