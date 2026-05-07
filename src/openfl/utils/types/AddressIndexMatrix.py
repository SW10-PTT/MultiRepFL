import numpy as np

class AddressIndexMatrix:
    def __init__(self, np_int_type = np.uint16, participants = None, external_address_list = None):
        if external_address_list is None and participants is None:
            raise TypeError('both participants and externalAddressList cannot be None')
        if external_address_list is not None and participants is not None:
            raise TypeError('both participants and externalAddressList cannot be defined')

        if external_address_list is None:
            n = len(participants)

            self._id_to_idx = {
                p.id: i
                for i, p in enumerate(participants)
            }
        else:
            self._id_to_idx = external_address_list
            n = len(external_address_list)
        self.np_int_type = np_int_type
        self._idx_to_id = {
                idx: user_id
                for user_id, idx in self._id_to_idx.items()
            }
        self._matrix = np.zeros((n, n), dtype=self.np_int_type)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            if isinstance(key, int):
                return self._matrix[key]

            return self._matrix[self._id_to_idx[key]]

        giver_id_or_index, receiver_id_or_index = key
        if not isinstance(giver_id_or_index, type(receiver_id_or_index)) :
            raise TypeError('giver_id_or_index & receiver_id_or_index must be same type')
        if isinstance(giver_id_or_index, int):
            return self._matrix[giver_id_or_index][receiver_id_or_index]

        return self._matrix[self._id_to_idx[giver_id_or_index]][self._id_to_idx[receiver_id_or_index]]

    def __setitem__(self, key, value):
        giver_id_or_index, receiver_id_or_index = key
        if isinstance(giver_id_or_index, int):
            self._matrix[giver_id_or_index][receiver_id_or_index] = min(value, np.iinfo(self.np_int_type).max)
        else:
            self._matrix[
                self._id_to_idx[giver_id_or_index]
            ][
                self._id_to_idx[receiver_id_or_index]
            ] = min(value, np.iinfo(self.np_int_type).max)

    def _label(self, i: int) -> str:
        return str(self._idx_to_id[i])[-6:]
    
    def _full_label(self, i: int) -> str:
        return str(self._idx_to_id[i])

    def __str__(self):
        n = len(self._idx_to_id)
        labels = [self._label(i) for i in range(n)]
        col_w = 10
        row_label_w = 8

        header  = " " * row_label_w + "  ".join(f"{lbl:>{col_w}}" for lbl in labels)
        divider = " " * row_label_w + "  ".join("-" * col_w for _ in labels)
        rows = [header, divider]

        for i, row_label in enumerate(labels):
            values = "  ".join(f"{int(v):>{col_w},}" for v in self._matrix[i])
            rows.append(f"{row_label:<{row_label_w}}{values}")

        return "\n".join(rows)
    
    def to_csv_cell(self):
        items = []

        for giver_idx in range(len(self._idx_to_id)):
            giver = self._full_label(giver_idx)

            for receiver_idx in range(len(self._idx_to_id)):
                value = int(self._matrix[giver_idx][receiver_idx])

                if value != 0:
                    receiver = self._full_label(receiver_idx)
                    items.append(f"({giver},{receiver},{value})")

        return f"[{", ".join(items)}]"

    def get_as_normal_int(self, key=None):
        if key is None:
            return self._matrix.astype(int)

        return self[key].astype(int)

    def get_user_address(self, index: int):
        return self._idx_to_id[index]