
class Filter:
    def __init__(self, selector, min_count=1):
        self._selector = selector

        count = 0
        class_idx = {}
        for i, (_, target) in enumerate(selector):
            if (target not in class_idx.keys()):
                class_idx[target] = []

            # Append index
            class_idx[target].append(i)
            count += 1

        # Allowed indices
        indices_allowed = list(range(0, count))

        # Remove classes not satisfying filter criterias
        for target, indices in class_idx.items():
            if (len(indices) < min_count):
                for idx in indices:
                    indices_allowed.pop(indices_allowed.index(idx))
        self.indices_allowed = indices_allowed

    def __iter__(self):
        for i, item in enumerate(self._selector):
            if (i in self.indices_allowed):
                yield item
