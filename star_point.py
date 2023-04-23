class StarPoint:
    def __init__(self, _id, _x, _y, _radius, _brightness):
        self._star_id = _id
        self._x = _x
        self._y = _y
        self._radius = _radius
        self._brightness = _brightness

    def __str__(self):
        return f"StarPoint: id: {self._star_id} x: {self._x} y: {self._y} r: {self._radius} b: {self._brightness}"

    def get_id(self) -> int:
        return self._star_id

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_radius(self):
        return self._radius

    def get_brightness(self):
        return self._brightness

    def __eq__(self, other):
        return isinstance(other, StarPoint) and other._star_id == self._star_id
