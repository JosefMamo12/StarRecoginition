class StarPoint:
    def __init__(self, _id, _x, _y, _radius, _brightness):
        self._star_id = _id
        self._x = _x
        self._y = _y
        self._radius = _radius
        self._brightness = _brightness

    def __str__(self):
        return f"StarPoint: id: {self._star_id} x: {self._x} y: {self._y} r: {self._radius} b: {self._brightness}"

    def calc_brightness(self):
        return None

