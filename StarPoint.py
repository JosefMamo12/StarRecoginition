class StarPoint:
    def __int__(self, _x, _y, _radius, _brightness):
        self._x = _x
        self._y = _y
        self._radius = _radius
        self._brightness = _brightness

    def __str__(self):
        return f"x: {self._x} y: {self._y} r: {self._radius} b: {self._brightness}"
