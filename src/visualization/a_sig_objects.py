class Attribute(object):
    """Super-class for all attributes. 
    """
    def __init__(self, name):
        self.name = name
    
    def get_value(self):
        pass
    
    def set_value(self):
        pass
    
    def __str__(self):
        return self.level + "." + self.name

class Number(Attribute):
    def __init__(self, name):
        super(Number, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Type(Attribute):
    def __init__(self, name):
        super(Type, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Size(Attribute):
    def __init__(self, name):
        super(Size, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Color(Attribute):
    def __init__(self, name):
        super(Color, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Angle(Attribute):
    def __init__(self, name):
        super(Angle, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Uniformity(Attribute):
    def __init__(self, name):
        super(Uniformity, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name

class Position(Attribute):
    def __init__(self, name):
        super(Position, self).__init__(name)

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.level + "." + self.name