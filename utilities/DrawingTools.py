from utilities.ColorBGR import Colors

class DrawingTools:
    # radius = 50
    # extension_len = 50  # 延长线的长度
    thickness = 3
    trajectory_color = Colors.RED.value  # 延长线的颜色
    angleMark_color = Colors.RED.value  # 弧线/角度的颜色
    coila_color = Colors.BLACK.value  # 关节点的颜色
    conLine_color = Colors.WHITE.value  # 关节之间连接线的颜色

    def __init__(self,radius,extension_len):
        self.radius = radius
        self.extension_len = extension_len


    def setTrajectoryColor(self,trajectoryColor):
        self.trajectory_color = trajectoryColor

    def getTrajectoryColor(self):
        return self.trajectory_color

    def setAngleMarkColor(self,angleMarkColor):
        self.angleMark_color = angleMarkColor

    def getAngleMarkColor(self):
        return self.angleMark_color

    def setCoilaColor(self,coilaColor):
        self.coila_color = coilaColor

    def getCoilaColor(self):
        return self.coila_color

    def setConLineColor(self,conLineColor):
        self.conLine_color = conLineColor

    def getConLineColor(self):
        return self.conLine_color

    def setRadius(self,radius):
        self.radius = radius

    def getRadius(self):
        return self.radius

    def setExtension(self,extension_len):
        self.extension_len = extension_len

    def getExtension(self):
        return self.extension_len