# -*- coding: utf-8 -*-
# @Time     : 2020/9/26 10:58
# @Author   : Michael_Zhouy

from math import pi

def a(vr, vci, Pr=2000):
    """
    计算a值
    @param vr: 额定风速
    @param vci: 切入风速
    @param Pr: 额定功率
    @return:
    """
    return Pr / (vr ** 3 - vci ** 3)


def b(vr, vci):
    """
    计算b值
    @param vr: 额定风速
    @param vci: 切入风速
    @return:
    """
    return vci ** 3 / (vr ** 3 - vci ** 3)


def low_limit(v, vci, vr, vco, D, Pr=2000, rate=0.05):
    """
    计算下限
    @param v: 风速
    @param vci: 切入风速
    @param vr: 额定风度
    @param vco: 切出风速
    @param D: 风轮直径
    @param Pr: 额定功率
    @return:
    """
    S = pi * (D / 2) ** 2 / 1000
    if v < vci or v > vco:
        return 0
    elif (v >= vci) and (v < vr):
        return rate * S * (a(vr, vci, Pr) * v ** 3 - b(vr, vci) * Pr)
    elif (v > vr) and (v < vco):
        return rate * S * Pr


def high_limit(v, D, max_, Cp=0.593):
    """
    计算上限
    @param v: 风速
    @param D: 风轮直径
    @param max_: 功率的上限
    @param Cp: 风能利用系数
    @return:
    """
    res = 0.5 * 1.29 / 1000 * pi * Cp * (D / 2) ** 2 * v ** 3
    if res > max_:
        return max_
    else:
        return res
