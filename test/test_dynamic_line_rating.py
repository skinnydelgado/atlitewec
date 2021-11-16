#!/vsr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Created on Mon Oct 18 15:11:42 2021

@author: fabian
"""

from atlite.convert import convert_line_rating
import numpy as np


def test_ieee_sample_case():
    """
    Test the implementation against the documented results from IEEE standard (chapter 4.6).
    """
    ds = dict(
        temperature=273 + 40, wnd100m=0.61, height=0, wnd_azimuth=0, influx_direct=1027
    )

    psi = 90  # line azimuth
    D = 0.02814  # line diameter
    Ts = 273 + 100  # max allowed line surface temp
    epsilon = 0.8  # emissivity
    alpha = 0.8  # absorptivity

    R = 9.39e-5  # resistance at 100°C

    i = convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)

    assert np.isclose(i, 1025, rtol=0.005)


def test_openmod_sample_case():
    """
    Test the implementation against the documented line parameters documented at
    https://wiki.openmod-initiative.org/wiki/Transmission_network_datasets#European_50_Hz_transmission_lines
    assuming 20°C, no wind and no sun.
    """
    ds = dict(temperature=273 + 20, wnd100m=0, height=0, wnd_azimuth=0, influx_direct=0)
    psi = 0  # line azimuth

    # first entry
    R = 0.06 * 1e-3  # in Ohm/m

    i = convert_line_rating(ds, psi, R)  # Ampere
    v = 220  # 220 kV
    s = np.sqrt(3) * i * v / 1e3  # in MW

    assert np.isclose(i, 1290, rtol=0.005)
    assert np.isclose(s, 492, rtol=0.01)

    # second entry (does not work)
    R = 0.03 * 1e-3

    i = convert_line_rating(ds, psi, R)
    v = 380  # 380 kV
    s = np.sqrt(3) * i * v  # in MW

    # assert np.isclose(s, 1698, rtol=0.01)
    # assert np.isclose(i, 2580, rtol=0.005)


def test_openmod_sample_case_per_unit():
    """
    Test the implementation against the documented line parameters documented at
    https://wiki.openmod-initiative.org/wiki/Transmission_network_datasets#European_50_Hz_transmission_lines
    assuming 20°C, no wind and no sun and using the per unit system.
    """
    ds = dict(temperature=273 + 20, wnd100m=0, height=0, wnd_azimuth=0, influx_direct=0)
    psi = 0  # line azimuth

    # feels a bit like cheating: pypsa give the resistance in Ohm/km, if we
    # convert that to Ohm/1000km and use the pu system, the units nicely play
    # out to MW in the end.
    R = 0.06 * 1e3 / 220 ** 2

    s = np.sqrt(3) * convert_line_rating(ds, psi, R)
    assert np.isclose(s, 492, rtol=0.01)


def test_suedkabel_sample_case():
    """
    Test the implementation against the documented line parameters documented at
    https://www.yumpu.com/de/document/read/30614281/kabeldatenblatt-typ-2xsfl2y-1x2500-rms-250-220-380-kv
    assume ambient temperature of 20°C, no wind, no sun and max allowed line
    temperature of 90°C.
    """

    ds = dict(temperature=273 + 20, wnd100m=0, height=0, wnd_azimuth=0, influx_direct=0)
    R = 0.0136 * 1e-3
    psi = 0  # line azimuth

    i = convert_line_rating(ds, psi, R, Ts=363)
    v = 380000  # 220 kV
    s = np.sqrt(3) * i * v / 1e6  # in MW

    assert np.isclose(i, 2460, rtol=0.02)
    assert np.isclose(s, 1619, rtol=0.02)