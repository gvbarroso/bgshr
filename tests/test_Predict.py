
import bgshr 
import numpy as np
import pandas
import unittest
import os


class TestLoadExtendedTable(unittest.TestCase):

    def test_load_extended_table(self):
        file = os.path.join(
            os.path.dirname(__file__), "test_files/lookup_tbl.csv.gz")
        df = bgshr.Predict.load_extended_lookup_table(file, num_s_vals=1)[0]
        self.assertTrue(len(np.unique(df["s"])) == 31)


class TestTableExpNeuPi0(unittest.TestCase):

    def test_table_expected_neu_pi0(self):
        file = os.path.join(
            os.path.dirname(__file__), "test_files/lookup_tbl.csv.gz")
        neu_mut = np.array([1e-7, 1e-8, 1e-9])
        expected = 4 * 10000 * neu_mut 
        df = pandas.read_csv(file)
        result = bgshr.Predict._table_expected_neu_pi0(df, neu_mut)
        self.assertTrue(np.all(np.isclose(result, expected)))

class TestExpDelPi0(unittest.TestCase):

    def setUp(self):
        self.s_vals = -np.append(np.logspace(1, -6, 30), 0)

    def test_fictitious_neutral_dfe(self):
        dfe = {"type": "gamma_neutral", "shape": 0.2, "scale": 0.02, "p_neu": 1}
        Ne = 20000
        del_sites = [np.array([1000, 500])]
        u_arrs = [np.array([1e-8, 2e-8])]
        result = bgshr.Predict._expected_del_pi0(
            [dfe], self.s_vals, del_sites, u_arrs, Ne=Ne, uL0=1e-8)
        expected = 4 * Ne * u_arrs[0]
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_bounds(self):
        dfe = {"type": "gamma", "shape": 0.1, "scale": 0.02}
        Ne = 10000
        del_sites = [np.array([500, 500])]
        u_arrs = [np.array([1e-8, 2e-8])]
        result = bgshr.Predict._expected_del_pi0(
            [dfe], self.s_vals, del_sites, u_arrs, Ne=Ne)
        neutral_exp = 4 * Ne * u_arrs[0]
        self.assertTrue(np.all(result < neutral_exp))


class TestTableExpDelPi0(unittest.TestCase):

    def setUp(self):
        file = os.path.join(
            os.path.dirname(__file__), "test_files/lookup_tbl.csv.gz")
        self.df = pandas.read_csv(file)

    def test_fictitious_neutral_dfe(self):
        dfe = {"type": "gamma_neutral", "shape": 0.2, "scale": 0.02, "p_neu": 1}
        del_sites = [np.array([1000, 500])]
        u_arrs = [np.array([1e-8, 2e-8])]
        result = bgshr.Predict._table_expected_del_pi0(
            self.df, [dfe], del_sites, u_arrs)
        expected = 4 * 10000 * u_arrs[0]
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_against_exp_del_pi0(self):
        dfe = {
            "type": "gamma_neutral", "shape": 0.2, "scale": 0.02, "p_neu": 0.3}
        del_sites = [np.array([1000, 500])]
        u_arrs = [np.array([1e-8, 2e-8])]
        result = bgshr.Predict._table_expected_del_pi0(
            self.df, [dfe], del_sites, u_arrs)
        s_vals = np.sort(np.unique(self.df["s"]))
        Ne = 10000
        Ne_result = bgshr.Predict._expected_del_pi0(
            [dfe], s_vals, del_sites, u_arrs, Ne=Ne, uL0=1e-8)
        self.assertTrue(np.all(np.isclose(result, Ne_result, atol=1e-6)))

    def test_bounds(self):
        dfe = {"type": "gamma", "shape": 0.1, "scale": 0.02}
        Ne = 10000
        del_sites = [np.array([500, 500])]
        u_arrs = [np.array([1e-8, 2e-8])]
        result = bgshr.Predict._table_expected_del_pi0(
            self.df, [dfe], del_sites, u_arrs)
        neutral_exp = 4 * Ne * u_arrs[0]
        self.assertTrue(np.all(result < neutral_exp))

    
