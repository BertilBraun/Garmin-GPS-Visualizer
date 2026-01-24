import unittest

from geo_utils import drop_spike_points


class TestDropSingleSpikePoint(unittest.TestCase):
    def test_drops_single_interior_spike(self):
        pts = [
            (0.0, 0.0),
            (0.0, 0.00010),
            (10.0, 10.0),  # spike
            (0.0, 0.00020),
            (0.0, 0.00030),
        ]
        out = drop_spike_points(pts)
        self.assertEqual(len(out), len(pts) - 1)
        self.assertNotIn((10.0, 10.0), out)

    def test_drops_multiple_spikes(self):
        pts = [
            (0.0, 0.0),
            (0.0, 0.00010),
            (10.0, 10.0),  # spike 1
            (0.0, 0.00020),
            (11.0, 11.0),  # spike 2
            (0.0, 0.00030),
            (0.0, 0.00040),
        ]
        out = drop_spike_points(pts)
        self.assertEqual(len(out), len(pts) - 2)
        self.assertNotIn((10.0, 10.0), out)
        self.assertNotIn((11.0, 11.0), out)

    def test_keeps_consistent_fast_movement(self):
        # roughly 111m per 0.001 deg lon at equator
        pts = [(0.0, i * 0.001) for i in range(10)]
        out = drop_spike_points(pts)
        self.assertEqual(out, pts)

    def test_handles_stationary_then_spike(self):
        pts = [
            (52.0, 13.0),
            (52.0, 13.0),
            (52.0, 13.0),
            (0.0, 0.0),  # spike
            (52.0, 13.00001),
            (52.0, 13.00002),
        ]
        out = drop_spike_points(pts)
        self.assertEqual(len(out), len(pts) - 1)
        self.assertNotIn((0.0, 0.0), out)

    def test_drops_first_point_spike(self):
        pts = [
            (10.0, 10.0),  # spike
            (0.0, 0.0),
            (0.0, 0.00005),
            (0.0, 0.00010),
            (0.0, 0.00015),
        ]
        out = drop_spike_points(pts)
        self.assertEqual(len(out), len(pts) - 1)
        self.assertNotEqual(out[0], (10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
