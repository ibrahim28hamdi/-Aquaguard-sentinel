import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from aquaguard.loaders import synthetic_demo, parse_usgs_timeseries, load_custom, load_uci_local
from aquaguard.model import Sentinel, split_three, select_features, conformal_pvalues
from aquaguard.experiment import inject_spikes, inject_missingness, test_benchmark


class TestAquaGuardSentinel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=synthetic_demo(450,seed=33)
        cls.train,cls.cal,cls.test,cls.order=split_three(cls.df)
        cls.detector=Sentinel().fit(cls.train,cls.cal)

    def test_three_way_ordered_split(self):
        self.assertEqual((len(self.train),len(self.cal),len(self.test)),(270,90,90))
        self.assertEqual(self.order,'observed_at')
        self.assertLess(self.train.observed_at.max(),self.cal.observed_at.min())
        self.assertLess(self.cal.observed_at.max(),self.test.observed_at.min())

    def test_only_training_selects_features(self):
        train=self.train.copy()
        cal=self.cal.copy()
        train['leakage_column']=np.nan
        cal['leakage_column']=np.arange(len(cal))
        self.assertNotIn('leakage_column',select_features(train))

    def test_per_station_models_present(self):
        self.assertGreaterEqual(len(self.detector.local),2)
        self.assertGreaterEqual(len(self.detector.site_cal),2)

    def test_conformal_rank_formula(self):
        p=conformal_pvalues(np.array([0.,5.,10.]),np.array([1.]*19+[5.]))
        self.assertTrue(np.allclose(p,[1.,2/21,1/21]))

    def test_scores_range_and_abstention(self):
        for method in ('robust_global','isolation_forest','sentinel_station'):
            pred=self.detector.alerts(self.test,method)
            finite=pred.p_value.dropna()
            self.assertTrue((finite>=0).all())
            self.assertTrue((finite<=1).all())
        test=self.test.copy()
        test.loc[test.index[:1],self.detector.features]=np.nan
        pred=self.detector.alerts(test,'sentinel_station')
        self.assertTrue(bool(pred.sensor_data_insufficient.iloc[0]))
        self.assertTrue(np.isnan(pred.p_value.iloc[0]))

    def test_spikes_do_not_modify_original(self):
        before=self.test.copy(deep=True)
        modified,rows=inject_spikes(self.test,self.detector,seed=5,strength=4.)
        pd.testing.assert_frame_equal(self.test,before)
        self.assertEqual(len(rows),18)
        self.assertGreater(np.nansum(abs(modified[self.detector.features]-before[self.detector.features]).to_numpy()),0)

    def test_sensor_dropout_separate_from_contamination(self):
        modified,rows=inject_missingness(self.test,self.detector,seed=2)
        pred=self.detector.alerts(modified,'sentinel_station')
        self.assertTrue(pred.sensor_data_insufficient.iloc[rows].any())

    def test_usgs_payload_parser(self):
        payload={'value':{'timeSeries':[
            {'variable':{'variableCode':[{'value':'00010'}]},'sourceInfo':{'siteCode':[{'value':'site1'}]},
             'values':[{'value':[{'dateTime':'2025-06-01T12:00:00Z','value':'15.2'}]}]},
            {'variable':{'variableCode':[{'value':'00400'}]},'sourceInfo':{'siteCode':[{'value':'site1'}]},
             'values':[{'value':[{'dateTime':'2025-06-01T12:00:00Z','value':'7.1'}]}]},
        ]}}
        parsed=parse_usgs_timeseries(payload)
        self.assertEqual(len(parsed),1)
        self.assertAlmostEqual(parsed.ph.iloc[0],7.1)
        self.assertAlmostEqual(parsed.temperature_c.iloc[0],15.2)

    def test_custom_csv_adapter(self):
        inp=pd.DataFrame({'place':['a','b'],'when':['2025-01-01','2025-01-02'],'read':[1,2]})
        frame=load_custom(inp,water_type='river',station_col='place',time_col='when')
        self.assertEqual(frame.station.iloc[0],'a')
        self.assertEqual(frame.water_type.iloc[0],'river')

    def test_local_uci_raw_file_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath=Path(tmp)/'water-treatment.data'
            # Official format has 1 sample date and 38 numeric fields.
            filepath.write_text('1-1-93,'+','.join(str(i) for i in range(38))+'\n', encoding='utf-8')
            frame=load_uci_local(filepath)
            self.assertEqual(len(frame),1)
            self.assertEqual(len(frame.columns),41)
            self.assertEqual(frame['water_type'].iloc[0],'wastewater')
            self.assertEqual(frame['Q-E'].iloc[0],0)

    def test_integer_measurements_spikes_are_float_safe(self):
        test=self.test.copy()
        for feature in self.detector.features:
            test[feature]=test[feature].fillna(0).round().astype('int64')
        altered,rows=inject_spikes(test,self.detector,seed=7,strength=3)
        self.assertTrue(any(altered[f].dtype.kind=='f' for f in self.detector.features))
        self.assertEqual(len(rows),18)

    def test_end_to_end_benchmark(self):
        summary,trials,manifest=test_benchmark(self.df,dataset='synthetic_unit',seeds=(1,2),strengths=(3.,))
        self.assertEqual(manifest['holdout'],90)
        self.assertEqual(manifest['verified_contamination_labels'],False)
        self.assertEqual(set(summary.method),{'robust_global','isolation_forest','sentinel_station'})
        self.assertTrue((summary.query("stress == 'artificial_spike'").mean_alert_rate<=1).all())
        self.assertTrue((summary.query("stress == 'artificial_spike'").mean_alert_rate>=0).all())
        self.assertGreater(len(trials),0)

if __name__=='__main__': unittest.main()
