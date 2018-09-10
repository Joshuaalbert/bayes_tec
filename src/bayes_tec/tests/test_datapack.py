from ..datapack import DataPack
import os

def test_datapack():
    datapack = DataPack('datapack_test_datapack.hdf5')
    with datapack:
        assert datapack._contexts_open == 1
        datapack.add_antennas()
        datapack.add_sources([[0,0]])
        datapack.add_freq_dep_tab('phase',times=[0,1],freqs=[0])
        assert datapack.phase[0].shape==(1,62,1,2)

        datapack.select(dir='patch_000',ant='RS*')
        # 14 RS*
        assert datapack.phase[0].shape==(1,14,1,2)

        datapack.switch_solset('sol_new',array_file=DataPack.lofar_array,directions=[[0,0],[1,1]])
        datapack.add_freq_dep_tab('phase',times=[0,1,2],freqs=[0,1])
        datapack.select_all()
        assert datapack.phase[0].shape==(2,62,2,3)
        datapack.select(dir='patch_000',ant='RS*')
        assert datapack.phase[0].shape==(1,14,2,3)
        datapack.select_all()
        assert datapack.phase[0].shape==(2,62,2,3)
        os.unlink('datapack_test_datapack.hdf5')

