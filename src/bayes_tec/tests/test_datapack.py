from ..datapack import DataPack

def test_datapack():
    datapack = DataPack('test.hdf5')
    with datapack:
        assert datapack._contexts_open == 1
        datapack.add_antennas()
        datapack.add_sources([[0,0]])
        datapack.add_freq_dep_tab('phase',times=[0,1],freqs=[0])
        print(str(datapack.H))
        datapack.select(dir='patch_000',ant='RS*')
        print(datapack.phase)
