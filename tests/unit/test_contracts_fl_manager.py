from types import SimpleNamespace

from openfl.contracts.FLManager import FLManager


class _DummyJobListing:
    def __init__(self):
        self.contract = SimpleNamespace(address="0xJobListing")


def test_init_sets_core_fields():
    publisher = SimpleNamespace(address="0xPublisher")

    mgr = FLManager(publisher=publisher)

    assert mgr.publisher is publisher
    assert mgr.manual_setup is False
    assert mgr.latestBlock is None
    assert mgr.contract is None
    assert mgr.challenge_contract is None
    assert mgr.modelOf == {}
    assert mgr.job_listings == []
    assert mgr.gas_deploy == []
    assert mgr.txHashes == []
    assert mgr.job_template_address is None


def test_init_honours_manual_ganache_flag():
    mgr = FLManager(publisher=SimpleNamespace(), manual_ganache_setup=True)
    assert mgr.manual_setup is True


def test_get_model_of_queries_contract():
    class FakeCall:
        def __init__(self, result):
            self._result = result
            self.called_with = None

        def call(self, params):
            self.called_with = params
            return self._result

    captured = {}

    class FakeFunctions:
        def getModel(self, participant_address, addr):
            captured["args"] = (participant_address, addr)
            return FakeCall("model-result")

    fake_contract = SimpleNamespace(functions=FakeFunctions(), address="0xManager")

    mgr = FLManager(publisher=SimpleNamespace())
    mgr.contract = fake_contract

    participant = SimpleNamespace(address="0xParticipant")
    result = mgr.get_model_of(participant, "0xModelAddr")

    assert result == "model-result"
    assert captured["args"] == ("0xParticipant", "0xModelAddr")
