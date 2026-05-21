import torch
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wisdom.utils.detection_loader as detection_loader
from wisdom.utils.detection_loader import load_detection_model, normalize_detection_output


def test_normalize_detection_output_keeps_ultralytics_layout():
    preds = torch.randn(2, 84, 10)
    out = normalize_detection_output(preds, num_classes=80)
    assert out.shape == (2, 84, 10)
    assert torch.equal(out, preds)


def test_normalize_detection_output_converts_yolov5_layout():
    preds = torch.zeros(1, 3, 7)
    preds[0, :, :4] = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    preds[0, :, 4] = torch.tensor([0.5, 1.0, 0.2])
    preds[0, :, 5:] = torch.tensor([[0.4, 0.6], [0.1, 0.9], [0.3, 0.7]])

    out = normalize_detection_output(preds, num_classes=2)

    assert out.shape == (1, 6, 3)
    assert torch.allclose(out[0, :4, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(out[0, 4:, 0], torch.tensor([0.2, 0.3]))
    assert torch.allclose(out[0, 4:, 1], torch.tensor([0.1, 0.9]))
    assert torch.allclose(out[0, 4:, 2], torch.tensor([0.06, 0.14]))


def test_load_detection_model_falls_back_to_yolov5(monkeypatch):
    class DummyYOLO:
        def __init__(self, weights):
            raise TypeError(
                "ERROR /tmp/best.pt appears to be an Ultralytics YOLOv5 model originally trained "
                "with https://github.com/ultralytics/yolov5."
            )

    def fake_attempt_load(weights, device='cpu', fuse=False):
        model = torch.nn.Identity()
        model.names = {0: 'person', 1: 'box'}
        model.eval = lambda: model
        model.to = lambda *_args, **_kwargs: model
        return model

    monkeypatch.setattr('ultralytics.YOLO', DummyYOLO)
    monkeypatch.setattr(detection_loader, '_ensure_yolov5_path', lambda: Path('/tmp'))
    fake_models = types.ModuleType('models')
    fake_experimental = types.ModuleType('models.experimental')
    fake_experimental.attempt_load = fake_attempt_load
    monkeypatch.setitem(sys.modules, 'models', fake_models)
    monkeypatch.setitem(sys.modules, 'models.experimental', fake_experimental)

    bundle = load_detection_model('/tmp/best.pt', device='cpu')

    assert bundle.family == 'yolov5'
    assert bundle.num_classes == 2
