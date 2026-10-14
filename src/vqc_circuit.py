import pennylane as qml
import torch
import numpy as np

def check_backend(device_name: str) -> str:
    try:
        qml.device(device_name, wires=1)
        return device_name
    except Exception:
        return 'default.qubit'

def _get_diff_method(device_name: str) -> str:
    return 'backprop'

def create_vqc_node(n_qubits=4, n_layers=2, encoding='angle',
                    entanglement='linear', measurement_mode='expectation',
                    device_name='default.qubit'):
    device_name = check_backend(device_name)
    diff_method = _get_diff_method(device_name)

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if encoding == 'angle':
            for i in range(n_qubits):
                if i % 2 == 0:
                    qml.RX(inputs[..., 0] * np.pi, wires=i)
                else:
                    qml.RY(inputs[..., 1] * np.pi, wires=i)
        elif encoding == 'amplitude':
            norm_factor = torch.sqrt(
                inputs[..., 0] ** 2 + inputs[..., 1] ** 2 + 1e-8)
            for i in range(n_qubits):
                qml.RY(np.pi * inputs[..., 0] / norm_factor, wires=i)
                qml.RZ(np.pi * inputs[..., 1] / norm_factor, wires=i)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            if entanglement in ('none', 'zero'):
                pass
            elif entanglement == 'linear':
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            elif entanglement == 'circular':
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            else:
                raise ValueError(f"Unknown entanglement topology: {entanglement}")

        if measurement_mode == 'expectation':
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        elif measurement_mode == 'probability':
            return qml.probs(wires=range(n_qubits))
        elif measurement_mode == 'state':
            return qml.state()
        else:
            raise ValueError(f"Unknown measurement_mode: {measurement_mode}")

    return circuit

def get_weight_shape(n_qubits=4, n_layers=2):
    return {"weights": (n_layers, n_qubits, 3)}
