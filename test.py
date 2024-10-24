import ultralytics
import importlib
import torch
from ultralytics.nn.modules.block import C2f


class QuantC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)  # Call the parent class constructor
        # Custom initialization (example: replacing nn with a simple Conv2d layer)
        self.nn = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.nn(x)  # Example of a forward pass through your new layer


# Dynamically get and print the C2f class
a = getattr(importlib.import_module(C2f.__module__), "C2f")
print(a)

# Replace C2f in the ultralytics module with your custom QuantC2f
setattr(importlib.import_module(C2f.__module__), "C2f", QuantC2f)

# Test the replacement by creating an instance of the modified C2f
test_m = ultralytics.nn.modules.block.C2f(1, 2)

print(test_m)
