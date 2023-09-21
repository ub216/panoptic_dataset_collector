# File reused from Lightning AI ServeGradio to make it applicable
# for iterative output
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, List, Optional

import gradio
from lightning.app.core.work import LightningWork


class ServeGradioIterative(LightningWork, abc.ABC):
    inputs: Any
    outputs: Any
    examples: Optional[List] = None
    title: Optional[str] = None
    description: Optional[str] = None

    _start_method = "spawn"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert self.inputs
        assert self.outputs
        self._model = None

        self.ready = False

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any):
        """Override with your logic to make a prediction."""

    def run(self, *args: Any, **kwargs: Any):
        self.ready = True
        demo = gradio.Interface(
            fn=self.predict,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            title=self.title,
            description=self.description,
        )
        demo.queue()
        demo.launch(
            server_name=self.host,
            server_port=self.port,
        )

    def configure_layout(self) -> str:
        return self.url
