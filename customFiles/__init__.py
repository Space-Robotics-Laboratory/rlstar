# Registers new environments with gym so that baselines can find them
# Should register all new environments and file paths in the two init files
from gym.envs.registration import register

register(
    id='clover-v1',
    entry_point='customFiles.environments:clover',
)  # car on sandy surface

