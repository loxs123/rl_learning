
import panda_gym
import gymnasium


def get_push_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPush-v3')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        return env
    return _init


def get_pick_and_place_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    def _init():
        env = gymnasium.make('PandaPickAndPlace-v3')
        env.unwrapped.sim.set_lateral_friction('object', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('object', -1, spinning_friction=spinning_friction)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        return env
    return _init