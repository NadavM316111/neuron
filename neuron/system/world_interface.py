"""The World interface: the clean boundary that frees NEURON's mind.

WHY THIS EXISTS. Until now NEURON's mind talked directly to one specific world
(weather, quakes, markets). That tied the brain to a sandbox. To scale the
mind (toward intelligence) and to let it live in richer worlds (toward real
participation), the mind must be WORLD-BLIND: it talks to an abstract World,
never to weather directly. Any world, weather, a grid, a game, a richer human
world, plugs in behind this one interface, and the mind neither knows nor
cares which. That separation is what makes the mind portable and scalable.

A World must provide four things, and nothing about the mind depends on how:

  signals()            the names of the things that can be perceived now.
  observe(name)        the current value of one signal, a number in [0,1].
  horizon_outcome(...) given a past observation, what actually happened later
                       (so the mind can be scored on its predictions).
  act(action, name)    optionally act on the world (spend to interpret, ask,
                       etc.); returns whatever the action yields, or None.

The mind calls these. It does not import weather, or markets, or any specific
world. Swap the World, and the same mind lives a different life.
"""

from abc import ABC, abstractmethod


class World(ABC):
    """The abstract world the mind perceives and acts in. Subclass this for
    any concrete world (weather, grid, game, richer worlds later)."""

    @abstractmethod
    def signals(self):
        """Return a list of signal names currently perceivable."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, name):
        """Return the current value of signal `name`, normalized to [0, 1].
        This is a single perception. The mind never parses raw world data;
        it only ever sees numbers through this method."""
        raise NotImplementedError

    def outcome(self, name, past_value):
        """Return the realized future direction of `name` versus `past_value`,
        as a class the mind predicts: 0 = down, 1 = same, 2 = up. Default
        implementation buckets the change; a world may override. Returns None
        if no outcome is available yet."""
        now = self.observe(name)
        if now is None or past_value is None:
            return None
        d = now - past_value
        if d < -0.02:
            return 0
        if d > 0.02:
            return 2
        return 1

    def act(self, action, name=None):
        """Optionally act on the world. Returns the result of the action, or
        None if the world supports no acting. The mind treats a None return as
        'this world cannot be acted on in that way' and carries on. Default:
        the world cannot be acted on."""
        return None

    def step(self):
        """Advance the world by one tick if it is time-based (fetch fresh data,
        move the simulation). Worlds that are static can leave this empty."""
        pass

    def describe(self):
        """A short human-readable name for this world, for logs. Optional."""
        return self.__class__.__name__
