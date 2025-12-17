# src/jaxatari/games/jax_superman.py
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment
import jaxatari.spaces as spaces

# -------------------------
# CONSTANTS
# -------------------------
class SupermanConstants(NamedTuple):
    SCREEN_H: int = 210
    SCREEN_W: int = 160
    PLAYER_SPEED: int = 2
    SATELLITE_SPEED: int = 1
    LEX_SPEED: int = 1
    # asset manifest (filled when you extract sprites)
    ASSET_CONFIG: tuple = ()


# -------------------------
# STATE / OBS / INFO
# -------------------------
class SupermanState(NamedTuple):
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    lex_x: jnp.ndarray
    lex_y: jnp.ndarray
    satellites_x: jnp.ndarray  # shape (N,)
    satellites_y: jnp.ndarray  # shape (N,)
    score: jnp.ndarray
    done: jnp.ndarray
    step_counter: jnp.ndarray


class SupermanObs(NamedTuple):
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    lex_x: jnp.ndarray
    lex_y: jnp.ndarray
    satellites_x: jnp.ndarray
    satellites_y: jnp.ndarray


class SupermanInfo(NamedTuple):
    difficulty: jnp.ndarray


# -------------------------
# HELPER: map action -> (dx, dy, fire)
# -------------------------
# 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
# 6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
# 10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
# 14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE

def _action_to_motion(action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # returns (dx, dy, fire) as integers
    # base movement mapping for 8 directions
    dx = jnp.zeros_like(action, dtype=jnp.int32)
    dy = jnp.zeros_like(action, dtype=jnp.int32)
    fire = jnp.zeros_like(action, dtype=jnp.int32)

    # helper to set dx/dy/fire using lax.cond chain
    # Right
    dx = jax.lax.cond(action == 3, lambda _: jnp.ones_like(dx), lambda _: dx, None)
    dx = jax.lax.cond(action == 4, lambda _: -jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 2, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dy = jax.lax.cond(action == 5, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    # diagonals
    dx = jax.lax.cond(action == 6, lambda _: jnp.ones_like(dx), lambda _: dx, None)   # UPRIGHT (dx=+1,dy=-1)
    dy = jax.lax.cond(action == 6, lambda _: -jnp.ones_like(dy), lambda _: dy, None)

    dx = jax.lax.cond(action == 7, lambda _: -jnp.ones_like(dx), lambda _: dx, None)  # UPLEFT
    dy = jax.lax.cond(action == 7, lambda _: -jnp.ones_like(dy), lambda _: dy, None)

    dx = jax.lax.cond(action == 8, lambda _: jnp.ones_like(dx), lambda _: dx, None)   # DOWNRIGHT
    dy = jax.lax.cond(action == 8, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    dx = jax.lax.cond(action == 9, lambda _: -jnp.ones_like(dx), lambda _: dx, None)  # DOWNLEFT
    dy = jax.lax.cond(action == 9, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    # FIRE-only and direction+fire actions
    fire = jax.lax.cond((action == 1) | (action == 10) | (action == 11) | (action == 12) |
                        (action == 13) | (action == 14) | (action == 15) | (action == 16) |
                        (action == 17),
                        lambda _: jnp.ones_like(fire), lambda _: fire, None)

    # Direction for the FIRE variants
    dx = jax.lax.cond(action == 11, lambda _: jnp.ones_like(dx), lambda _: dx, None)  # RIGHTFIRE
    dx = jax.lax.cond(action == 12, lambda _: -jnp.ones_like(dx), lambda _: dx, None) # LEFTFIRE
    dy = jax.lax.cond(action == 10, lambda _: -jnp.ones_like(dy), lambda _: dy, None) # UPFIRE
    dy = jax.lax.cond(action == 13, lambda _: jnp.ones_like(dy), lambda _: dy, None)  # DOWNFIRE

    # diagonal fire
    dx = jax.lax.cond(action == 14, lambda _: jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 14, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 15, lambda _: -jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 15, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 16, lambda _: jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 16, lambda _: jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 17, lambda _: -jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 17, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    return dx, dy, fire




# -------------------------
# MAIN ENV CLASS
# -------------------------
class JaxSuperman(JaxEnvironment):
    
    def __init__(self, consts: SupermanConstants = None):
        consts = consts or SupermanConstants()
        super().__init__(consts)
        self.renderer = SupermanRenderer(self.consts)
        self.action_set = list(range(18))
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        play.py always calls env.render(), so Superman must implement it.
        We simply delegate to the attached renderer instance.
        """
        return self.renderer.render(state)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray):
        player_x = jnp.array(80, dtype=jnp.int32)
        player_y = jnp.array(150, dtype=jnp.int32)
        lex_x = jnp.array(80, dtype=jnp.int32)
        lex_y = jnp.array(30, dtype=jnp.int32)

        satellites_x = jnp.array([30, 90, 140], dtype=jnp.int32)
        satellites_y = jnp.array([40, 10, 70], dtype=jnp.int32)

        state = SupermanState(
            player_x, player_y, lex_x, lex_y,
            satellites_x, satellites_y,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(0, dtype=jnp.int32)
        )
        obs = self._get_observation(state)
        return obs, state

    
    def step(self, state: SupermanState, action: int):
        prev = state

        # action -> dx,dy,fire
        dx, dy, fire = _action_to_motion(jnp.array(action, dtype=jnp.int32))

        state = self._player_step(state, dx, dy, fire)
        state = self._lex_step(state)
        state = self._satellite_step(state)

        reward = self._get_reward(prev, state)
        done = self._get_done(state)

        state = state._replace(done=done, step_counter=state.step_counter + 1)

        obs = self._get_observation(state)
        info = self._get_info(state)

        return obs, state, reward, done, info

    # -------------------------
    # helper steps
    # -------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: SupermanState, dx: jnp.ndarray, dy: jnp.ndarray, fire: jnp.ndarray):
        new_x = state.player_x + dx * self.consts.PLAYER_SPEED
        new_y = state.player_y + dy * self.consts.PLAYER_SPEED

        # clamp within screen
        new_x = jnp.clip(new_x, 0, self.consts.SCREEN_W - 1)
        new_y = jnp.clip(new_y, 0, self.consts.SCREEN_H - 1)
        return state._replace(player_x=new_x, player_y=new_y)

    @partial(jax.jit, static_argnums=(0,))
    def _lex_step(self, state: SupermanState):
        # simple vertical patrol as placeholder
        new_y = (state.lex_y + self.consts.LEX_SPEED).astype(jnp.int32)
        # wrap / bounce
        new_y = jnp.where(new_y > self.consts.SCREEN_H - 10, 10, new_y)
        return state._replace(lex_y=new_y)

    @partial(jax.jit, static_argnums=(0,))
    def _satellite_step(self, state: SupermanState):
        # vectorized move for satellites (vmap pattern)
        new_y = state.satellites_y + self.consts.SATELLITE_SPEED
        # simple wrap
        new_y = jnp.where(new_y > self.consts.SCREEN_H - 4, 0, new_y)
        return state._replace(satellites_y=new_y)

    # -------------------------
    # observation / info / reward / done
    # -------------------------
    def _get_observation(self, state: SupermanState) -> SupermanObs:
        return SupermanObs(
            state.player_x, state.player_y,
            state.lex_x, state.lex_y,
            state.satellites_x, state.satellites_y
        )

    def _get_info(self, state: SupermanState) -> SupermanInfo:
        return SupermanInfo(jnp.array(0))

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: SupermanState, state: SupermanState) -> jnp.ndarray:
        # +10 if player reaches lex (very simple)
        reached = (jnp.abs(state.player_x - state.lex_x) < 8) & (jnp.abs(state.player_y - state.lex_y) < 8)
        reward = jnp.where(reached, 10, 0)
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SupermanState) -> jnp.ndarray:
        # done if player collides with any satellite (simple box collision)
        dx = jnp.abs(state.player_x - state.satellites_x)
        dy = jnp.abs(state.player_y - state.satellites_y)
        collision = jnp.any((dx < 8) & (dy < 8))
        return collision

    # -------------------------
    # spaces
    # -------------------------
    def action_space(self):
        return spaces.Discrete(18)

    def observation_space(self):
        return spaces.Dict({
            "player_x": spaces.Box(0, self.consts.SCREEN_W, (), jnp.int32),
            "player_y": spaces.Box(0, self.consts.SCREEN_H, (), jnp.int32),
            "lex_x": spaces.Box(0, self.consts.SCREEN_W, (), jnp.int32),
            "lex_y": spaces.Box(0, self.consts.SCREEN_H, (), jnp.int32),
            # satellites omitted from flat obs; add as needed
        })
class SupermanRenderer(JAXGameRenderer):
    """
    Minimal, pure-JAX renderer for Superman.
    Draws colored rectangles for player (blue), lex (green), satellites (red),
    on a gray background. Returns uint8 array shape (H, W, 3).
    """

    def __init__(self, consts: SupermanConstants = None):
        # store constants for sizes/colors
        self.consts = consts or SupermanConstants()
        # simple sprite sizes
        self.player_h, self.player_w = 8, 8
        self.lex_h, self.lex_w = 10, 10
        self.sat_h, self.sat_w = 4, 4

    def _empty_canvas(self):
        # create gray background, dtype uint8
        h, w = self.consts.SCREEN_H, self.consts.SCREEN_W
        return jnp.full((h, w, 3), 120, dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        state: SupermanState NamedTuple
        returns: jnp.ndarray uint8 (H, W, 3)
        """
        canvas = self._empty_canvas()

        # helper to draw a filled rectangle in-place (via at[].set)
        def draw_rect(img, x, y, hh, ww, color):
            # compute integer bounds
            x0 = jnp.clip(x - ww // 2, 0, self.consts.SCREEN_W - 1)
            x1 = jnp.clip(x0 + ww, 0, self.consts.SCREEN_W)
            y0 = jnp.clip(y - hh // 2, 0, self.consts.SCREEN_H - 1)
            y1 = jnp.clip(y0 + hh, 0, self.consts.SCREEN_H)

            # create mask area indices
            xs = jnp.arange(self.consts.SCREEN_W)
            ys = jnp.arange(self.consts.SCREEN_H)[:, None]
            mask_x = (xs >= x0) & (xs < x1)
            mask_y = (ys >= y0) & (ys < y1)
            mask = (mask_x[None, :] & mask_y[:, None, 0])  # shape (H, W)

            # expand mask for 3 channels and set color
            color_arr = jnp.array(color, dtype=jnp.uint8)
            mask3 = mask[:, :, None]
            filled = jnp.where(mask3, color_arr, img)
            return filled

        # draw player (blue)
        canvas = draw_rect(canvas, state.player_x, state.player_y, self.player_h, self.player_w, (30, 60, 200))

        # draw lex (green)
        canvas = draw_rect(canvas, state.lex_x, state.lex_y, self.lex_h, self.lex_w, (30, 200, 60))

        # draw satellites (red) - vectorized by looping with a fori_loop
        def draw_sat_loop(i, img):
            x = state.satellites_x[i]
            y = state.satellites_y[i]
            return draw_rect(img, x, y, self.sat_h, self.sat_w, (200, 30, 30))

        num_sat = state.satellites_x.shape[0]
        canvas = jax.lax.fori_loop(0, num_sat, draw_sat_loop, canvas)

        return canvas