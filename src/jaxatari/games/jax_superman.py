# src/jaxatari/games/jax_superman.py
"""
Superman Game - JAX Implementation

Game Mechanics:
- Player can be Superman (flying) or Clark Kent (walking)
- Must capture Lex Luthor and henchmen, put them in jail
- Must rebuild bridge with 3 pieces
- Must avoid Kryptonite satellites (3 of them)
- Can carry crooks, Lois Lane, and bridge pieces
- X-Ray Vision feature (FIRE + direction)
- Multi-screen city blocks connected at 4 sides
- Subway system with 4 colored areas
- Helicopter that can help or hinder
"""
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
    
    # Speeds
    SUPERMAN_FLY_SPEED: int = 2
    CLARK_WALK_SPEED: int = 1
    SATELLITE_SPEED: int = 1
    LEX_SPEED: int = 1
    HENCHMAN_SPEED: int = 1
    HELICOPTER_SPEED: int = 1
    
    # Game constants
    NUM_HENCHMEN: int = 5  # Lex + 5 henchmen = 6 total crooks
    NUM_SATELLITES: int = 3
    NUM_BRIDGE_PIECES: int = 3
    
    # Collision distances
    CAPTURE_DISTANCE: int = 8
    SATELLITE_HIT_DISTANCE: int = 6
    JAIL_DISTANCE: int = 10
    
    # Locations (simplified - using screen coordinates)
    PHONE_BOOTH_X: int = 20
    PHONE_BOOTH_Y: int = 180
    BRIDGE_LEFT_X: int = 60
    BRIDGE_RIGHT_X: int = 100
    BRIDGE_Y: int = 50
    JAIL_X: int = 10
    JAIL_Y: int = 100
    DAILY_PLANET_X: int = 150
    DAILY_PLANET_Y: int = 180
    
    # Player states
    PLAYER_SUPERMAN: int = 0
    PLAYER_CLARK: int = 1
    PLAYER_WEAKENED: int = 2
    
    # Multi-screen system
    NUM_FRAMES: int = 8  # 8 city block frames
    FRAME_WIDTH: int = 160  # Each frame is full screen width
    
    # Subway system
    SUBWAY_YELLOW: int = 0
    SUBWAY_BLUE: int = 1
    SUBWAY_GREEN: int = 2
    SUBWAY_PINK: int = 3
    SUBWAY_ENTRANCE_Y: int = 170  # Y position for subway entrances
    
    # Special frame locations
    PHONE_BOOTH_FRAME: int = 0  # Frame 0 has phone booth
    BRIDGE_FRAME: int = 2  # Frame 2 has bridge
    JAIL_FRAME: int = 1  # Frame 1 has jail
    DAILY_PLANET_FRAME: int = 7  # Frame 7 has Daily Planet
    
    # Asset manifest
    ASSET_CONFIG: tuple = ()


# -------------------------
# STATE / OBS / INFO
# -------------------------
class SupermanState(NamedTuple):
    # Player state
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    player_state: jnp.ndarray  # 0=Superman, 1=Clark, 2=Weakened
    player_altitude: jnp.ndarray  # 0=ground, >0=flying
    
    # Frame/Screen tracking (multi-screen system)
    current_frame: jnp.ndarray  # Current city block frame (0-7 for 8 frames)
    in_subway: jnp.ndarray  # bool - whether player is in subway system
    subway_area: jnp.ndarray  # 0-3: yellow, blue, green, pink subway areas
    
    # Carrying state (-1 = nothing, 0-5 = crook index, 6 = Lois, 7-9 = bridge piece)
    carrying: jnp.ndarray
    
    # Lex Luthor
    lex_x: jnp.ndarray
    lex_y: jnp.ndarray
    lex_captured: jnp.ndarray  # bool
    lex_in_jail: jnp.ndarray  # bool
    
    # Henchmen (5 total)
    henchmen_x: jnp.ndarray  # shape (5,)
    henchmen_y: jnp.ndarray  # shape (5,)
    henchmen_captured: jnp.ndarray  # shape (5,)
    henchmen_in_jail: jnp.ndarray  # shape (5,)
    
    # Kryptonite satellites (3 total)
    satellites_x: jnp.ndarray  # shape (3,)
    satellites_y: jnp.ndarray  # shape (3,)
    satellites_active: jnp.ndarray  # shape (3,) - bool
    
    # Lois Lane
    lois_x: jnp.ndarray
    lois_y: jnp.ndarray
    lois_in_daily_planet: jnp.ndarray  # bool
    
    # Bridge pieces (3 total)
    bridge_pieces_x: jnp.ndarray  # shape (3,)
    bridge_pieces_y: jnp.ndarray  # shape (3,)
    bridge_pieces_placed: jnp.ndarray  # shape (3,) - bool
    bridge_complete: jnp.ndarray  # bool
    
    # Helicopter
    helicopter_x: jnp.ndarray
    helicopter_y: jnp.ndarray
    helicopter_direction: jnp.ndarray  # -1 or 1
    
    # Game state
    score: jnp.ndarray
    timer: jnp.ndarray  # in frames
    game_started: jnp.ndarray  # bool - true when player first moves
    done: jnp.ndarray
    step_counter: jnp.ndarray
    
    # X-Ray Vision state
    xray_active: jnp.ndarray  # bool
    xray_direction: jnp.ndarray  # 0-3: up, right, down, left
    
    # Frame/Screen tracking (multi-screen system)
    current_frame: jnp.ndarray  # Current city block frame (0-7 for 8 frames)
    in_subway: jnp.ndarray  # bool - whether player is in subway system
    subway_area: jnp.ndarray  # 0-3: yellow, blue, green, pink subway areas
    
    # Entity frame locations (which frame each entity is in)
    lex_frame: jnp.ndarray
    henchmen_frames: jnp.ndarray  # shape (5,)
    lois_frame: jnp.ndarray
    bridge_pieces_frames: jnp.ndarray  # shape (3,)
    satellites_frames: jnp.ndarray  # shape (3,)
    helicopter_frame: jnp.ndarray
    
    # RNG key for random behaviors
    rng_key: jnp.ndarray


class SupermanObs(NamedTuple):
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    player_state: jnp.ndarray
    player_altitude: jnp.ndarray
    carrying: jnp.ndarray
    lex_x: jnp.ndarray
    lex_y: jnp.ndarray
    lex_captured: jnp.ndarray
    henchmen_x: jnp.ndarray
    henchmen_y: jnp.ndarray
    satellites_x: jnp.ndarray
    satellites_y: jnp.ndarray
    lois_x: jnp.ndarray
    lois_y: jnp.ndarray
    bridge_pieces_x: jnp.ndarray
    bridge_pieces_y: jnp.ndarray
    bridge_complete: jnp.ndarray
    crooks_in_jail: jnp.ndarray  # count


class SupermanInfo(NamedTuple):
    difficulty: jnp.ndarray
    timer_seconds: jnp.ndarray
    crooks_captured: jnp.ndarray


# -------------------------
# HELPER: map action -> (dx, dy, fire)
# -------------------------
# 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
# 6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
# 10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
# 14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE

def _action_to_motion(action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert action integer to (dx, dy, fire) tuple."""
    dx = jnp.zeros_like(action, dtype=jnp.int32)
    dy = jnp.zeros_like(action, dtype=jnp.int32)
    fire = jnp.zeros_like(action, dtype=jnp.int32)

    # Right
    dx = jax.lax.cond(action == 3, lambda _: jnp.ones_like(dx), lambda _: dx, None)
    dx = jax.lax.cond(action == 4, lambda _: -jnp.ones_like(dx), lambda _: dx, None)
    dy = jax.lax.cond(action == 2, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dy = jax.lax.cond(action == 5, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    # Diagonals
    dx = jax.lax.cond(action == 6, lambda _: jnp.ones_like(dx), lambda _: dx, None)   # UPRIGHT
    dy = jax.lax.cond(action == 6, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 7, lambda _: -jnp.ones_like(dx), lambda _: dx, None)  # UPLEFT
    dy = jax.lax.cond(action == 7, lambda _: -jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 8, lambda _: jnp.ones_like(dx), lambda _: dx, None)   # DOWNRIGHT
    dy = jax.lax.cond(action == 8, lambda _: jnp.ones_like(dy), lambda _: dy, None)
    dx = jax.lax.cond(action == 9, lambda _: -jnp.ones_like(dx), lambda _: dx, None)  # DOWNLEFT
    dy = jax.lax.cond(action == 9, lambda _: jnp.ones_like(dy), lambda _: dy, None)

    # FIRE button pressed
    fire = jax.lax.cond((action == 1) | (action == 10) | (action == 11) | (action == 12) |
                        (action == 13) | (action == 14) | (action == 15) | (action == 16) |
                        (action == 17),
                        lambda _: jnp.ones_like(fire), lambda _: fire, None)

    # Direction for FIRE variants
    dx = jax.lax.cond(action == 11, lambda _: jnp.ones_like(dx), lambda _: dx, None)  # RIGHTFIRE
    dx = jax.lax.cond(action == 12, lambda _: -jnp.ones_like(dx), lambda _: dx, None) # LEFTFIRE
    dy = jax.lax.cond(action == 10, lambda _: -jnp.ones_like(dy), lambda _: dy, None) # UPFIRE
    dy = jax.lax.cond(action == 13, lambda _: jnp.ones_like(dy), lambda _: dy, None)  # DOWNFIRE

    # Diagonal fire
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
        """Render the game state."""
        return self.renderer.render(state)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray):
        """Reset the game to initial state."""
        # Player starts as Clark Kent on the RIGHT SIDE of first screen
        player_x = jnp.array(self.consts.SCREEN_W - 20, dtype=jnp.int32)  # Right side of screen
        player_y = jnp.array(self.consts.SCREEN_H - 30, dtype=jnp.int32)  # Near ground level
        player_state = jnp.array(self.consts.PLAYER_CLARK, dtype=jnp.int32)  # Start as Clark Kent
        player_altitude = jnp.array(0, dtype=jnp.int32)  # On ground
        carrying = jnp.array(-1, dtype=jnp.int32)  # Not carrying anything
        
        # Lex Luthor - starts near bridge
        lex_x = jnp.array(self.consts.BRIDGE_LEFT_X + 20, dtype=jnp.int32)
        lex_y = jnp.array(self.consts.BRIDGE_Y, dtype=jnp.int32)
        lex_captured = jnp.array(False)
        lex_in_jail = jnp.array(False)
        
        # Henchmen - scattered around
        henchmen_x = jnp.array([40, 70, 100, 130, 50], dtype=jnp.int32)
        henchmen_y = jnp.array([80, 120, 90, 110, 140], dtype=jnp.int32)
        henchmen_captured = jnp.zeros(5, dtype=jnp.bool_)
        henchmen_in_jail = jnp.zeros(5, dtype=jnp.bool_)
        
        # Kryptonite satellites - start at top
        satellites_x = jnp.array([40, 80, 120], dtype=jnp.int32)
        satellites_y = jnp.array([20, 20, 20], dtype=jnp.int32)
        satellites_active = jnp.ones(3, dtype=jnp.bool_)
        
        # Lois Lane - starts near Daily Planet
        lois_x = jnp.array(self.consts.DAILY_PLANET_X, dtype=jnp.int32)
        lois_y = jnp.array(self.consts.DAILY_PLANET_Y - 20, dtype=jnp.int32)
        lois_in_daily_planet = jnp.array(False)
        
        # Bridge pieces - COLLAPSED and scattered to DIFFERENT FRAMES
        # Bridge collapses at start, pieces thrown to different pages/frames
        bridge_pieces_x = jnp.array([50, 80, 110], dtype=jnp.int32)  # Scattered positions
        bridge_pieces_y = jnp.array([100, 120, 90], dtype=jnp.int32)  # Different heights
        bridge_pieces_placed = jnp.zeros(3, dtype=jnp.bool_)  # Not placed yet
        bridge_complete = jnp.array(False)  # Bridge is collapsed
        
        # Helicopter - starts moving left to right
        helicopter_x = jnp.array(0, dtype=jnp.int32)
        helicopter_y = jnp.array(30, dtype=jnp.int32)
        helicopter_direction = jnp.array(1, dtype=jnp.int32)
        
        # Game state
        score = jnp.array(0, dtype=jnp.int32)
        timer = jnp.array(0, dtype=jnp.int32)
        game_started = jnp.array(False)  # Timer starts when player first moves
        done = jnp.array(False)
        step_counter = jnp.array(0, dtype=jnp.int32)
        
        # X-Ray Vision
        xray_active = jnp.array(False)
        xray_direction = jnp.array(0, dtype=jnp.int32)
        
        # Frame/Screen tracking
        current_frame = jnp.array(0, dtype=jnp.int32)  # Start at frame 0
        in_subway = jnp.array(False)
        subway_area = jnp.array(0, dtype=jnp.int32)
        
        # Entity frame locations (distribute across frames)
        lex_frame = jnp.array(2, dtype=jnp.int32)  # Bridge is in frame 2
        henchmen_frames = jnp.array([1, 3, 4, 5, 6], dtype=jnp.int32)  # Scatter across frames
        lois_frame = jnp.array(7, dtype=jnp.int32)  # Daily Planet is in frame 7
        bridge_pieces_frames = jnp.array([1, 3, 5], dtype=jnp.int32)  # Scatter bridge pieces
        satellites_frames = jnp.array([0, 2, 4], dtype=jnp.int32)  # Scatter satellites
        helicopter_frame = jnp.array(0, dtype=jnp.int32)

        state = SupermanState(
            player_x, player_y, player_state, player_altitude, carrying,
            current_frame, in_subway, subway_area,
            lex_x, lex_y, lex_captured, lex_in_jail,
            henchmen_x, henchmen_y, henchmen_captured, henchmen_in_jail,
            satellites_x, satellites_y, satellites_active,
            lois_x, lois_y, lois_in_daily_planet,
            bridge_pieces_x, bridge_pieces_y, bridge_pieces_placed, bridge_complete,
            helicopter_x, helicopter_y, helicopter_direction,
            score, timer, game_started, done, step_counter,
            xray_active, xray_direction,
            lex_frame, henchmen_frames, lois_frame, bridge_pieces_frames, satellites_frames, helicopter_frame,
            key
        )
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SupermanState, action: jnp.ndarray):
        """Take a game step."""
        prev = state

        # Parse action - ensure it's int32 scalar
        action_int = jnp.asarray(action, dtype=jnp.int32)
        # Flatten and take first element (handles both scalar and array)
        action_int = jnp.ravel(action_int)[0]
        
        dx, dy, fire = _action_to_motion(action_int)
        
        # Check if game should start (any movement input)
        has_movement = (dx != 0) | (dy != 0)
        game_started = state.game_started | has_movement
        
        # Update game state
        state = state._replace(game_started=game_started)
        state = self._update_timer(state)
        state = self._handle_xray_vision(state, dx, dy, fire)
        state = self._handle_subway_transitions(state, dx, dy)
        state = self._player_step(state, dx, dy, fire)
        state = self._handle_frame_transitions(state, dx, dy)
        state = self._update_satellites(state)
        state = self._update_lex(state)
        state = self._update_henchmen(state)
        state = self._update_helicopter(state)
        state = self._check_collisions(state)
        state = self._check_captures(state)
        state = self._check_jail(state)
        state = self._check_bridge(state)
        state = self._check_phone_booth(state)
        state = self._check_lois_revival(state)
        
        # Check win condition
        done = self._get_done(state)
        reward = self._get_reward(prev, state)

        state = state._replace(done=done, step_counter=state.step_counter + 1)
        obs = self._get_observation(state)
        info = self._get_info(state)

        return obs, state, reward, done, info

    # -------------------------
    # Helper methods
    # -------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _update_timer(self, state: SupermanState) -> SupermanState:
        """Update game timer - only runs when game started and not finished."""
        # Timer only increments if game has started and not finished
        should_increment = state.game_started & ~state.done
        new_timer = jnp.where(should_increment, state.timer + 1, state.timer)
        return state._replace(timer=new_timer)
    
    @partial(jax.jit, static_argnums=(0,))
    def _handle_xray_vision(self, state: SupermanState, dx: jnp.ndarray, dy: jnp.ndarray, fire: jnp.ndarray) -> SupermanState:
        """Handle X-Ray Vision (FIRE + direction)."""
        # If FIRE pressed and moving, activate X-Ray in that direction
        has_fire = fire > 0
        has_movement = (dx != 0) | (dy != 0)
        
        # Determine direction: 0=up, 1=right, 2=down, 3=left
        direction = jnp.where(dy < 0, 0,  # up
                      jnp.where(dx > 0, 1,  # right
                      jnp.where(dy > 0, 2,  # down
                      3)))  # left
        
        xray_active = has_fire & has_movement
        xray_direction = jnp.where(has_fire & has_movement, direction, state.xray_direction)
        
        return state._replace(xray_active=xray_active, xray_direction=xray_direction)
    
    @partial(jax.jit, static_argnums=(0,))
    def _handle_frame_transitions(self, state: SupermanState, dx: jnp.ndarray, dy: jnp.ndarray) -> SupermanState:
        """Handle transitions between city block frames (east/west only)."""
        # Only transition if not in subway
        can_transition = ~state.in_subway
        
        # Transition east (right) - move to next frame
        # Check if player is at or near right edge and trying to move right
        at_right_edge = state.player_x >= self.consts.SCREEN_W - 10
        moving_right = (dx > 0) & can_transition
        transition_east = at_right_edge & moving_right
        
        # Transition west (left) - move to previous frame
        # Check if player is at or near left edge and trying to move left
        at_left_edge = state.player_x <= 10
        moving_left = (dx < 0) & can_transition
        transition_west = at_left_edge & moving_left
        
        # Update frame
        new_frame = jnp.where(
            transition_east,
            (state.current_frame + 1) % self.consts.NUM_FRAMES,
            jnp.where(
                transition_west,
                (state.current_frame - 1) % self.consts.NUM_FRAMES,
                state.current_frame
            )
        )
        
        # Reset player position when transitioning
        new_x = jnp.where(
            transition_east,
            jnp.array(10, dtype=jnp.int32),  # Enter from left
            jnp.where(
                transition_west,
                jnp.array(self.consts.SCREEN_W - 10, dtype=jnp.int32),  # Enter from right
                state.player_x
            )
        )
        
        return state._replace(current_frame=new_frame, player_x=new_x)
    
    @partial(jax.jit, static_argnums=(0,))
    def _handle_subway_transitions(self, state: SupermanState, dx: jnp.ndarray, dy: jnp.ndarray) -> SupermanState:
        """Handle subway system entry/exit and navigation."""
        # Subway entrance detection (at subway Y position, near center)
        at_subway_entrance = (jnp.abs(state.player_y - self.consts.SUBWAY_ENTRANCE_Y) < 10) & \
                            (jnp.abs(state.player_x - self.consts.SCREEN_W // 2) < 20)
        
        # Enter subway: move down at entrance
        enter_subway = at_subway_entrance & (dy > 0) & ~state.in_subway & (state.player_altitude == 0)
        
        # Exit subway: move left, right, or down while in subway
        exit_subway = state.in_subway & ((dx != 0) | (dy > 0))
        
        # Determine which subway area based on current frame
        subway_area_from_frame = state.current_frame % 4
        
        # In subway: moving up connects to next area (areas connected at top)
        moving_up_in_subway = state.in_subway & (dy < 0)
        next_area = (state.subway_area + 1) % 4
        
        # Update subway area
        new_subway_area = jnp.where(
            enter_subway,
            subway_area_from_frame,
            jnp.where(
                moving_up_in_subway,
                next_area,
                jnp.where(
                    exit_subway,
                    jnp.array(0, dtype=jnp.int32),
                    state.subway_area
                )
            )
        )
        
        # Update subway state
        new_in_subway = (state.in_subway | enter_subway) & ~exit_subway
        
        return state._replace(in_subway=new_in_subway, subway_area=new_subway_area)
    
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: SupermanState, dx: jnp.ndarray, dy: jnp.ndarray, fire: jnp.ndarray) -> SupermanState:
        """Update player position and state."""
        # Determine speed based on player state
        speed = jnp.where(state.player_state == self.consts.PLAYER_SUPERMAN,
                         self.consts.SUPERMAN_FLY_SPEED,
                         self.consts.CLARK_WALK_SPEED)
        
        # CLARK KENT can only move LEFT/RIGHT (walking)
        # SUPERMAN can FLY (all directions)
        is_clark = state.player_state == self.consts.PLAYER_CLARK
        is_superman = state.player_state == self.consts.PLAYER_SUPERMAN
        is_weakened = state.player_state == self.consts.PLAYER_WEAKENED
        
        # Clark Kent: only horizontal movement (left/right)
        # Superman: can fly (all directions)
        # Weakened: only horizontal movement
        can_fly = is_superman & (state.player_altitude > 0)
        can_move_vertically = (is_superman | (state.player_altitude > 0)) & ~is_clark & ~is_weakened
        
        # Apply movement restrictions
        dx_allowed = dx  # All can move horizontally
        dy_allowed = jnp.where(can_move_vertically, dy, jnp.array(0, dtype=jnp.int32))  # Only Superman can move vertically
        
        # Update position
        new_x = state.player_x + dx_allowed * speed
        new_y = state.player_y + dy_allowed * speed
        
        # Clamp to screen bounds (allow slight overflow for frame transitions)
        # Frame transitions will handle resetting position if needed
        new_x = jnp.clip(new_x, -5, self.consts.SCREEN_W + 5)
        new_y = jnp.clip(new_y, 0, self.consts.SCREEN_H - 1)
        
        # Update altitude (only Superman can fly)
        is_superman = state.player_state == self.consts.PLAYER_SUPERMAN
        is_moving = (dx_allowed != 0) | (dy_allowed != 0)
        landing = (dx_allowed == 0) & (dy_allowed == 0) & (state.player_altitude > 0)
        
        # Clark Kent always stays on ground (altitude = 0)
        # Superman can fly (altitude > 0)
        new_altitude = jnp.where(
            ~is_superman,
            jnp.array(0, dtype=jnp.int32),  # Clark Kent always on ground
            jnp.where(
                landing,
                jnp.maximum(state.player_altitude - 1, 0),  # Float down
                jnp.where(
                    is_moving & (dy_allowed != 0),  # Moving vertically
                    jnp.minimum(state.player_altitude + 1, 10),  # Rise up
                    state.player_altitude
                )
            )
        )
        
        # If carrying something, move it with player
        new_state = state._replace(player_x=new_x, player_y=new_y, player_altitude=new_altitude)
        
        # Update carried entity position
        new_state = self._update_carried_entity(new_state)
        
        return new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_carried_entity(self, state: SupermanState) -> SupermanState:
        """Update position of entity being carried."""
        # If carrying Lex (index 0)
        lex_carried = state.carrying == 0
        new_lex_x = jnp.where(lex_carried, state.player_x, state.lex_x)
        new_lex_y = jnp.where(lex_carried, state.player_y, state.lex_y)
        
        # If carrying a henchman (indices 1-5)
        def update_henchman(i, carry_idx):
            carried = state.carrying == carry_idx
            new_x = jnp.where(carried, state.player_x, state.henchmen_x[i])
            new_y = jnp.where(carried, state.player_y, state.henchmen_y[i])
            return new_x, new_y
        
        # Update all henchmen
        henchmen_indices = jnp.arange(5)
        henchmen_carry_indices = henchmen_indices + 1  # 1-5
        new_henchmen_x = jnp.zeros_like(state.henchmen_x)
        new_henchmen_y = jnp.zeros_like(state.henchmen_y)
        
        for i in range(5):
            new_x, new_y = update_henchman(i, i + 1)
            new_henchmen_x = new_henchmen_x.at[i].set(new_x)
            new_henchmen_y = new_henchmen_y.at[i].set(new_y)
        
        # If carrying Lois (index 6)
        lois_carried = state.carrying == 6
        new_lois_x = jnp.where(lois_carried, state.player_x, state.lois_x)
        new_lois_y = jnp.where(lois_carried, state.player_y, state.lois_y)
        
        # If carrying bridge piece (indices 7-9)
        bridge_carried = (state.carrying >= 7) & (state.carrying <= 9)
        bridge_idx = state.carrying - 7
        new_bridge_x = state.bridge_pieces_x
        new_bridge_y = state.bridge_pieces_y
        
        for i in range(3):
            carried = (bridge_carried) & (bridge_idx == i)
            new_bridge_x = new_bridge_x.at[i].set(jnp.where(carried, state.player_x, state.bridge_pieces_x[i]))
            new_bridge_y = new_bridge_y.at[i].set(jnp.where(carried, state.player_y, state.bridge_pieces_y[i]))
        
        return state._replace(
            lex_x=new_lex_x, lex_y=new_lex_y,
            henchmen_x=new_henchmen_x, henchmen_y=new_henchmen_y,
            lois_x=new_lois_x, lois_y=new_lois_y,
            bridge_pieces_x=new_bridge_x, bridge_pieces_y=new_bridge_y
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_satellites(self, state: SupermanState) -> SupermanState:
        """Update Kryptonite satellites - they track the player."""
        def update_satellite(i, carry):
            # Move towards player if active
            dx = state.player_x - state.satellites_x[i]
            dy = state.player_y - state.satellites_y[i]
            
            # Normalize direction
            dist = jnp.sqrt(dx * dx + dy * dy + 1e-6)
            dx_norm = jnp.where(dist > 0, dx / dist, 0)
            dy_norm = jnp.where(dist > 0, dy / dist, 0)
            
            # Move only if active
            move_x = dx_norm * self.consts.SATELLITE_SPEED * state.satellites_active[i]
            move_y = dy_norm * self.consts.SATELLITE_SPEED * state.satellites_active[i]
            
            new_x = state.satellites_x[i] + move_x
            new_y = state.satellites_y[i] + move_y
            
            # Clamp to screen
            new_x = jnp.clip(new_x, 0, self.consts.SCREEN_W - 1)
            new_y = jnp.clip(new_y, 0, self.consts.SCREEN_H - 1)
            
            return new_x, new_y
        
        new_sat_x = jnp.zeros_like(state.satellites_x)
        new_sat_y = jnp.zeros_like(state.satellites_y)
        
        for i in range(3):
            new_x, new_y = update_satellite(i, None)
            new_sat_x = new_sat_x.at[i].set(new_x)
            new_sat_y = new_sat_y.at[i].set(new_y)
        
        return state._replace(satellites_x=new_sat_x, satellites_y=new_sat_y)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_lex(self, state: SupermanState) -> SupermanState:
        """Update Lex Luthor AI - runs away from player if not captured and in same frame."""
        current_frame = state.current_frame.astype(jnp.int32)
        
        # Only update if in same frame as player
        lex_in_frame = (state.lex_frame == current_frame) & ~state.in_subway
        
        # Run away from player
        dx = state.lex_x - state.player_x
        dy = state.lex_y - state.player_y
        
        # Normalize
        dist = jnp.sqrt(dx * dx + dy * dy + 1e-6)
        dx_norm = jnp.where(dist > 0, dx / dist, 0)
        dy_norm = jnp.where(dist > 0, dy / dist, 0)
        
        # Only move if not captured, not in jail, and in same frame
        can_move = ~(state.lex_captured | state.lex_in_jail) & lex_in_frame
        move_x = dx_norm * self.consts.LEX_SPEED * can_move
        move_y = dy_norm * self.consts.LEX_SPEED * can_move
        
        new_x = state.lex_x + move_x
        new_y = state.lex_y + move_y
        
        new_x = jnp.clip(new_x, 0, self.consts.SCREEN_W - 1)
        new_y = jnp.clip(new_y, 0, self.consts.SCREEN_H - 1)
        
        return state._replace(lex_x=new_x, lex_y=new_y)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_henchmen(self, state: SupermanState) -> SupermanState:
        """Update henchmen AI - run away from player (only if in same frame)."""
        current_frame = state.current_frame.astype(jnp.int32)
        
        def update_henchman(i, carry):
            # Only update if in same frame as player
            hench_in_frame = (state.henchmen_frames[i] == current_frame) & ~state.in_subway
            
            # Run away from player
            dx = state.henchmen_x[i] - state.player_x
            dy = state.henchmen_y[i] - state.player_y
            
            dist = jnp.sqrt(dx * dx + dy * dy + 1e-6)
            dx_norm = jnp.where(dist > 0, dx / dist, 0)
            dy_norm = jnp.where(dist > 0, dy / dist, 0)
            
            # Only move if not captured, not in jail, and in same frame
            can_move = ~(state.henchmen_captured[i] | state.henchmen_in_jail[i]) & hench_in_frame
            move_x = dx_norm * self.consts.HENCHMAN_SPEED * can_move
            move_y = dy_norm * self.consts.HENCHMAN_SPEED * can_move
            
            new_x = state.henchmen_x[i] + move_x
            new_y = state.henchmen_y[i] + move_y
            
            new_x = jnp.clip(new_x, 0, self.consts.SCREEN_W - 1)
            new_y = jnp.clip(new_y, 0, self.consts.SCREEN_H - 1)
            
            return new_x, new_y
        
        new_hench_x = jnp.zeros_like(state.henchmen_x)
        new_hench_y = jnp.zeros_like(state.henchmen_y)
        
        for i in range(5):
            new_x, new_y = update_henchman(i, None)
            new_hench_x = new_hench_x.at[i].set(new_x)
            new_hench_y = new_hench_y.at[i].set(new_y)
        
        return state._replace(henchmen_x=new_hench_x, henchmen_y=new_hench_y)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_helicopter(self, state: SupermanState) -> SupermanState:
        """Update helicopter - moves left to right, can help or hinder."""
        # Simple left-right movement
        new_x = state.helicopter_x + state.helicopter_direction * self.consts.HELICOPTER_SPEED
        
        # Bounce at edges
        new_direction = jnp.where(new_x >= self.consts.SCREEN_W - 1, -1,
                         jnp.where(new_x <= 0, 1, state.helicopter_direction))
        new_x = jnp.clip(new_x, 0, self.consts.SCREEN_W - 1)
        
        return state._replace(helicopter_x=new_x, helicopter_direction=new_direction)
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_collisions(self, state: SupermanState) -> SupermanState:
        """Check collisions with satellites (weakens player)."""
        # Check collision with any active satellite
        def check_satellite(i, carry):
            dx = jnp.abs(state.player_x - state.satellites_x[i])
            dy = jnp.abs(state.player_y - state.satellites_y[i])
            hit = (dx < self.consts.SATELLITE_HIT_DISTANCE) & (dy < self.consts.SATELLITE_HIT_DISTANCE)
            hit = hit & state.satellites_active[i]
            
            # If hit and not already weakened, become weakened
            return jnp.where(hit & (state.player_state == self.consts.PLAYER_SUPERMAN),
                           self.consts.PLAYER_WEAKENED, state.player_state)
        
        new_player_state = state.player_state
        for i in range(3):
            new_player_state = check_satellite(i, None)
        
        return state._replace(player_state=new_player_state)
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_captures(self, state: SupermanState) -> SupermanState:
        """Check if player can capture entities (must be flying as Superman)."""
        can_capture = (state.player_state == self.consts.PLAYER_SUPERMAN) & (state.player_altitude > 0) & (state.carrying == -1) & ~state.in_subway
        
        # Only capture entities in current frame
        current_frame = state.current_frame.astype(jnp.int32)
        
        # Try to capture Lex (only if in current frame)
        lex_in_frame = (state.lex_frame == current_frame)
        lex_dist = jnp.sqrt((state.player_x - state.lex_x)**2 + (state.player_y - state.lex_y)**2)
        capture_lex = can_capture & lex_in_frame & (lex_dist < self.consts.CAPTURE_DISTANCE) & ~state.lex_captured & ~state.lex_in_jail
        
        # Try to capture henchmen (only if in current frame)
        def check_henchman(i, carry):
            # If already captured or in jail, return current captured state
            already_captured_or_jailed = state.henchmen_captured[i] | state.henchmen_in_jail[i]
            hench_in_frame = (state.henchmen_frames[i] == current_frame)
            dist = jnp.sqrt((state.player_x - state.henchmen_x[i])**2 + (state.player_y - state.henchmen_y[i])**2)
            can_capture_this = can_capture & hench_in_frame & (dist < self.consts.CAPTURE_DISTANCE)
            # Return True if already captured, or if we can capture this one
            return jnp.where(already_captured_or_jailed, state.henchmen_captured[i], can_capture_this)
        
        new_lex_captured = state.lex_captured | capture_lex
        new_carrying = jnp.where(capture_lex, 0, state.carrying)  # 0 = Lex
        
        new_hench_captured = state.henchmen_captured
        for i in range(5):
            captured = check_henchman(i, None)
            new_hench_captured = new_hench_captured.at[i].set(captured)
            new_carrying = jnp.where(captured & (new_carrying == -1), i + 1, new_carrying)  # 1-5 = henchmen
        
        # Try to capture Lois (only if in current frame)
        lois_in_frame = (state.lois_frame == current_frame)
        lois_dist = jnp.sqrt((state.player_x - state.lois_x)**2 + (state.player_y - state.lois_y)**2)
        capture_lois = can_capture & lois_in_frame & (lois_dist < self.consts.CAPTURE_DISTANCE) & (new_carrying == -1)
        new_carrying = jnp.where(capture_lois, 6, new_carrying)  # 6 = Lois
        
        # Try to capture bridge pieces (only if in current frame)
        def check_bridge_piece(i, carry):
            # If already placed, can't capture
            already_placed = state.bridge_pieces_placed[i]
            piece_in_frame = (state.bridge_pieces_frames[i] == current_frame)
            dist = jnp.sqrt((state.player_x - state.bridge_pieces_x[i])**2 + (state.player_y - state.bridge_pieces_y[i])**2)
            can_capture_this = can_capture & piece_in_frame & (dist < self.consts.CAPTURE_DISTANCE)
            # Return False if already placed, otherwise return capture check
            return jnp.where(already_placed, False, can_capture_this)
        
        for i in range(3):
            captured = check_bridge_piece(i, None)
            new_carrying = jnp.where(captured & (new_carrying == -1), i + 7, new_carrying)  # 7-9 = bridge pieces
        
        # Release if landed
        release = (state.player_altitude == 0) & (state.carrying >= 0)
        new_carrying = jnp.where(release, -1, new_carrying)
        
        # If released a crook, they escape
        new_lex_captured = jnp.where(release & (state.carrying == 0), False, new_lex_captured)
        for i in range(5):
            new_hench_captured = new_hench_captured.at[i].set(
                jnp.where(release & (state.carrying == i + 1), False, new_hench_captured[i])
            )
        
        return state._replace(
            lex_captured=new_lex_captured,
            henchmen_captured=new_hench_captured,
            carrying=new_carrying
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_jail(self, state: SupermanState) -> SupermanState:
        """Check if player can put crooks in jail."""
        # Must be flying through jail bars while carrying a crook
        jail_dist = jnp.sqrt((state.player_x - self.consts.JAIL_X)**2 + (state.player_y - self.consts.JAIL_Y)**2)
        at_jail = (jail_dist < self.consts.JAIL_DISTANCE) & (state.player_altitude > 0)
        
        # Put Lex in jail
        put_lex_in_jail = at_jail & (state.carrying == 0) & state.lex_captured & ~state.lex_in_jail
        new_lex_in_jail = state.lex_in_jail | put_lex_in_jail
        new_carrying = jnp.where(put_lex_in_jail, -1, state.carrying)
        
        # Put henchmen in jail
        new_hench_in_jail = state.henchmen_in_jail
        for i in range(5):
            put_in_jail = at_jail & (new_carrying == i + 1) & state.henchmen_captured[i] & ~state.henchmen_in_jail[i]
            new_hench_in_jail = new_hench_in_jail.at[i].set(state.henchmen_in_jail[i] | put_in_jail)
            new_carrying = jnp.where(put_in_jail, -1, new_carrying)
        
        return state._replace(
            lex_in_jail=new_lex_in_jail,
            henchmen_in_jail=new_hench_in_jail,
            carrying=new_carrying
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_bridge(self, state: SupermanState) -> SupermanState:
        """Check if bridge pieces can be placed."""
        # Bridge pieces must be between bridge left and right positions
        bridge_center_x = (self.consts.BRIDGE_LEFT_X + self.consts.BRIDGE_RIGHT_X) // 2
        bridge_center_y = self.consts.BRIDGE_Y
        
        def check_place_piece(i, carry):
            # If already placed, return placed state and current position
            already_placed = state.bridge_pieces_placed[i]
            
            # If carrying this piece and near bridge, place it
            carrying_piece = (state.carrying == i + 7) & (state.player_altitude == 0)
            dist_to_bridge = jnp.sqrt((state.player_x - bridge_center_x)**2 + (state.player_y - bridge_center_y)**2)
            near_bridge = dist_to_bridge < 15
            
            placed = carrying_piece & near_bridge
            new_x = jnp.where(placed, bridge_center_x + (i - 1) * 10, state.bridge_pieces_x[i])
            new_y = jnp.where(placed, bridge_center_y, state.bridge_pieces_y[i])
            
            # If already placed, keep it placed; otherwise use new placed state
            final_placed = already_placed | placed
            final_x = jnp.where(already_placed, state.bridge_pieces_x[i], new_x)
            final_y = jnp.where(already_placed, state.bridge_pieces_y[i], new_y)
            
            return final_placed, final_x, final_y
        
        new_placed = state.bridge_pieces_placed
        new_bridge_x = state.bridge_pieces_x
        new_bridge_y = state.bridge_pieces_y
        new_carrying = state.carrying
        
        for i in range(3):
            placed, new_x, new_y = check_place_piece(i, None)
            new_placed = new_placed.at[i].set(placed)
            new_bridge_x = new_bridge_x.at[i].set(new_x)
            new_bridge_y = new_bridge_y.at[i].set(new_y)
            new_carrying = jnp.where(placed & (new_carrying == i + 7), -1, new_carrying)
        
        # Check if bridge is complete (all 3 pieces placed)
        bridge_complete = jnp.all(new_placed)
        
        return state._replace(
            bridge_pieces_placed=new_placed,
            bridge_pieces_x=new_bridge_x,
            bridge_pieces_y=new_bridge_y,
            bridge_complete=bridge_complete,
            carrying=new_carrying
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_phone_booth(self, state: SupermanState) -> SupermanState:
        """Check if Clark Kent can transform to Superman at phone booth."""
        # Phone booth is in frame 0
        in_phone_booth_frame = (state.current_frame == self.consts.PHONE_BOOTH_FRAME) & ~state.in_subway
        booth_dist = jnp.sqrt((state.player_x - self.consts.PHONE_BOOTH_X)**2 + (state.player_y - self.consts.PHONE_BOOTH_Y)**2)
        
        # Clark Kent can transform to Superman when at phone booth (on ground, not carrying anything)
        at_booth = (booth_dist < 15) & (state.player_state == self.consts.PLAYER_CLARK) & \
                   (state.player_altitude == 0) & (state.carrying == -1) & in_phone_booth_frame
        
        # Transform: Clark -> Superman (one-way transformation at phone booth)
        new_player_state = jnp.where(at_booth, self.consts.PLAYER_SUPERMAN, state.player_state)
        
        return state._replace(player_state=new_player_state)

    @partial(jax.jit, static_argnums=(0,))
    def _check_lois_revival(self, state: SupermanState) -> SupermanState:
        """Check if touching Lois revives weakened player."""
        lois_dist = jnp.sqrt((state.player_x - state.lois_x)**2 + (state.player_y - state.lois_y)**2)
        touch_lois = lois_dist < self.consts.CAPTURE_DISTANCE
        
        # Revive if weakened and touching Lois
        revived = touch_lois & (state.player_state == self.consts.PLAYER_WEAKENED)
        new_state = jnp.where(revived, self.consts.PLAYER_SUPERMAN, state.player_state)
        
        return state._replace(player_state=new_state)

    # -------------------------
    # Observation / Info / Reward / Done
    # -------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SupermanState) -> SupermanObs:
        """Get observation from state."""
        crooks_in_jail = jnp.sum(state.henchmen_in_jail.astype(jnp.int32)) + jnp.sum(state.lex_in_jail.astype(jnp.int32))
        
        return SupermanObs(
            state.player_x, state.player_y, state.player_state, state.player_altitude, state.carrying,
            state.lex_x, state.lex_y, state.lex_captured,
            state.henchmen_x, state.henchmen_y,
            state.satellites_x, state.satellites_y,
            state.lois_x, state.lois_y,
            state.bridge_pieces_x, state.bridge_pieces_y,
            state.bridge_complete,
            crooks_in_jail
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SupermanState) -> SupermanInfo:
        """Get info from state."""
        timer_seconds = state.timer // 60  # Assuming 60 FPS
        crooks_captured = jnp.sum(state.henchmen_in_jail.astype(jnp.int32)) + jnp.sum(state.lex_in_jail.astype(jnp.int32))
        return SupermanInfo(
            jnp.array(0, dtype=jnp.int32),
            timer_seconds,
            crooks_captured
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: SupermanState, state: SupermanState) -> jnp.ndarray:
        """Calculate reward."""
        reward = jnp.array(0.0, dtype=jnp.float32)
        
        # Reward for capturing crooks
        prev_captured = jnp.sum(prev.henchmen_in_jail.astype(jnp.int32)) + jnp.sum(prev.lex_in_jail.astype(jnp.int32))
        curr_captured = jnp.sum(state.henchmen_in_jail.astype(jnp.int32)) + jnp.sum(state.lex_in_jail.astype(jnp.int32))
        reward = reward + (curr_captured - prev_captured) * 100.0
        
        # Reward for placing bridge pieces
        prev_placed = jnp.sum(prev.bridge_pieces_placed.astype(jnp.int32))
        curr_placed = jnp.sum(state.bridge_pieces_placed.astype(jnp.int32))
        reward = reward + (curr_placed - prev_placed) * 50.0
        
        # Reward for completing bridge
        bridge_completed = state.bridge_complete & ~prev.bridge_complete
        reward = reward + jnp.where(bridge_completed, 200.0, 0.0)
        
        # Small penalty for being weakened
        is_weakened = state.player_state == self.consts.PLAYER_WEAKENED
        reward = reward - jnp.where(is_weakened, 0.1, 0.0)
        
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SupermanState) -> jnp.ndarray:
        """Check win condition."""
        # Win: All crooks in jail, bridge complete, player is Clark Kent at Daily Planet
        all_crooks_jailed = (jnp.sum(state.henchmen_in_jail) == 5) & state.lex_in_jail
        bridge_done = state.bridge_complete
        
        daily_planet_dist = jnp.sqrt((state.player_x - self.consts.DAILY_PLANET_X)**2 + 
                                     (state.player_y - self.consts.DAILY_PLANET_Y)**2)
        at_daily_planet = (daily_planet_dist < 10) & (state.player_state == self.consts.PLAYER_CLARK)
        
        won = all_crooks_jailed & bridge_done & at_daily_planet
        
        return won

    # -------------------------
    # Spaces
    # -------------------------
    def action_space(self):
        return spaces.Discrete(18)

    def observation_space(self):
        return spaces.Dict({
            "player_x": spaces.Box(0, self.consts.SCREEN_W, (), jnp.int32),
            "player_y": spaces.Box(0, self.consts.SCREEN_H, (), jnp.int32),
            "player_state": spaces.Discrete(3),
            "player_altitude": spaces.Box(0, 10, (), jnp.int32),
            "carrying": spaces.Box(-1, 9, (), jnp.int32),
        })


# -------------------------
# RENDERER
# -------------------------
class SupermanRenderer(JAXGameRenderer):
    """Renderer for Superman game - matches original Atari 2600 visuals."""

    def __init__(self, consts: SupermanConstants = None):
        self.consts = consts or SupermanConstants()
        # Sprite sizes (pixelated, blocky style)
        self.player_h, self.player_w = 10, 8
        self.lex_h, self.lex_w = 10, 8
        self.henchman_h, self.henchman_w = 8, 6
        self.sat_h, self.sat_w = 6, 6
        self.lois_h, self.lois_w = 8, 6
        self.bridge_h, self.bridge_w = 6, 16
        self.helicopter_h, self.helicopter_w = 8, 14
        self.phone_booth_h, self.phone_booth_w = 16, 10
        self.jail_h, self.jail_w = 30, 20
        self.daily_planet_h, self.daily_planet_w = 20, 15
        
        # UI bar height
        self.ui_bar_height = 8
        
        # City block colors (different frames have different backgrounds)
        # Store as JAX array for JIT compatibility
        self.city_colors = jnp.array([
            [128, 0, 128],    # Purple
            [255, 0, 255],    # Magenta
            [0, 0, 255],      # Blue
            [0, 255, 0],      # Green
            [255, 255, 0],   # Yellow
        ], dtype=jnp.uint8)
        
        # Ground color (dark green/blue)
        self.ground_color = jnp.array([0, 100, 50], dtype=jnp.uint8)

    def _empty_canvas(self, frame_idx: int, in_subway: bool, subway_area: int):
        """Create empty canvas with city block background or subway."""
        h, w = self.consts.SCREEN_H, self.consts.SCREEN_W
        
        # Subway colors
        subway_colors = [
            jnp.array([255, 255, 0], dtype=jnp.uint8),   # Yellow
            jnp.array([0, 0, 255], dtype=jnp.uint8),    # Blue
            jnp.array([0, 255, 0], dtype=jnp.uint8),    # Green
            jnp.array([255, 192, 203], dtype=jnp.uint8), # Pink
        ]
        
        # Choose background color
        bg_color = jax.lax.cond(
            in_subway,
            lambda: subway_colors[subway_area],
            lambda: self.city_colors[frame_idx % len(self.city_colors)],
            None
        )
        
        canvas = jnp.full((h, w, 3), bg_color, dtype=jnp.uint8)
        
        # Draw ground (bottom portion) - only if not in subway
        ground_start = h - 40
        ground_mask = (jnp.arange(h)[:, None] >= ground_start) & ~in_subway
        canvas = jnp.where(ground_mask[:, :, None], self.ground_color, canvas)
        
        return canvas

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Render the game state matching original Atari 2600 visuals."""
        # Create canvas with frame-appropriate background
        frame_idx = state.current_frame.astype(jnp.int32)
        in_subway = state.in_subway
        subway_area = state.subway_area.astype(jnp.int32)
        
        # Subway colors (as JAX array for JIT compatibility)
        subway_colors = jnp.array([
            [255, 255, 0],      # Yellow
            [0, 0, 255],        # Blue
            [0, 255, 0],        # Green
            [255, 192, 203],    # Pink
        ], dtype=jnp.uint8)
        
        # Choose background using JAX indexing
        num_city_colors = self.city_colors.shape[0]
        city_color_idx = frame_idx % num_city_colors
        city_bg = self.city_colors[city_color_idx]  # Shape: (3,)
        subway_bg = subway_colors[subway_area]  # Shape: (3,)
        
        # Use jnp.where with proper broadcasting
        bg_color = jnp.where(
            in_subway,
            subway_bg,
            city_bg
        )  # Shape: (3,)
        
        h, w = self.consts.SCREEN_H, self.consts.SCREEN_W
        canvas = jnp.full((h, w, 3), bg_color, dtype=jnp.uint8)
        
        # Draw ground (only if not in subway)
        ground_start = h - 40
        ground_mask = (jnp.arange(h)[:, None] >= ground_start) & ~in_subway
        canvas = jnp.where(ground_mask[:, :, None], self.ground_color, canvas)

        # Helper to draw rectangle
        def draw_rect(img, x, y, hh, ww, color):
            x0 = jnp.clip(x - ww // 2, 0, self.consts.SCREEN_W - 1)
            x1 = jnp.clip(x0 + ww, 0, self.consts.SCREEN_W)
            y0 = jnp.clip(y - hh // 2, 0, self.consts.SCREEN_H - 1)
            y1 = jnp.clip(y0 + hh, 0, self.consts.SCREEN_H)

            xs = jnp.arange(self.consts.SCREEN_W)
            ys = jnp.arange(self.consts.SCREEN_H)[:, None]
            mask_x = (xs >= x0) & (xs < x1)
            mask_y = (ys >= y0) & (ys < y1)
            mask = (mask_x[None, :] & mask_y[:, None, 0])

            color_arr = jnp.asarray(color, dtype=jnp.uint8)
            mask3 = mask[:, :, None]
            return jnp.where(mask3, color_arr, img)
        
        # Helper to draw vertical bar (for jail)
        def draw_vertical_bar(img, x, y, hh, ww, color):
            return draw_rect(img, x, y, hh, ww, color)
        
        # Helper to draw building (blocky cityscape)
        def draw_building(img, x, y, height, width, color):
            return draw_rect(img, x, y, height, width, color)
        
        # --- 1. Draw UI Bar (Red bar at top) ---
        ui_bar_color = jnp.array([200, 0, 0], dtype=jnp.uint8)  # Red
        ui_bar_mask = jnp.arange(self.consts.SCREEN_H)[:, None] < self.ui_bar_height
        canvas = jnp.where(ui_bar_mask[:, :, None], ui_bar_color, canvas)
        
        # --- 2. Draw Crook Markers (I IIII format) ---
        # Count remaining crooks
        lex_remaining = ~state.lex_in_jail
        henchmen_remaining = jnp.sum(~state.henchmen_in_jail.astype(jnp.int32))
        total_remaining = lex_remaining.astype(jnp.int32) + henchmen_remaining
        
        # Draw markers (white vertical bars)
        marker_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        marker_y = self.ui_bar_height // 2
        marker_spacing = 8
        
        def draw_marker(i, img):
            # Lex marker (larger) if not jailed
            if i == 0:
                marker_x = 5
                marker_w = 3
                marker_h = 6
                should_draw = lex_remaining
            else:
                # Henchman markers (smaller)
                marker_x = 10 + (i - 1) * marker_spacing
                marker_w = 2
                marker_h = 4
                should_draw = (i <= henchmen_remaining) & lex_remaining
            
            return jax.lax.cond(
                should_draw,
                lambda x: draw_rect(x, marker_x, marker_y, marker_h, marker_w, marker_color),
                lambda x: x,
                img
            )
        
        # Draw Lex marker (largest)
        canvas = jax.lax.cond(
            lex_remaining,
            lambda x: draw_rect(x, 5, marker_y, 6, 3, marker_color),
            lambda x: x,
            canvas
        )
        
        # Draw henchman markers
        for i in range(5):
            marker_x = 10 + i * marker_spacing
            should_draw = (i < henchmen_remaining) & lex_remaining
            canvas = jax.lax.cond(
                should_draw,
                lambda x, mx=marker_x: draw_rect(x, mx, marker_y, 4, 2, marker_color),
                lambda x: x,
                canvas
            )
        
        # --- 3. Draw Timer (MM:SS format) in TOP RIGHT ---
        timer_seconds = state.timer // 60  # Assuming 60 FPS
        minutes = timer_seconds // 60
        seconds = timer_seconds % 60
        
        # Draw timer in TOP RIGHT corner (on the red UI bar)
        timer_y = self.ui_bar_height // 2
        white_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White background
        black_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black digits
        
        # Extract individual digits
        min_tens = minutes // 10
        min_ones = minutes % 10
        sec_tens = seconds // 10
        sec_ones = seconds % 10
        
        # Draw digit with black number inside white block
        def draw_digit_with_number(digit_val, x_pos, img):
            """Draw a white block with black digit inside."""
            # Each digit is 6 pixels wide, 7 pixels tall
            digit_h = 7
            digit_w = 6
            
            # Draw white background block
            img = draw_rect(img, x_pos, timer_y, digit_h, digit_w, white_color)
            
            # Draw black digit pattern inside (simple 7-segment style)
            # Each digit is represented by drawing black segments
            # Simple pattern: draw black pixels to form the digit shape
            center_x = x_pos
            center_y = timer_y
            
            # Draw digit segments based on value (simple blocky representation)
            # For each digit 0-9, draw a pattern of black segments
            segment_size = 1
            
            # Top horizontal segment
            def draw_top_segment(img_inner):
                return draw_rect(img_inner, center_x, center_y - 3, segment_size, digit_w - 2, black_color)
            
            # Middle horizontal segment
            def draw_mid_segment(img_inner):
                return draw_rect(img_inner, center_x, center_y, segment_size, digit_w - 2, black_color)
            
            # Bottom horizontal segment
            def draw_bot_segment(img_inner):
                return draw_rect(img_inner, center_x, center_y + 3, segment_size, digit_w - 2, black_color)
            
            # Top left vertical segment
            def draw_tl_segment(img_inner):
                return draw_rect(img_inner, center_x - 2, center_y - 1, 2, segment_size, black_color)
            
            # Top right vertical segment
            def draw_tr_segment(img_inner):
                return draw_rect(img_inner, center_x + 2, center_y - 1, 2, segment_size, black_color)
            
            # Bottom left vertical segment
            def draw_bl_segment(img_inner):
                return draw_rect(img_inner, center_x - 2, center_y + 1, 2, segment_size, black_color)
            
            # Bottom right vertical segment
            def draw_br_segment(img_inner):
                return draw_rect(img_inner, center_x + 2, center_y + 1, 2, segment_size, black_color)
            
            # Draw segments based on digit value (0-9)
            # 0: top, tl, tr, bl, br, bottom
            # 1: tr, br
            # 2: top, tr, mid, bl, bottom
            # 3: top, tr, mid, br, bottom
            # 4: tl, tr, mid, br
            # 5: top, tl, mid, br, bottom
            # 6: top, tl, mid, bl, br, bottom
            # 7: top, tr, br
            # 8: all segments
            # 9: top, tl, tr, mid, br, bottom
            
            # Use jax.lax.switch to select digit pattern
            def draw_digit_0(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_bl_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_1(img_inner):
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                return img_inner
            
            def draw_digit_2(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_bl_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_3(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_4(img_inner):
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                return img_inner
            
            def draw_digit_5(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_6(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_bl_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_7(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                return img_inner
            
            def draw_digit_8(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_bl_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            def draw_digit_9(img_inner):
                img_inner = draw_top_segment(img_inner)
                img_inner = draw_tl_segment(img_inner)
                img_inner = draw_tr_segment(img_inner)
                img_inner = draw_mid_segment(img_inner)
                img_inner = draw_br_segment(img_inner)
                img_inner = draw_bot_segment(img_inner)
                return img_inner
            
            # Select and draw the appropriate digit pattern
            digit_val_int = digit_val.astype(jnp.int32)
            img = jax.lax.switch(
                digit_val_int,
                [
                    draw_digit_0, draw_digit_1, draw_digit_2, draw_digit_3, draw_digit_4,
                    draw_digit_5, draw_digit_6, draw_digit_7, draw_digit_8, draw_digit_9
                ],
                img
            )
            
            return img
        
        # Calculate timer position (right-aligned)
        digit_width = 6
        digit_spacing = 2
        colon_width = 2
        total_width = 4 * digit_width + 3 * digit_spacing + colon_width
        timer_x_start = self.consts.SCREEN_W - total_width - 5  # 5 pixels from right edge
        
        # Draw timer: MM:SS format
        # Minutes tens
        canvas = draw_digit_with_number(min_tens, timer_x_start, canvas)
        # Minutes ones
        canvas = draw_digit_with_number(min_ones, timer_x_start + digit_width + digit_spacing, canvas)
        # Colon separator (white dots)
        colon_x = timer_x_start + 2 * (digit_width + digit_spacing)
        canvas = draw_rect(canvas, colon_x, timer_y - 2, 2, 1, white_color)
        canvas = draw_rect(canvas, colon_x, timer_y + 2, 2, 1, white_color)
        # Seconds tens
        canvas = draw_digit_with_number(sec_tens, colon_x + colon_width + digit_spacing, canvas)
        # Seconds ones
        canvas = draw_digit_with_number(sec_ones, colon_x + colon_width + digit_spacing + digit_width + digit_spacing, canvas)
        
        # --- 3b. Draw Frame Number (LARGE and VISIBLE) ---
        # Draw frame number prominently so player can see which frame they're in
        frame_num = frame_idx.astype(jnp.int32)
        frame_bg_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White background
        frame_text_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black text
        
        # Draw frame number in TOP CENTER of screen (large and visible)
        frame_display_y = self.ui_bar_height // 2
        frame_display_x = self.consts.SCREEN_W // 2  # Center of screen
        
        # Draw white background box for frame number
        frame_box_h = 8
        frame_box_w = 12
        canvas = draw_rect(canvas, frame_display_x, frame_display_y, frame_box_h, frame_box_w, frame_bg_color)
        
        # Draw frame number digit (0-7) using same digit rendering as timer
        def draw_frame_digit(digit_val, x_pos, img):
            """Draw frame digit with black number inside white block."""
            digit_h = 7
            digit_w = 6
            # Draw white background
            img = draw_rect(img, x_pos, frame_display_y, digit_h, digit_w, frame_bg_color)
            
            # Draw black digit pattern (simplified - just draw the digit value as pattern)
            # For frame numbers 0-7, draw simple patterns
            center_x = x_pos
            center_y = frame_display_y
            
            # Draw digit segments based on value
            def draw_segment(img_inner, seg_x, seg_y, seg_w, seg_h):
                return draw_rect(img_inner, seg_x, seg_y, seg_h, seg_w, frame_text_color)
            
            # Simple patterns for 0-7
            # 0: full box outline
            def draw_0(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x - 2, center_y, 1, 5)  # left
                img_inner = draw_segment(img_inner, center_x + 2, center_y, 1, 5)  # right
                img_inner = draw_segment(img_inner, center_x, center_y + 3, 4, 1)  # bottom
                return img_inner
            
            # 1: vertical line
            def draw_1(img_inner):
                return draw_segment(img_inner, center_x, center_y, 1, 6)
            
            # 2: top, top-right, middle, bottom-left, bottom
            def draw_2(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x + 2, center_y - 1, 1, 2)  # top-right
                img_inner = draw_segment(img_inner, center_x, center_y, 4, 1)  # middle
                img_inner = draw_segment(img_inner, center_x - 2, center_y + 1, 1, 2)  # bottom-left
                img_inner = draw_segment(img_inner, center_x, center_y + 3, 4, 1)  # bottom
                return img_inner
            
            # 3: top, top-right, middle, bottom-right, bottom
            def draw_3(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x + 2, center_y - 1, 1, 2)  # top-right
                img_inner = draw_segment(img_inner, center_x, center_y, 4, 1)  # middle
                img_inner = draw_segment(img_inner, center_x + 2, center_y + 1, 1, 2)  # bottom-right
                img_inner = draw_segment(img_inner, center_x, center_y + 3, 4, 1)  # bottom
                return img_inner
            
            # 4: top-left, top-right, middle, bottom-right
            def draw_4(img_inner):
                img_inner = draw_segment(img_inner, center_x - 2, center_y - 2, 1, 2)  # top-left
                img_inner = draw_segment(img_inner, center_x + 2, center_y - 2, 1, 4)  # top-right
                img_inner = draw_segment(img_inner, center_x, center_y, 4, 1)  # middle
                return img_inner
            
            # 5: top, top-left, middle, bottom-right, bottom
            def draw_5(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x - 2, center_y - 1, 1, 2)  # top-left
                img_inner = draw_segment(img_inner, center_x, center_y, 4, 1)  # middle
                img_inner = draw_segment(img_inner, center_x + 2, center_y + 1, 1, 2)  # bottom-right
                img_inner = draw_segment(img_inner, center_x, center_y + 3, 4, 1)  # bottom
                return img_inner
            
            # 6: top, top-left, middle, bottom-left, bottom-right, bottom
            def draw_6(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x - 2, center_y - 1, 1, 4)  # left
                img_inner = draw_segment(img_inner, center_x, center_y, 4, 1)  # middle
                img_inner = draw_segment(img_inner, center_x + 2, center_y + 1, 1, 2)  # bottom-right
                img_inner = draw_segment(img_inner, center_x, center_y + 3, 4, 1)  # bottom
                return img_inner
            
            # 7: top, top-right, bottom-right
            def draw_7(img_inner):
                img_inner = draw_segment(img_inner, center_x, center_y - 3, 4, 1)  # top
                img_inner = draw_segment(img_inner, center_x + 2, center_y - 1, 1, 5)  # right
                return img_inner
            
            # Select and draw the appropriate digit pattern
            digit_val_int = digit_val.astype(jnp.int32)
            img = jax.lax.switch(
                digit_val_int,
                [draw_0, draw_1, draw_2, draw_3, draw_4, draw_5, draw_6, draw_7],
                img
            )
            
            return img
        
        # Draw frame number (single digit 0-7)
        canvas = draw_frame_digit(frame_num, frame_display_x, canvas)
        
        # --- 4. Draw Cityscape Buildings (blocky, varying heights) - only if not in subway ---
        def draw_city_buildings(img):
            building_colors = [
                jnp.array([100, 100, 150], dtype=jnp.uint8),  # Light blue buildings
                jnp.array([150, 100, 150], dtype=jnp.uint8),  # Light purple buildings
                jnp.array([100, 150, 100], dtype=jnp.uint8),  # Light green buildings
            ]
            
            building_positions = [
                (20, 120, 30, 15),   # x, y, height, width
                (50, 140, 20, 12),
                (80, 130, 25, 14),
                (110, 150, 15, 10),
                (140, 125, 35, 16),
            ]
            
            result = img
            for i, (bx, by, bh, bw) in enumerate(building_positions):
                bcolor = building_colors[i % len(building_colors)]
                result = draw_building(result, bx, by, bh, bw, bcolor)
            return result
        
        canvas = jax.lax.cond(
            ~in_subway,
            draw_city_buildings,
            lambda x: x,
            canvas
        )
        
        # --- 4b. Draw Subway Structure (if in subway) ---
        def draw_subway_structure(img):
            platform_color = jnp.array([150, 150, 150], dtype=jnp.uint8)  # Gray platforms
            # Draw horizontal platform lines
            for py in [100, 150]:
                img = draw_rect(img, self.consts.SCREEN_W // 2, py, 4, self.consts.SCREEN_W, platform_color)
            # Draw subway doorway/entrance indicator
            doorway_color = jnp.array([255, 0, 0], dtype=jnp.uint8)  # Red outline
            img = draw_rect(img, self.consts.SCREEN_W // 2, self.consts.SCREEN_H - 20, 20, 30, doorway_color)
            return img
        
        canvas = jax.lax.cond(
            in_subway,
            draw_subway_structure,
            lambda x: x,
            canvas
        )
        
        # --- 4c. Draw Frame Boundary Indicators (left/right edges) ---
        def draw_frame_boundaries(img):
            boundary_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White indicators
            # Left edge indicator
            img = draw_rect(img, 2, self.consts.SCREEN_H // 2, 40, 2, boundary_color)
            # Right edge indicator
            img = draw_rect(img, self.consts.SCREEN_W - 2, self.consts.SCREEN_H // 2, 40, 2, boundary_color)
            return img
        
        canvas = jax.lax.cond(
            ~in_subway,
            draw_frame_boundaries,
            lambda x: x,
            canvas
        )
        
        # --- 4d. Draw Subway Entrance Indicator (when not in subway) ---
        def draw_subway_entrance(img):
            entrance_color = jnp.array([255, 255, 0], dtype=jnp.uint8)  # Yellow indicator
            # Draw entrance marker at subway Y position
            img = draw_rect(img, self.consts.SCREEN_W // 2, self.consts.SUBWAY_ENTRANCE_Y, 8, 30, entrance_color)
            return img
        
        canvas = jax.lax.cond(
            ~in_subway,
            draw_subway_entrance,
            lambda x: x,
            canvas
        )
        
        # --- 5. Draw Landmarks (only in their respective frames) ---
        frame_idx = state.current_frame.astype(jnp.int32)
        
        # Draw Jail (only in frame 1)
        def draw_jail(img):
            jail_base_color = jnp.array([200, 100, 0], dtype=jnp.uint8)  # Orange-brown
            jail_bar_color = jnp.array([255, 255, 0], dtype=jnp.uint8)  # Yellow bars
            
            # Jail base
            img = draw_rect(img, self.consts.JAIL_X, self.consts.JAIL_Y, 
                          self.jail_h, self.jail_w, jail_base_color)
            
            # Jail vertical bars
            bar_spacing = 3
            num_bars = 5
            for i in range(num_bars):
                bar_x = self.consts.JAIL_X - self.jail_w // 2 + 4 + i * bar_spacing
                img = draw_vertical_bar(img, bar_x, self.consts.JAIL_Y, self.jail_h - 4, 1, jail_bar_color)
            return img
        
        canvas = jax.lax.cond(
            (frame_idx == self.consts.JAIL_FRAME) & ~in_subway,
            draw_jail,
            lambda x: x,
            canvas
        )
        
        # Draw Phone Booth (only in frame 0)
        def draw_phone_booth(img):
            phone_booth_color = jnp.array([0, 100, 255], dtype=jnp.uint8)  # Blue
            phone_booth_outline = jnp.array([100, 200, 255], dtype=jnp.uint8)  # Light blue
            
            # Outline
            img = draw_rect(img, self.consts.PHONE_BOOTH_X, self.consts.PHONE_BOOTH_Y, 
                          self.phone_booth_h + 2, self.phone_booth_w + 2, phone_booth_outline)
            # Main body
            img = draw_rect(img, self.consts.PHONE_BOOTH_X, self.consts.PHONE_BOOTH_Y, 
                          self.phone_booth_h, self.phone_booth_w, phone_booth_color)
            return img
        
        canvas = jax.lax.cond(
            (frame_idx == self.consts.PHONE_BOOTH_FRAME) & ~in_subway,
            draw_phone_booth,
            lambda x: x,
            canvas
        )
        
        # Draw Daily Planet (only in frame 7)
        def draw_daily_planet(img):
            daily_planet_color = jnp.array([255, 255, 0], dtype=jnp.uint8)  # Yellow
            return draw_rect(img, self.consts.DAILY_PLANET_X, self.consts.DAILY_PLANET_Y, 
                          self.daily_planet_h, self.daily_planet_w, daily_planet_color)
        
        canvas = jax.lax.cond(
            (frame_idx == self.consts.DAILY_PLANET_FRAME) & ~in_subway,
            draw_daily_planet,
            lambda x: x,
            canvas
        )
        
        # --- 8. Draw Bridge Area and Pieces (only in bridge frame) ---
        def draw_bridge_area(img):
            bridge_support_color = jnp.array([150, 150, 150], dtype=jnp.uint8)  # Gray
            
            # Bridge supports
            img = draw_rect(img, self.consts.BRIDGE_LEFT_X, self.consts.BRIDGE_Y, 
                          8, 6, bridge_support_color)
            img = draw_rect(img, self.consts.BRIDGE_RIGHT_X, self.consts.BRIDGE_Y, 
                          8, 6, bridge_support_color)
            
            # Bridge pieces (red/orange when not placed, gray when placed)
            def draw_bridge_piece(i, img_inner):
                bridge_x = (self.consts.BRIDGE_LEFT_X + self.consts.BRIDGE_RIGHT_X) // 2 + (i - 1) * 12
                placed_x = jnp.where(state.bridge_pieces_placed[i], bridge_x, state.bridge_pieces_x[i])
                placed_y = jnp.where(state.bridge_pieces_placed[i], self.consts.BRIDGE_Y, state.bridge_pieces_y[i])
                placed_color = jnp.where(state.bridge_pieces_placed[i],
                                       jnp.array([200, 200, 200], dtype=jnp.uint8),  # Gray when placed
                                       jnp.array([255, 100, 0], dtype=jnp.uint8))  # Red-orange when not placed
                return draw_rect(img_inner, placed_x, placed_y, self.bridge_h, self.bridge_w, placed_color)
            
            for i in range(3):
                img = draw_bridge_piece(i, img)
            return img
        
        bridge_frame_match = (frame_idx == self.consts.BRIDGE_FRAME) & ~in_subway
        canvas = jax.lax.cond(
            bridge_frame_match,
            draw_bridge_area,
            lambda x: x,
            canvas
        )
        
        # --- 9. Draw Characters ---
        
        # Draw Superman (blue body, red cape) or Clark Kent (gray) or Weakened (red)
        def draw_superman(img, x, y, alt):
            # Body (blue)
            body_color = jnp.array([0, 0, 255], dtype=jnp.uint8)
            img = draw_rect(img, x, y - alt, 8, 6, body_color)
            # Cape (red)
            cape_color = jnp.array([255, 0, 0], dtype=jnp.uint8)
            img = draw_rect(img, x - 3, y - alt, 6, 2, cape_color)
            return img
        
        def draw_clark(img, x, y):
            # Gray suit
            clark_color = jnp.array([150, 150, 150], dtype=jnp.uint8)
            return draw_rect(img, x, y, 8, 6, clark_color)
        
        def draw_weakened(img, x, y):
            # Red (weakened)
            weak_color = jnp.array([255, 0, 0], dtype=jnp.uint8)
            return draw_rect(img, x, y, 8, 6, weak_color)
        
        # Draw player based on state
        def draw_player_superman(img):
            return draw_superman(img, state.player_x, state.player_y, state.player_altitude)
        
        def draw_player_clark(img):
            return draw_clark(img, state.player_x, state.player_y)
        
        def draw_player_weakened(img):
            return draw_weakened(img, state.player_x, state.player_y)
        
        canvas = jax.lax.cond(
            state.player_state == self.consts.PLAYER_SUPERMAN,
            draw_player_superman,
            lambda img: jax.lax.cond(
                state.player_state == self.consts.PLAYER_CLARK,
                draw_player_clark,
                draw_player_weakened,
                img
            ),
            canvas
        )
        
        # Draw Lex Luthor (green body, pink details) if not in jail
        def draw_lex_sprite(img):
            # Body (green)
            lex_body = jnp.array([0, 255, 0], dtype=jnp.uint8)
            img = draw_rect(img, state.lex_x, state.lex_y, 8, 6, lex_body)
            # Details (pink)
            lex_detail = jnp.array([255, 192, 203], dtype=jnp.uint8)
            img = draw_rect(img, state.lex_x, state.lex_y - 2, 2, 4, lex_detail)
            return img
        
        canvas = jax.lax.cond(
            ~state.lex_in_jail,
            lambda x: draw_lex_sprite(x),
            lambda x: x,
            canvas
        )
        
        # Draw Henchmen (orange body, pink details) if not in jail and in current frame
        def draw_henchman_sprite(i, img):
            hench_body = jnp.array([255, 165, 0], dtype=jnp.uint8)  # Orange
            hench_detail = jnp.array([255, 192, 203], dtype=jnp.uint8)  # Pink
            img = draw_rect(img, state.henchmen_x[i], state.henchmen_y[i], 6, 5, hench_body)
            img = draw_rect(img, state.henchmen_x[i], state.henchmen_y[i] - 1, 1, 3, hench_detail)
            return img
        
        for i in range(5):
            hench_in_frame = (state.henchmen_frames[i] == frame_idx) & ~in_subway
            canvas = jax.lax.cond(
                hench_in_frame & ~state.henchmen_in_jail[i],
                lambda x, idx=i: draw_henchman_sprite(idx, x),
                lambda x: x,
                canvas
            )
        
        # Draw Kryptonite Satellites (red, glowing) - only in current frame
        def draw_satellite_sprite(i, img):
            sat_color = jnp.array([255, 0, 0], dtype=jnp.uint8)  # Red
            sat_glow = jnp.array([255, 200, 200], dtype=jnp.uint8)  # Light red glow
            # Glow
            img = draw_rect(img, state.satellites_x[i], state.satellites_y[i], 8, 8, sat_glow)
            # Core
            img = draw_rect(img, state.satellites_x[i], state.satellites_y[i], 6, 6, sat_color)
            return img
        
        for i in range(3):
            sat_in_frame = (state.satellites_frames[i] == frame_idx) & ~in_subway
            canvas = jax.lax.cond(
                sat_in_frame & state.satellites_active[i],
                lambda x, idx=i: draw_satellite_sprite(idx, x),
                lambda x: x,
                canvas
            )
        
        # Draw Lois Lane (pink dress) - only in current frame
        def draw_lois_sprite(img):
            lois_color = jnp.array([255, 192, 203], dtype=jnp.uint8)  # Pink
            return draw_rect(img, state.lois_x, state.lois_y, 6, 5, lois_color)
        
        lois_in_frame = (state.lois_frame == frame_idx) & ~in_subway
        canvas = jax.lax.cond(
            lois_in_frame & ~state.lois_in_daily_planet,
            lambda x: draw_lois_sprite(x),
            lambda x: x,
            canvas
        )
        
        # Draw Bridge pieces - only in bridge frame
        def draw_bridge_piece(i, img):
            bridge_x = (self.consts.BRIDGE_LEFT_X + self.consts.BRIDGE_RIGHT_X) // 2 + (i - 1) * 12
            placed_x = jnp.where(state.bridge_pieces_placed[i], bridge_x, state.bridge_pieces_x[i])
            placed_y = jnp.where(state.bridge_pieces_placed[i], self.consts.BRIDGE_Y, state.bridge_pieces_y[i])
            placed_color = jnp.where(state.bridge_pieces_placed[i],
                                   jnp.array([200, 200, 200], dtype=jnp.uint8),  # Gray when placed
                                   jnp.array([255, 100, 0], dtype=jnp.uint8))  # Red-orange when not placed
            return draw_rect(img, placed_x, placed_y, self.bridge_h, self.bridge_w, placed_color)
        
        bridge_in_frame = (frame_idx == self.consts.BRIDGE_FRAME) & ~in_subway
        for i in range(3):
            piece_in_frame = (state.bridge_pieces_frames[i] == frame_idx) | state.bridge_pieces_placed[i]
            canvas = jax.lax.cond(
                bridge_in_frame & piece_in_frame,
                lambda x, idx=i: draw_bridge_piece(idx, x),
                lambda x: x,
                canvas
            )
        
        # Draw Helicopter (yellow body, black rotors) - only in current frame
        def draw_helicopter_sprite(img):
            heli_body = jnp.array([255, 255, 0], dtype=jnp.uint8)  # Yellow
            heli_rotor = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black
            # Body
            img = draw_rect(img, state.helicopter_x, state.helicopter_y, 6, 12, heli_body)
            # Rotors (horizontal bar on top)
            img = draw_rect(img, state.helicopter_x, state.helicopter_y - 4, 1, 16, heli_rotor)
            return img
        
        heli_in_frame = (state.helicopter_frame == frame_idx) & ~in_subway
        canvas = jax.lax.cond(
            heli_in_frame,
            lambda x: draw_helicopter_sprite(x),
            lambda x: x,
            canvas
        )

        return canvas
