import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:34:59.663700
# Source Brief: brief_00495.md
# Brief Index: 495
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Cosmic Bowl: A zero-G bowling game with a tile-matching puzzle mechanic.

    Game Phases:
    1. TILE_MATCHING: Match pairs of colored tiles to earn 'Gravity Charge'.
    2. AIMING: Use charge to select a gravity modifier, aim the ball, and apply spin.
    3. ROLLING: Watch the ball roll in real-time under the influence of physics.
    4. SCORING: Tally points, check for strikes/spares, and set up the next roll/frame.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A futuristic zero-G bowling game where you solve tile puzzles to unlock powerful gravity-bending shots."
    )
    user_guide = (
        "Controls: Use ←→ to aim, ↑↓ to apply spin, and space to launch. "
        "In the puzzle phase, use arrow keys to move and space to select. Use shift to cycle special abilities."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors (Neon on Dark)
    COLOR_BG = (10, 5, 30) # #0a051e
    COLOR_LANE_EDGE = (70, 220, 240, 50) # #46dcf0 with alpha
    COLOR_LANE_GRID = (44, 33, 92, 100) # #2c215c with alpha
    COLOR_BALL = (240, 240, 70) # #f0f046
    COLOR_BALL_GLOW = (240, 240, 70, 50)
    COLOR_PIN = (230, 230, 255)
    COLOR_PIN_HEAD = (70, 220, 240)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_REWARD_PLUS = (100, 255, 100)
    COLOR_REWARD_MINUS = (255, 100, 100)
    TILE_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80), (255, 80, 255)]

    # Game Phases
    PHASE_TILE_MATCHING = "TILE_MATCHING"
    PHASE_AIMING = "AIMING"
    PHASE_ROLLING = "ROLLING"
    PHASE_SCORING = "SCORING"

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Game state variables
        self.game_phase = None
        self.ball = {}
        self.pins = []
        self.particles = []
        self.stars = []
        self.ball_trail = []
        
        # Bowling state
        self.current_frame = 0
        self.current_roll = 0
        self.pins_down_this_frame = []
        self.frame_scores = []

        # Tile matching state
        self.tile_grid = []
        self.tile_cursor = [0, 0]
        self.selected_tile = None
        self.tile_phase_timer = 0
        self.gravity_charge = 0
        
        # Aiming state
        self.aim_pos = 0.5 # 0 to 1 across the lane
        self.aim_spin = 0 # -1, 0, 1
        self.selected_gravity_mod = 0 # 0: None, 1: Well, 2: Boost

        # Action state
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Floating text for rewards
        self.reward_popups = []

        # self.reset() is called by the wrapper, but we need an RNG for _create_stars
        super().reset(seed=random.randint(0, 1_000_000_000))
        self._create_stars()
        # self.validate_implementation() # this is for dev, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.reward_popups = []
        
        # Bowling state
        self.current_frame = 1
        self.current_roll = 1
        self.frame_scores = []
        
        # Player resources
        self.gravity_charge = 1
        self.selected_gravity_mod = 0
        
        self._setup_new_frame()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        reward = 0
        self.steps += 1
        
        self._update_particles_and_popups()

        if self.game_phase == self.PHASE_TILE_MATCHING:
            reward += self._update_tile_matching(movement, space_pressed)
        elif self.game_phase == self.PHASE_AIMING:
            self._update_aiming(movement, space_pressed, shift_pressed)
        elif self.game_phase == self.PHASE_ROLLING:
            self._update_rolling()
        elif self.game_phase == self.PHASE_SCORING:
            reward += self._update_scoring()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        terminated = self.steps >= self.MAX_STEPS or self.current_frame > 10
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # --- Game Logic Updates ---

    def _update_tile_matching(self, movement, space_pressed):
        reward = 0
        self.tile_phase_timer -= 1

        # Move cursor
        if movement == 1: self.tile_cursor[1] = max(0, self.tile_cursor[1] - 1) # Up
        elif movement == 2: self.tile_cursor[1] = min(3, self.tile_cursor[1] + 1) # Down
        elif movement == 3: self.tile_cursor[0] = max(0, self.tile_cursor[0] - 1) # Left
        elif movement == 4: self.tile_cursor[0] = min(3, self.tile_cursor[0] + 1) # Right

        if space_pressed:
            x, y = self.tile_cursor
            if self.tile_grid[y][x] is not None:
                if self.selected_tile is None:
                    self.selected_tile = (x, y)
                else:
                    sel_x, sel_y = self.selected_tile
                    if (sel_x, sel_y) != (x, y) and self.tile_grid[y][x] == self.tile_grid[sel_y][sel_x]:
                        # Match found
                        self.gravity_charge = min(5, self.gravity_charge + 1)
                        self.tile_grid[y][x] = None
                        self.tile_grid[sel_y][sel_x] = None
                        reward += 1.0
                        self._create_reward_popup("+1 Charge!", self.COLOR_REWARD_PLUS)
                        # sound: match_success.wav
                    else:
                        # Mismatch
                        reward -= 0.1
                        self._create_reward_popup("Mismatch", self.COLOR_REWARD_MINUS)
                        # sound: match_fail.wav
                    self.selected_tile = None
        
        if self.tile_phase_timer <= 0:
            self.game_phase = self.PHASE_AIMING

        return reward

    def _update_aiming(self, movement, space_pressed, shift_pressed):
        # Move aim
        if movement == 3: self.aim_pos = max(0.05, self.aim_pos - 0.02)
        elif movement == 4: self.aim_pos = min(0.95, self.aim_pos + 0.02)
        
        # Cycle spin
        if movement == 1: self.aim_spin = 1 # "Up" for right spin
        elif movement == 2: self.aim_spin = -1 # "Down" for left spin
        
        # Cycle gravity modifier
        if shift_pressed and self.gravity_charge > 0:
            self.selected_gravity_mod = (self.selected_gravity_mod + 1) % 3
            # sound: cycle_powerup.wav
        
        if space_pressed:
            self.ball['vel'] = [0, 20] # Initial forward velocity
            self.ball['spin'] = self.aim_spin * 0.1
            
            if self.selected_gravity_mod != 0 and self.gravity_charge > 0:
                self.gravity_charge -= 1
            else:
                self.selected_gravity_mod = 0 # Ensure no effect if no charge
                
            self.game_phase = self.PHASE_ROLLING
            # sound: ball_launch.wav

    def _update_rolling(self):
        # Apply spin force
        self.ball['pos'][0] += self.ball['spin']
        
        # Apply gravity modifier
        if self.selected_gravity_mod == 1: # Gravity Well
            well_pos = [self.WIDTH / 2, self.HEIGHT / 2 + 50]
            dist_x = well_pos[0] - self.ball['pos'][0]
            dist_y = well_pos[1] - self.ball['pos'][1]
            dist = math.hypot(dist_x, dist_y)
            if dist > 1:
                force = 50 / dist
                self.ball['vel'][0] += force * (dist_x / dist) * 0.1
                self.ball['vel'][1] += force * (dist_y / dist) * 0.1
        elif self.selected_gravity_mod == 2: # Boost
            self.ball['vel'][1] *= 1.02
        
        # Update position
        self.ball['pos'][0] += self.ball['vel'][0]
        self.ball['pos'][1] += self.ball['vel'][1]
        
        # Friction
        self.ball['vel'][0] *= 0.98
        self.ball['vel'][1] *= 0.98
        self.ball['spin'] *= 0.99
        
        # Add to trail
        self.ball_trail.append(list(self.ball['pos']))
        if len(self.ball_trail) > 15:
            self.ball_trail.pop(0)

        # Gutter check
        lane_width_at_depth = self._get_lane_width_at(self.ball['pos'][1])
        center_x = self.WIDTH / 2
        if abs(self.ball['pos'][0] - center_x) > lane_width_at_depth / 2:
            self.ball['active'] = False
            self.ball['is_gutter'] = True
            self.game_phase = self.PHASE_SCORING
            # sound: gutter.wav
            return

        # Pin collision
        ball_radius = self._project_size(15, self.ball['pos'][1])
        for pin in self.pins:
            if pin['standing']:
                pin_pos_2d = self._project_pos(pin['pos'])
                dist = math.hypot(self.ball['pos'][0] - pin_pos_2d[0], self.ball['pos'][1] - pin_pos_2d[1])
                if dist < ball_radius + 5:
                    pin['standing'] = False
                    self.ball['vel'][0] *= 0.8
                    self.ball['vel'][1] *= 0.8
                    self._create_particles(pin_pos_2d, 20, self.COLOR_PIN)
                    # sound: pin_hit.wav

        # Check if ball stopped
        if math.hypot(*self.ball['vel']) < 0.1:
            self.ball['active'] = False
            self.game_phase = self.PHASE_SCORING

    def _update_scoring(self):
        reward = 0
        if self.ball['is_gutter']:
            reward -= 5
            self._create_reward_popup("Gutter Ball", self.COLOR_REWARD_MINUS)
        
        pins_down_this_roll = 0
        current_pins_down = []
        for i, pin in enumerate(self.pins):
            if not pin['standing'] and i not in self.pins_down_this_frame:
                pins_down_this_roll += 1
                current_pins_down.append(i)
        
        self.pins_down_this_frame.extend(current_pins_down)
        
        reward += pins_down_this_roll * 10
        self.score += pins_down_this_roll
        self.frame_scores.append(pins_down_this_roll)

        is_strike = self.current_roll == 1 and len(self.pins_down_this_frame) == 10
        is_spare = self.current_roll == 2 and len(self.pins_down_this_frame) == 10

        if is_strike:
            reward += 50
            self.score += 20 # Bonus for strike
            self._create_reward_popup("STRIKE!", self.COLOR_REWARD_PLUS)
            self.current_frame += 1
            self._setup_new_frame()
        elif is_spare:
            reward += 25
            self.score += 10 # Bonus for spare
            self._create_reward_popup("SPARE!", self.COLOR_REWARD_PLUS)
            self.current_frame += 1
            self._setup_new_frame()
        else:
            if self.current_roll == 1:
                self.current_roll = 2
                self._setup_new_roll()
            else:
                self.current_frame += 1
                self._setup_new_frame()
        
        return reward

    def _update_particles_and_popups(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

        self.reward_popups = [r for r in self.reward_popups if r['life'] > 0]
        for r in self.reward_popups:
            r['pos'][1] -= 0.5
            r['life'] -= 1

    # --- Setup ---

    def _setup_new_frame(self):
        self.current_roll = 1
        self.pins_down_this_frame = []
        self._setup_pins()
        self._setup_new_roll()
        self._setup_tile_board()
        self.game_phase = self.PHASE_TILE_MATCHING
        self.tile_phase_timer = 5 * self.FPS

    def _setup_new_roll(self):
        self.ball = {
            'pos': [self.WIDTH / 2, 50],
            'vel': [0, 0],
            'spin': 0,
            'active': True,
            'is_gutter': False,
        }
        self.ball_trail = []
        self.aim_pos = 0.5
        self.aim_spin = 0
        self.selected_gravity_mod = 0
        self.game_phase = self.PHASE_AIMING
        if self.current_roll == 1: # Only do tile matching on first roll
            self.game_phase = self.PHASE_TILE_MATCHING
            self.tile_phase_timer = 5 * self.FPS
            self._setup_tile_board()

    def _setup_pins(self):
        self.pins = []
        layout = [(0, 0), (-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]
        for i, (dx, dy) in enumerate(layout):
            self.pins.append({
                'id': i,
                'pos': [dx * 15, 300 + dy * 20],
                'standing': True
            })

    def _setup_tile_board(self):
        colors = self.TILE_COLORS * 4
        self.np_random.shuffle(colors)
        self.tile_grid = [colors[i*4:(i+1)*4] for i in range(4)]
        self.tile_cursor = [0, 0]
        self.selected_tile = None

    def _create_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                'size': self.np_random.uniform(0.5, 2),
                'speed': self.np_random.uniform(0.1, 0.3)
            })

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_lane()
        self._render_pins()
        self._render_ball_and_effects()
        self._render_particles()

        if self.game_phase == self.PHASE_TILE_MATCHING:
            self._render_tile_board()
        elif self.game_phase == self.PHASE_AIMING:
            self._render_aim_guide()
        
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            star['pos'][1] = (star['pos'][1] + star['speed']) % self.HEIGHT
            pygame.draw.circle(self.screen, self.COLOR_TEXT, star['pos'], star['size'])

    def _render_lane(self):
        # Vanishing point perspective
        vp_y = 120
        for i in range(21):
            y = 50 + i * (self.HEIGHT - 50) / 20
            width = self._get_lane_width_at(y)
            x1 = self.WIDTH / 2 - width / 2
            x2 = self.WIDTH / 2 + width / 2
            
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLOR_LANE_GRID, (x1, y), (x2, y), 1)

        # Vertical lines
        for i in range(6):
            px = i / 5.0
            p1 = self._project_pos([(px - 0.5) * 120, 50])
            p2 = self._project_pos([(px - 0.5) * 120, self.HEIGHT])
            pygame.draw.aaline(self.screen, self.COLOR_LANE_GRID, p1, p2)

    def _render_pins(self):
        sorted_pins = sorted(self.pins, key=lambda p: p['pos'][1])
        for pin in sorted_pins:
            if pin['standing']:
                pos_2d = self._project_pos(pin['pos'])
                size = self._project_size(30, pos_2d[1])
                self._draw_pin(pos_2d, size)

    def _render_ball_and_effects(self):
        if not self.ball.get('active', False): return
        
        # Trail
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = (i / len(self.ball_trail)) * 100
                size = self._project_size(15, pos[1])
                color = (*self.COLOR_BALL, int(alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(size), color)

        # Ball
        size = self._project_size(15, self.ball['pos'][1])
        self._draw_glowing_circle(self.ball['pos'], size, self.COLOR_BALL, self.COLOR_BALL_GLOW)
        
        # Gravity effect
        if self.selected_gravity_mod == 1 and self.game_phase == self.PHASE_ROLLING:
            well_pos = [self.WIDTH / 2, self.HEIGHT / 2 + 50]
            radius = 50 + math.sin(self.steps * 0.2) * 5
            pygame.gfxdraw.aacircle(self.screen, int(well_pos[0]), int(well_pos[1]), int(radius), (0, 100, 255, 150))
            pygame.gfxdraw.aacircle(self.screen, int(well_pos[0]), int(well_pos[1]), int(radius-2), (0, 100, 255, 100))

    def _render_tile_board(self):
        board_w, board_h = 200, 200
        start_x = (self.WIDTH - board_w) / 2
        start_y = (self.HEIGHT - board_h) / 2
        tile_size = 48
        
        s = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
        s.fill((20, 10, 50, 200))
        pygame.draw.rect(s, self.COLOR_LANE_EDGE, s.get_rect(), 2, 5)

        for r in range(4):
            for c in range(4):
                color = self.tile_grid[r][c]
                rect = (c * tile_size + 2, r * tile_size + 2, tile_size - 4, tile_size - 4)
                if color:
                    pygame.draw.rect(s, color, rect, 0, 3)
        
        # Cursor
        cur_x, cur_y = self.tile_cursor
        cursor_rect = (cur_x * tile_size, cur_y * tile_size, tile_size, tile_size)
        pygame.draw.rect(s, self.COLOR_BALL, cursor_rect, 3, 3)

        # Selected
        if self.selected_tile:
            sel_x, sel_y = self.selected_tile
            sel_rect = (sel_x * tile_size, sel_y * tile_size, tile_size, tile_size)
            pygame.draw.rect(s, self.COLOR_TEXT, sel_rect, 3, 3)

        self.screen.blit(s, (start_x, start_y))
        
        timer_text = f"TIME: {self.tile_phase_timer / self.FPS:.1f}"
        self._draw_text(timer_text, (self.WIDTH / 2, start_y - 20), self.font_m, self.COLOR_TEXT)


    def _render_aim_guide(self):
        start_pos = [self.WIDTH * self.aim_pos, 50]
        end_pos = [self.WIDTH / 2, self.HEIGHT]
        
        # Simple line for now
        pygame.draw.aaline(self.screen, self.COLOR_BALL_GLOW, start_pos, (start_pos[0], 100))
        
        # Spin indicator
        spin_text = "SPIN: " + ("RIGHT" if self.aim_spin > 0 else "LEFT" if self.aim_spin < 0 else "NONE")
        self._draw_text(spin_text, (self.WIDTH / 2, 20), self.font_s, self.COLOR_TEXT)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, (p['life'] / p['max_life']) * 255))
            color = (*p['color'], alpha)
            size = max(1, p['size'] * (p['life'] / p['max_life']))
            pygame.draw.circle(self.screen, color, p['pos'], size)

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (10, 10), self.font_m, self.COLOR_TEXT, align="topleft")
        # Frame
        self._draw_text(f"FRAME: {self.current_frame}/10", (self.WIDTH - 10, 10), self.font_m, self.COLOR_TEXT, align="topright")
        self._draw_text(f"ROLL: {self.current_roll}", (self.WIDTH - 10, 40), self.font_s, self.COLOR_TEXT, align="topright")
        # Gravity Charge
        self._draw_text(f"GRAVITY CHARGE: {self.gravity_charge}", (10, 40), self.font_s, self.COLOR_TEXT, align="topleft")
        
        # Selected Gravity Modifier
        mod_name = ["NONE", "GRAVITY WELL", "BOOST"][self.selected_gravity_mod]
        mod_color = self.COLOR_TEXT if self.gravity_charge > 0 and self.selected_gravity_mod != 0 else (100,100,100)
        self._draw_text(f"MOD: {mod_name}", (self.WIDTH/2, self.HEIGHT - 20), self.font_m, mod_color)

        # Reward popups
        for r in self.reward_popups:
            alpha = max(0, min(255, (r['life'] / r['max_life']) * 255))
            color = (*r['color'], alpha)
            self._draw_text(r['text'], r['pos'], self.font_m, color)

    # --- Helpers & Drawing ---

    def _get_lane_width_at(self, y):
        start_y, end_y = 50, self.HEIGHT
        start_w, end_w = self.WIDTH * 0.8, self.WIDTH * 0.2
        progress = (y - start_y) / (end_y - start_y)
        return start_w + (end_w - start_w) * progress**1.5
    
    def _project_pos(self, pos_3d):
        # pos_3d: [x_offset_from_center, y_depth]
        y = pos_3d[1]
        width_at_y = self._get_lane_width_at(y)
        max_x_offset = self._get_lane_width_at(50) * 0.5
        x = self.WIDTH/2 + (pos_3d[0] / max_x_offset) * (width_at_y / 2)
        return [x, y]

    def _project_size(self, base_size, y):
        start_y, end_y = 50, self.HEIGHT
        start_s, end_s = 1.0, 0.3
        progress = max(0, min(1, (y - start_y) / (end_y - start_y)))
        scale = start_s + (end_s - start_s) * progress
        return base_size * scale

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius * 1.8), glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

    def _draw_pin(self, pos, size):
        x, y = int(pos[0]), int(pos[1])
        h = int(size)
        w = int(size * 0.3)
        pygame.draw.polygon(self.screen, self.COLOR_PIN, [
            (x - w/2, y), (x + w/2, y), (x + w/3, y - h*0.8), (x, y - h), (x - w/3, y - h*0.8)
        ])
        pygame.draw.circle(self.screen, self.COLOR_PIN_HEAD, (x, y - h*0.9), w/3)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })
            
    def _create_reward_popup(self, text, color):
        self.reward_popups.append({
            'text': text,
            'pos': [self.WIDTH / 2, self.HEIGHT / 2],
            'color': color,
            'life': 45,
            'max_life': 45
        })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "frame": self.current_frame, "roll": self.current_roll}
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play example
    # This block is not used for evaluation but is helpful for testing.
    # It needs a display, so we unset the dummy driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cosmic Bowl")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- CONTROLS ---")
    print(GameEnv.user_guide)
    print("----------------\n")

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
        if terminated or truncated:
            print("Episode finished.")
            # Optional: reset and play again
            # obs, info = env.reset()
            # terminated = False
            # total_reward = 0

    env.close()