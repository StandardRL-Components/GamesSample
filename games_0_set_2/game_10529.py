import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:34:32.941858
# Source Brief: brief_00529.md
# Brief Index: 529
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Constants ---

# Screen and Grid
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
GRID_COLS, GRID_ROWS = 16, 10
CELL_SIZE = 40  # 640/16 = 40, 400/10 = 40

# Colors (Minimalist, High Contrast)
COLOR_BG = (34, 40, 49)          # Dark Slate Gray
COLOR_GRID = (57, 62, 70)        # Lighter Gray
COLOR_TEXT = (238, 238, 238)     # Off-white
COLOR_CURSOR = (0, 173, 181, 150) # Cyan, semi-transparent
COLOR_DOMINO = (238, 238, 238)   # Off-white
COLOR_DOMINO_TIMER = (255, 173, 64) # Orange for timer warning
COLOR_DOMINO_FALLING = (214, 90, 49) # Red/Orange
COLOR_DOMINO_CHAIN = (155, 197, 61) # Green
COLOR_DOMINO_FALLEN = (80, 80, 80) # Dark Gray

# Game Parameters
DOMINO_WIDTH = 18
DOMINO_HEIGHT = 36
FPS = 30
MAX_STEPS = 1000
WIN_SCORE = 1000
FALL_ANIMATION_SPEED = 0.1 # Progress per frame, 10 frames to fall
DOMINO_MAX_TIMER = 3.0 # seconds
DOMINO_MIN_TIMER = 1.0 # seconds

# --- Domino Class ---

class Domino:
    def __init__(self, grid_x, grid_y, angle, seed_rng):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = (grid_x + 0.5) * CELL_SIZE
        self.y = (grid_y + 0.5) * CELL_SIZE
        
        self.angle = angle # 0:Up, 90:Left, 180:Down, 270:Right
        self.state = 'upright' # upright, falling, fallen
        
        self.fall_timer = seed_rng.uniform(DOMINO_MIN_TIMER, DOMINO_MAX_TIMER) * FPS
        self.fall_progress = 0.0 # 0.0 to 1.0
        self.fall_direction = 1 # Can be changed by impact direction
        
        self.color = COLOR_DOMINO
        self.is_chain_reaction = False

    def update(self):
        if self.state == 'upright':
            self.fall_timer -= 1
            if self.fall_timer <= 0:
                self.start_fall()
                # SFX: domino_wobble_and_fall
        elif self.state == 'falling':
            self.fall_progress += FALL_ANIMATION_SPEED
            if self.fall_progress >= 1.0:
                self.fall_progress = 1.0
                self.state = 'fallen'
                self.color = COLOR_DOMINO_FALLEN

    def start_fall(self):
        if self.state == 'upright':
            self.state = 'falling'
            if not self.is_chain_reaction:
                self.color = COLOR_DOMINO_FALLING
            # SFX: domino_swoosh

    def trigger_fall(self, impact_from_domino):
        if self.state == 'upright':
            self.is_chain_reaction = True
            self.color = COLOR_DOMINO_CHAIN # Flash green
            
            # Basic logic to determine fall direction
            dx = self.x - impact_from_domino.x
            dy = self.y - impact_from_domino.y
            
            # Determine fall direction based on which quadrant the impact came from relative to orientation
            if self.angle == 0 or self.angle == 180: # Up or Down
                self.fall_direction = 1 if dx < 0 else -1
            else: # Left or Right
                self.fall_direction = 1 if dy < 0 else -1
            
            self.start_fall()
            return True
        return False

    def get_rotated_points(self):
        w, h = DOMINO_WIDTH / 2, DOMINO_HEIGHT
        
        # Define corners relative to a center pivot at (0,0)
        points = [
            (-w, -h/2), (w, -h/2), (w, h/2), (-w, h/2)
        ]
        
        # Apply fall animation rotation around the base
        fall_rad = math.radians(self.fall_progress * 90 * self.fall_direction)
        cos_fall, sin_fall = math.cos(fall_rad), math.sin(fall_rad)
        
        pivot_y = h/2
        
        rotated_points = []
        for x, y in points:
            y_shifted = y - pivot_y
            rx = x * cos_fall - y_shifted * sin_fall
            ry = x * sin_fall + y_shifted * cos_fall
            rotated_points.append((rx, ry + pivot_y))
            
        # Apply main orientation rotation
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        final_points = []
        for x, y in rotated_points:
            rx = self.x + (x * cos_a - y * sin_a)
            ry = self.y + (x * sin_a + y * cos_a)
            final_points.append((int(rx), int(ry)))
            
        return final_points

    def get_tip_position(self):
        if self.state != 'falling':
            return None
        
        w, h = DOMINO_WIDTH / 2, DOMINO_HEIGHT
        tip_local_x, tip_local_y = 0, -h/2
        
        fall_rad = math.radians(self.fall_progress * 90 * self.fall_direction)
        cos_fall, sin_fall = math.cos(fall_rad), math.sin(fall_rad)
        pivot_y = h/2
        
        y_shifted = tip_local_y - pivot_y
        rx_fall = tip_local_x * cos_fall - y_shifted * sin_fall
        ry_fall = tip_local_x * sin_fall + y_shifted * cos_fall
        
        rx_fall, ry_fall = rx_fall, ry_fall + pivot_y
        
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rx_final = self.x + (rx_fall * cos_a - ry_fall * sin_a)
        ry_final = self.y + (rx_fall * sin_a + ry_fall * cos_a)
        
        return (rx_final, ry_final)

    def get_bounding_box(self):
        points = self.get_rotated_points()
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def draw(self, surface):
        points = self.get_rotated_points()
        
        current_color = self.color
        if self.state == 'upright':
            timer_ratio = max(0, self.fall_timer / (DOMINO_MAX_TIMER * FPS))
            current_color = tuple(
                int(c1 + (c2 - c1) * (1 - timer_ratio)) 
                for c1, c2 in zip(COLOR_DOMINO, COLOR_DOMINO_TIMER)
            )
        elif self.is_chain_reaction and self.state == 'falling' and self.fall_progress < 0.2:
            current_color = COLOR_DOMINO_CHAIN
        elif self.state == 'falling':
            current_color = COLOR_DOMINO_FALLING
            
        pygame.gfxdraw.aapolygon(surface, points, current_color)
        pygame.gfxdraw.filled_polygon(surface, points, current_color)

# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}
    
    game_description = (
        "Strategically place dominos on a grid to create the longest possible chain reaction. "
        "Each domino has a timer; when it expires, the domino falls, potentially toppling others."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a domino and shift to rotate it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dominoes = []
        self.cursor_pos = [0, 0]
        self.cursor_angle = 0
        self.last_space_held = False
        self.last_shift_held = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dominoes = []
        self.cursor_pos = [GRID_COLS // 2, GRID_ROWS // 2]
        self.cursor_angle = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        placement_reward = self._handle_input(action)
        chain_reward = self._update_game_state()
        
        reward = placement_reward + chain_reward
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, GRID_ROWS - 1)

        # Rotate (on press)
        if shift_held and not self.last_shift_held:
            self.cursor_angle = (self.cursor_angle + 90) % 360
            # SFX: cursor_rotate

        # Place Domino (on press)
        if space_held and not self.last_space_held:
            can_place = True
            new_domino_box = pygame.Rect(
                self.cursor_pos[0] * CELL_SIZE, self.cursor_pos[1] * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            for d in self.dominoes:
                domino_box = pygame.Rect(
                    d.grid_x * CELL_SIZE, d.grid_y * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                if new_domino_box.colliderect(domino_box):
                    can_place = False
                    break
            
            if can_place:
                domino = Domino(self.cursor_pos[0], self.cursor_pos[1], self.cursor_angle, self.np_random)
                self.dominoes.append(domino)
                reward += 10
                self.score += 10 # Score for placement
                # SFX: place_domino
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return reward

    def _update_game_state(self):
        chain_reward = 0
        
        for d in self.dominoes:
            d.update()

        falling_dominoes = [d for d in self.dominoes if d.state == 'falling']
        upright_dominoes = [d for d in self.dominoes if d.state == 'upright']

        for d_fall in falling_dominoes:
            tip_pos = d_fall.get_tip_position()
            if not tip_pos: continue

            for d_up in upright_dominoes:
                if d_fall == d_up: continue
                
                if abs(d_fall.x - d_up.x) > DOMINO_HEIGHT or abs(d_fall.y - d_up.y) > DOMINO_HEIGHT:
                    continue
                
                if d_up.get_bounding_box().collidepoint(tip_pos):
                    if d_up.trigger_fall(d_fall):
                        chain_reward += 1
                        self.score += 1
                        # SFX: domino_click
        
        return chain_reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0

        if self.score >= WIN_SCORE:
            terminated = True
            terminal_reward = 100
        
        if self.steps >= MAX_STEPS:
            terminated = True
            
        if len(self.dominoes) > 0 and all(d.state == 'fallen' for d in self.dominoes):
            terminated = True
        
        return terminated, terminal_reward

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_frame(self):
        self.screen.fill(COLOR_BG)
        
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y))

        for d in self.dominoes:
            d.draw(self.screen)
            
        if not self.game_over:
            self._render_cursor()

        score_text = self.font.render(f"SCORE: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"STEPS: {self.steps}/{MAX_STEPS}", True, COLOR_TEXT)
        self.screen.blit(steps_text, (SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    def _render_cursor(self):
        preview_domino = Domino(self.cursor_pos[0], self.cursor_pos[1], self.cursor_angle, self.np_random)
        points = preview_domino.get_rotated_points()
        
        pygame.gfxdraw.aapolygon(self.screen, points, COLOR_CURSOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, COLOR_CURSOR)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging
    # It will not be executed by the test suite
    try:
        # Check if we are in a headless environment
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            raise RuntimeError("Cannot run human player in a headless environment")

        env = GameEnv()
        obs, info = env.reset()
        
        pygame.display.set_caption("Domino Chain Reaction")
        human_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        running = True
        while running:
            action = [0, 0, 0] # Default to no-op
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Handle key presses for single-press actions (like rotation)
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                        action[2] = 1
                    if event.key == pygame.K_SPACE:
                        action[1] = 1

            # Key holds for continuous actions (like movement)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Since space and shift are handled on press in the env, we need to manage held state here
            # to only send the action on the first frame it's pressed.
            # However, the environment logic already handles this with `last_space_held`
            # So, we can just send the raw key state.
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                obs, info = env.reset()
            
            # Render the game screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(FPS)
            
        env.close()
    except Exception as e:
        print(f"Could not run human-playable demo: {e}")
        # Create a dummy env to validate the implementation without rendering
        print("Running headless validation...")
        env = GameEnv()
        obs, info = env.reset()
        assert obs is not None
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs is not None
        env.close()
        print("Headless validation successful.")