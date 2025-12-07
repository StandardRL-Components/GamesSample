
# Generated: 2025-08-28T02:24:37.875727
# Source Brief: brief_01693.md
# Brief Index: 1693

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing strings ---
    user_guide = (
        "Controls: Arrows to move cursor, Shift to cycle gates, Space to place a gate."
    )
    game_description = (
        "Guide a rolling ball through a labyrinth by placing logic gates to alter its path. Reach the green goal!"
    )

    # --- Game Configuration ---
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_AREA_SIZE = 380
    CELL_SIZE = GRID_AREA_SIZE // GRID_SIZE
    GRID_ORIGIN_X = (WIDTH - GRID_AREA_SIZE) // 2
    GRID_ORIGIN_Y = (HEIGHT - GRID_AREA_SIZE) // 2
    
    # Colors
    COLOR_BG = (32, 32, 32)
    COLOR_GRID = (64, 64, 64)
    COLOR_BALL = (255, 255, 255)
    COLOR_START = (255, 64, 64)
    COLOR_GOAL = (64, 255, 64)
    COLOR_TEXT = (220, 220, 220)
    
    # Gate definitions
    GATE_NONE = 0
    GATE_AND = 1   # Pass-through
    GATE_OR = 2    # 90-deg right turn
    GATE_NOT = 3   # 180-deg reverse
    GATE_XOR = 4   # 90-deg left turn
    GATE_DELAY = 5 # Pause

    GATE_COLORS = {
        GATE_AND: (0, 150, 255),    # Blue
        GATE_OR: (255, 200, 0),     # Yellow
        GATE_NOT: (255, 80, 80),    # Red
        GATE_XOR: (200, 80, 255),   # Purple
        GATE_DELAY: (255, 140, 0),  # Orange
    }
    GATE_NAMES = {
        GATE_AND: "AND",
        GATE_OR: "OR",
        GATE_NOT: "NOT",
        GATE_XOR: "XOR",
        GATE_DELAY: "DELAY",
    }
    
    # Physics
    BALL_SPEED = 2.0  # Pixels per step
    MAX_STEPS = 1500
    DELAY_DURATION = 30 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24)
        
        # State variables initialized in reset()
        self.grid = None
        self.start_pos = None
        self.goal_pos = None
        self.ball_pixel_pos = None
        self.ball_grid_pos = None
        self.ball_vel = None
        self.cursor_pos = None
        self.gate_types = list(self.GATE_COLORS.keys())
        self.selected_gate_idx = None
        self.gate_inventory = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.delay_timer = None

        self.reset()
        # self.validate_implementation() # Call this to check your implementation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        # Place start and goal
        self.start_pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
        self.goal_pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
        while math.dist(self.start_pos, self.goal_pos) < self.GRID_SIZE / 2:
            self.goal_pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))

        # Ball state
        self.ball_grid_pos = list(self.start_pos)
        self.ball_pixel_pos = self._grid_to_pixel(self.ball_grid_pos)
        
        # Initial velocity (not pointing directly off-screen)
        possible_vels = []
        if self.start_pos[0] > 0: possible_vels.append([-1, 0])
        if self.start_pos[0] < self.GRID_SIZE - 1: possible_vels.append([1, 0])
        if self.start_pos[1] > 0: possible_vels.append([0, -1])
        if self.start_pos[1] < self.GRID_SIZE - 1: possible_vels.append([0, 1])
        vel_idx = self.np_random.integers(0, len(possible_vels))
        self.ball_vel = possible_vels[vel_idx]

        # Player state
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gate_idx = 0
        self.gate_inventory = {
            self.GATE_AND: 5,
            self.GATE_OR: 5,
            self.GATE_NOT: 5,
            self.GATE_XOR: 5,
            self.GATE_DELAY: 3,
        }

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.delay_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Survival reward

        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cycle selected gate on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_gate_idx = (self.selected_gate_idx + 1) % len(self.gate_types)
            # sfx: UI_switch_sound
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Place gate on space press
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            if self.grid[y, x] == self.GATE_NONE and tuple(self.cursor_pos) != self.start_pos and tuple(self.cursor_pos) != self.goal_pos:
                gate_to_place = self.gate_types[self.selected_gate_idx]
                if self.gate_inventory[gate_to_place] > 0:
                    self.grid[y, x] = gate_to_place
                    self.gate_inventory[gate_to_place] -= 1
                    # sfx: place_gate_sound

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- 2. Update Game Logic ---
        # Handle delay gate
        if self.delay_timer > 0:
            self.delay_timer -= 1
        else:
            # Move ball towards next grid center
            target_pixel_pos = self._grid_to_pixel([
                self.ball_grid_pos[0] + self.ball_vel[0],
                self.ball_grid_pos[1] + self.ball_vel[1]
            ])
            
            # Calculate vector to target and move
            direction_vec = [target_pixel_pos[0] - self.ball_pixel_pos[0], target_pixel_pos[1] - self.ball_pixel_pos[1]]
            dist_to_target = math.hypot(*direction_vec)

            if dist_to_target < self.BALL_SPEED:
                # Reached intersection
                self.ball_pixel_pos = target_pixel_pos
                self.ball_grid_pos = [self.ball_grid_pos[0] + self.ball_vel[0], self.ball_grid_pos[1] + self.ball_vel[1]]
                
                # Check for termination conditions
                if not (0 <= self.ball_grid_pos[0] < self.GRID_SIZE and 0 <= self.ball_grid_pos[1] < self.GRID_SIZE):
                    self.game_over = True
                    reward -= 100
                    # sfx: fall_off_sound
                elif tuple(self.ball_grid_pos) == self.goal_pos:
                    self.game_over = True
                    reward += 100
                    self.score += 100
                    # sfx: win_sound
                else:
                    # Interact with gate
                    gate_type = self.grid[self.ball_grid_pos[1], self.ball_grid_pos[0]]
                    if gate_type != self.GATE_NONE:
                        reward += self._apply_gate_logic(gate_type)
            else:
                # Move smoothly towards target
                norm_vec = [direction_vec[0] / dist_to_target, direction_vec[1] / dist_to_target]
                self.ball_pixel_pos[0] += norm_vec[0] * self.BALL_SPEED
                self.ball_pixel_pos[1] += norm_vec[1] * self.BALL_SPEED
        
        # --- 3. Finalize Step ---
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _apply_gate_logic(self, gate_type):
        """Applies gate logic to ball velocity and returns shaping reward."""
        dist_before = math.dist(self.ball_grid_pos, self.goal_pos)
        vx, vy = self.ball_vel
        
        if gate_type == self.GATE_AND: # Pass-through
            pass # sfx: pass_through_sound
        elif gate_type == self.GATE_OR: # 90-deg right
            self.ball_vel = [vy, -vx] # sfx: turn_sound
        elif gate_type == self.GATE_NOT: # 180-deg reverse
            self.ball_vel = [-vx, -vy] # sfx: reverse_sound
        elif gate_type == self.GATE_XOR: # 90-deg left
            self.ball_vel = [-vy, vx] # sfx: turn_sound
        elif gate_type == self.GATE_DELAY: # Pause
            self.delay_timer = self.DELAY_DURATION # sfx: pause_sound
            
        dist_after = math.dist(self.ball_grid_pos, self.goal_pos)
        
        if dist_after < dist_before:
            return 5.0 # Positive reward for getting closer
        elif dist_after > dist_before:
            return -1.0 # Negative reward for getting further
        return 0.0 # No change

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates (center of cell)."""
        px = self.GRID_ORIGIN_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_ORIGIN_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [px, py]

    def _render_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_ORIGIN_X + i * self.CELL_SIZE
            y = self.GRID_ORIGIN_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_ORIGIN_Y), (x, self.GRID_ORIGIN_Y + self.GRID_AREA_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_ORIGIN_X, y), (self.GRID_ORIGIN_X + self.GRID_AREA_SIZE, y))

        # Draw start and goal
        start_px, start_py = self._grid_to_pixel(self.start_pos)
        goal_px, goal_py = self._grid_to_pixel(self.goal_pos)
        pygame.gfxdraw.filled_circle(self.screen, start_px, start_py, self.CELL_SIZE // 3, self.COLOR_START)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (goal_px - self.CELL_SIZE//3, goal_py - self.CELL_SIZE//3, self.CELL_SIZE*2/3, self.CELL_SIZE*2/3))

        # Draw placed gates
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                gate_type = self.grid[y, x]
                if gate_type != self.GATE_NONE:
                    px, py = self._grid_to_pixel((x, y))
                    color = self.GATE_COLORS[gate_type]
                    rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))

        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos)
        selected_gate_type = self.gate_types[self.selected_gate_idx]
        cursor_color = self.GATE_COLORS[selected_gate_type]
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*cursor_color, 80))
        self.screen.blit(s, (cursor_px - self.CELL_SIZE//2, cursor_py - self.CELL_SIZE//2))
        pygame.draw.rect(self.screen, cursor_color, (cursor_px - self.CELL_SIZE//2, cursor_py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE), 2)

        # Draw ball
        ball_x, ball_y = int(self.ball_pixel_pos[0]), int(self.ball_pixel_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.CELL_SIZE // 2, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.CELL_SIZE // 3, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.CELL_SIZE // 3, self.COLOR_BALL)

    def _render_ui(self):
        # Top UI: Score and Steps
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), self.font_large, self.COLOR_TEXT)
        time_left = max(0, self.MAX_STEPS - self.steps)
        self._render_text(f"STEPS: {time_left}", (self.WIDTH - 150, 10), self.font_large, self.COLOR_TEXT)
        
        # Bottom UI: Gate selection and inventory
        total_width = len(self.gate_types) * 80
        start_x = (self.WIDTH - total_width) // 2
        
        for i, gate_type in enumerate(self.gate_types):
            x_pos = start_x + i * 80
            y_pos = self.HEIGHT - 35
            
            # Draw box
            box_rect = pygame.Rect(x_pos, y_pos, 70, 30)
            is_selected = i == self.selected_gate_idx
            box_color = self.GATE_COLORS[gate_type] if is_selected else self.COLOR_GRID
            pygame.draw.rect(self.screen, box_color, box_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.GATE_COLORS[gate_type], box_rect.inflate(-4,-4), border_radius=3)
            
            # Draw text
            name_text = self.GATE_NAMES[gate_type]
            count_text = f"x{self.gate_inventory[gate_type]}"
            self._render_text(name_text, (box_rect.centerx, box_rect.centery - 7), self.font_small, (0,0,0), center=True)
            self._render_text(count_text, (box_rect.centerx, box_rect.centery + 7), self.font_small, (0,0,0), center=True)
            
        if self.game_over:
             s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
             s.fill((0,0,0,180))
             self.screen.blit(s, (0,0))
             result_text = "GOAL REACHED!" if tuple(self.ball_grid_pos) == self.goal_pos else "GAME OVER"
             self._render_text(result_text, (self.WIDTH//2, self.HEIGHT//2 - 20), self.font_large, self.COLOR_TEXT, center=True)
             self._render_text(f"Final Score: {int(self.score)}", (self.WIDTH//2, self.HEIGHT//2 + 20), self.font_large, self.COLOR_TEXT, center=True)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Resetting environment...")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before reset

        clock.tick(30) # Match the intended FPS

    env.close()