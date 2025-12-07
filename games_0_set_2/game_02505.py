
# Generated: 2025-08-28T05:03:58.432009
# Source Brief: brief_02505.md
# Brief Index: 2505

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your square. "
        "Catch the falling blocks to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you catch falling colors. "
        "Catch multiple blocks at once for bonus points. Don't miss too many!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.W, self.H = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GRID_W = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_H = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.W - self.GRID_W) // 2
        self.GRID_Y = (self.H - self.GRID_H) // 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_MISS_X = (0, 0, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_MAP = {
            "green": {"rgb": (100, 255, 100), "value": 1},
            "blue": {"rgb": (100, 150, 255), "value": 2},
            "red": {"rgb": (255, 100, 100), "value": 3},
            "yellow": {"rgb": (255, 255, 100), "value": 5},
        }
        self.COLOR_TYPES = list(self.COLOR_MAP.keys())

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 60)
        
        # Game state variables
        self.player_pos = None
        self.falling_colors = None
        self.particles = None
        self.missed_locations = None
        self.score = None
        self.misses = None
        self.catches = None
        self.game_over = None
        self.win_condition = None
        self.total_steps = None
        self.stage = None
        self.stage_timer = None
        self.color_fall_period = None
        self.color_fall_counter = None
        self.color_spawn_period = None
        self.color_spawn_counter = None
        self.catches_for_speedup = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
             self.np_random = np.random.default_rng(seed)

        self.player_pos = [self.GRID_SIZE // 2 -1, self.GRID_SIZE - 1]
        self.falling_colors = []
        self.particles = []
        self.missed_locations = []
        
        self.score = 0
        self.misses = 0
        self.catches = 0
        self.game_over = False
        self.win_condition = False
        
        self.total_steps = 0
        self.stage = 1
        self.stage_timer = 600  # 60 seconds at 10 steps/sec
        
        self.color_fall_period = 20  # Fall 1 cell every 20 steps
        self.color_fall_counter = 0
        self.color_spawn_period = 25  # Spawn a new color every 25 steps
        self.color_spawn_counter = 0
        self.catches_for_speedup = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1 # Small penalty for existing

        # 1. Update Player Position
        player_moved = self._move_player(movement)

        # 2. Update Game Timers
        self._update_timers()

        # 3. Spawn and Move Colors
        self._spawn_colors()
        missed_this_step = self._move_falling_colors()
        
        # 4. Check for Catches
        caught_this_step = self._check_catches()

        # 5. Update State and Calculate Rewards
        reward += self._update_score_and_state(caught_this_step, missed_this_step, player_moved)
        
        # 6. Check for Termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_player(self, movement):
        moved = False
        original_pos = list(self.player_pos)
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_SIZE - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_SIZE - 1)

        if original_pos != self.player_pos:
            moved = True
        return moved

    def _update_timers(self):
        self.total_steps += 1
        self.stage_timer -= 1
        self.color_spawn_counter += 1
        self.color_fall_counter += 1

        if self.stage_timer <= 0:
            if self.stage < 3:
                self.stage += 1
                self.stage_timer = 600
            else:
                self.game_over = True

    def _spawn_colors(self):
        if self.color_spawn_counter >= self.color_spawn_period:
            self.color_spawn_counter = 0
            spawn_x = self.np_random.integers(0, self.GRID_SIZE)
            color_type = self.np_random.choice(self.COLOR_TYPES, p=[0.4, 0.3, 0.2, 0.1])
            self.falling_colors.append({
                "pos": [spawn_x, 0],
                "type": color_type,
                "value": self.COLOR_MAP[color_type]["value"],
                "color_rgb": self.COLOR_MAP[color_type]["rgb"],
            })

    def _move_falling_colors(self):
        missed_this_step = []
        if self.color_fall_counter >= self.color_fall_period:
            self.color_fall_counter = 0
            for color in self.falling_colors:
                color["pos"][1] += 1
            
            i = len(self.falling_colors) - 1
            while i >= 0:
                if self.falling_colors[i]["pos"][1] >= self.GRID_SIZE:
                    missed_item = self.falling_colors.pop(i)
                    missed_this_step.append(missed_item)
                i -= 1
        return missed_this_step

    def _check_catches(self):
        caught_this_step = []
        i = len(self.falling_colors) - 1
        while i >= 0:
            color = self.falling_colors[i]
            if color["pos"] == self.player_pos:
                caught_item = self.falling_colors.pop(i)
                caught_this_step.append(caught_item)
                # SFX: Catch sound
                self._create_particles(self.player_pos, caught_item["color_rgb"])
            i -= 1
        return caught_this_step
    
    def _update_score_and_state(self, caught, missed, player_moved):
        reward = 0
        
        # Reward for catches
        if len(caught) > 0:
            for item in caught:
                reward += item["value"]
                self.score += item["value"]
            self.catches += len(caught)
            self.catches_for_speedup += len(caught)
            if len(caught) > 1:
                reward += 10 # Multi-catch bonus
        
        # Penalty for misses
        if len(missed) > 0:
            reward -= 5 * len(missed)
            self.misses += len(missed)
            for item in missed:
                if len(self.missed_locations) < 5:
                    self.missed_locations.append(item["pos"])

        # Reward for positioning
        is_safe_column = True
        for color in self.falling_colors:
            if color["pos"][0] == self.player_pos[0]:
                reward += 0.5
                is_safe_column = False
                break
        if player_moved and is_safe_column:
            reward -= 0.2

        # Update difficulty
        if self.catches_for_speedup >= 5:
            self.catches_for_speedup -= 5
            self.color_fall_period = max(5, self.color_fall_period - 1)
        
        return reward

    def _check_termination(self):
        terminal_reward = 0
        if self.misses >= 5:
            self.game_over = True
            self.win_condition = False
            terminal_reward = -50
        elif self.catches >= 50:
            self.game_over = True
            self.win_condition = True
            terminal_reward = 50
        elif self.total_steps >= 1800:
            self.game_over = True
            self.win_condition = self.score > 0 # Simple win/loss on timeout
        
        return self.game_over, terminal_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._draw_grid()
        self._update_and_render_particles()
        
        # Draw missed locations
        for pos in self.missed_locations:
            px, py = self._grid_to_pixels(pos)
            pygame.draw.line(self.screen, self.COLOR_MISS_X, (px, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 4)
            pygame.draw.line(self.screen, self.COLOR_MISS_X, (px + self.CELL_SIZE, py), (px, py + self.CELL_SIZE), 4)

        # Draw falling colors
        for color in self.falling_colors:
            px, py = self._grid_to_pixels(color["pos"])
            pygame.draw.rect(self.screen, color["color_rgb"], (px, py, self.CELL_SIZE, self.CELL_SIZE))
            
        # Draw player
        px, py = self._grid_to_pixels(self.player_pos)
        glow_size = int(self.CELL_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (px - (glow_size - self.CELL_SIZE)//2, py - (glow_size - self.CELL_SIZE)//2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px, py, self.CELL_SIZE, self.CELL_SIZE), border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Misses
        miss_text = self.ui_font.render(f"MISSES: {self.misses}/5", True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_text, (self.W - miss_text.get_width() - 20, 15))
        
        # Stage and Timer
        time_sec = self.stage_timer / 10.0
        timer_text = self.ui_font.render(f"STAGE: {self.stage}/3 | TIME: {time_sec:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.W // 2 - timer_text.get_width() // 2, 15))
        
        # Catches
        catch_text = self.ui_font.render(f"CAUGHT: {self.catches}/50", True, self.COLOR_UI_TEXT)
        self.screen.blit(catch_text, (20, self.H - 35))

        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text = self.title_font.render("YOU WIN!", True, (150, 255, 150))
            else:
                end_text = self.title_font.render("GAME OVER", True, (255, 150, 150))
            
            text_rect = end_text.get_rect(center=(self.W // 2, self.H // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "catches": self.catches,
            "misses": self.misses,
            "stage": self.stage,
            "total_steps": self.total_steps,
        }

    def _grid_to_pixels(self, grid_pos):
        px = self.GRID_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_Y + grid_pos[1] * self.CELL_SIZE
        return int(px), int(py)

    def _create_particles(self, grid_pos, color, count=20):
        px, py = self._grid_to_pixels(grid_pos)
        center_x, center_y = px + self.CELL_SIZE / 2, py + self.CELL_SIZE / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": [center_x, center_y], "vel": vel, "life": life, "max_life": life, "color": color})

    def _update_and_render_particles(self):
        i = len(self.particles) - 1
        while i >= 0:
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            
            if p["life"] <= 0:
                self.particles.pop(i)
            else:
                alpha = int(255 * (p["life"] / p["max_life"]))
                radius = int(5 * (p["life"] / p["max_life"]))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, (*p["color"], alpha))
            i -= 1

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.GRID_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.GRID_Y), (start_x, self.GRID_Y + self.GRID_H))
            # Horizontal lines
            start_y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, start_y), (self.GRID_X + self.GRID_W, start_y))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Color Catch")
    screen = pygame.display.set_mode((env.W, env.H))
    clock = pygame.time.Clock()
    running = True
    
    obs, info = env.reset()
    
    # --- Game Loop ---
    while running:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4

        # The MultiDiscrete action is always constructed, even if parts are unused
        action = [movement_action, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Control the frame rate

    env.close()