
# Generated: 2025-08-28T06:12:25.836294
# Source Brief: brief_02869.md
# Brief Index: 2869

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the reticle. Press space to squash a bug."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Squash swarming bugs before they reach the bottom of the grid. Faster bugs are worth more points!"
    )

    # Frames auto-advance at 30fps.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - 360) // 2
        self.GRID_OFFSET_Y = 20
        self.CELL_SIZE = 360 // self.GRID_SIZE

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_RETICLE = (255, 255, 0)
        self.BUG_COLORS = {
            "green": (50, 220, 50),
            "yellow": (255, 200, 0),
            "red": (255, 50, 50),
        }
        
        # Game parameters
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 30  # 30-second game
        self.WIN_CONDITION_SQUASHED = 15
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bugs = []
        self.particles = []
        self.reticle_pos = [0, 0]
        self.bugs_squashed = 0
        self.spawn_timer = 0
        self.last_space_held = False
        self.current_spawn_interval = 0
        self.current_speed_multipliers = {}

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.game_over = False
        
        self.bugs.clear()
        self.particles.clear()
        
        self.reticle_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_space_held = False

        # Difficulty reset
        self.spawn_timer = self.FPS * 2  # First bug in 2 seconds
        self._update_difficulty()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.reticle_pos[1] -= 1  # Up
        if movement == 2: self.reticle_pos[1] += 1  # Down
        if movement == 3: self.reticle_pos[0] -= 1  # Left
        if movement == 4: self.reticle_pos[0] += 1  # Right

        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.GRID_SIZE - 1)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.GRID_SIZE - 1)

        squash_action = space_held and not self.last_space_held
        if squash_action:
            # SQUASH sfx
            squashed_bug_this_frame = False
            for bug in reversed(self.bugs):
                if bug["pos"] == self.reticle_pos and not squashed_bug_this_frame:
                    self.bugs.remove(bug)
                    self.bugs_squashed += 1
                    squashed_bug_this_frame = True
                    
                    # Add score and reward based on bug type
                    if bug["type"] == "green":
                        self.score += 1
                        reward += 1
                    elif bug["type"] == "yellow":
                        self.score += 2
                        reward += 2
                    elif bug["type"] == "red":
                        self.score += 3
                        reward += 3
                    
                    self._create_particles(bug["pos"], self.BUG_COLORS[bug["type"]])
                    break # Only squash one bug per click

        self.last_space_held = space_held

        # --- Game Logic Update ---
        self._update_difficulty()
        self._update_bugs()
        self._update_particles()
        self._spawn_bugs()

        # --- Termination Check ---
        terminated = False
        if self.bugs_squashed >= self.WIN_CONDITION_SQUASHED:
            reward += 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 5 # Time out penalty
            terminated = True
            self.game_over = True
        
        # Check if any bug reached the bottom
        for bug in self.bugs:
            if bug["pos"][1] >= self.GRID_SIZE -1:
                reward -= 10
                terminated = True
                self.game_over = True
                break
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_difficulty(self):
        # Difficulty increases every 5 seconds (150 steps)
        difficulty_tier = self.steps // (self.FPS * 5)
        
        # Spawn rate increases
        self.current_spawn_interval = max(self.FPS * 0.5, self.FPS * 2 - difficulty_tier * 15)

        # Speed increases
        speed_multiplier = 1.0 - difficulty_tier * 0.1
        self.current_speed_multipliers = {
            "green": max(5, int(15 * speed_multiplier)),
            "yellow": max(4, int(10 * speed_multiplier)),
            "red": max(3, int(7 * speed_multiplier)),
        }

    def _spawn_bugs(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.current_spawn_interval
            
            bug_type_roll = self.np_random.random()
            if bug_type_roll < 0.5:
                bug_type = "green"
            elif bug_type_roll < 0.85:
                bug_type = "yellow"
            else:
                bug_type = "red"

            spawn_pos = [self.np_random.integers(0, self.GRID_SIZE), 0]
            
            # Prevent spawning on existing bugs in the top row
            if any(b["pos"] == spawn_pos for b in self.bugs):
                return

            self.bugs.append({
                "pos": spawn_pos,
                "type": bug_type,
                "move_timer": self.current_speed_multipliers[bug_type],
                "draw_size": self.CELL_SIZE * 0.4
            })

    def _update_bugs(self):
        for bug in self.bugs:
            bug["move_timer"] -= 1
            if bug["move_timer"] <= 0:
                bug["pos"][1] += 1
                bug["move_timer"] = self.current_speed_multipliers[bug["type"]]
                # BUG MOVE sfx

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 25),
                "color": color,
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Verticals
            start_x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.GRID_OFFSET_Y), (start_x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE))
            # Horizontals
            start_y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, start_y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, start_y))

        # Draw bugs
        for bug in self.bugs:
            px, py = self._grid_to_pixel(bug["pos"])
            color = self.BUG_COLORS[bug["type"]]
            radius = int(bug["draw_size"])
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = (*p["color"], alpha)
            size = max(1, int(p["life"] * 0.2))
            pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), size)
        
        # Draw reticle
        rx, ry = self._grid_to_pixel(self.reticle_pos)
        size = self.CELL_SIZE // 2
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - size, ry), (rx + size, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - size), (rx, ry + size), 2)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Bugs Squashed
        squashed_text = self.font_small.render(f"SQUASHED: {self.bugs_squashed} / {self.WIN_CONDITION_SQUASHED}", True, self.COLOR_TEXT)
        self.screen.blit(squashed_text, (10, self.SCREEN_HEIGHT - squashed_text.get_height() - 5))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_squashed": self.bugs_squashed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy" or "windows"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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
        
        # Transpose observation back for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(env.FPS)
        
    env.close()