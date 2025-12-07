
# Generated: 2025-08-28T05:39:54.861075
# Source Brief: brief_02701.md
# Brief Index: 2701

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ← to move the catcher left, → to move right. Catch the good critters, avoid the red ones!"
    )

    # User-facing description of the game
    game_description = (
        "A retro arcade game where you catch falling critters. Green and blue critters give points, but red ones are trouble. Catch 25 to win, but miss 5 and you lose."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS

    WIN_CATCHES = 25
    MAX_MISSES = 5
    MAX_STEPS = 1500 # Increased to allow for longer games

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_CATCHER = (255, 200, 0)
    COLOR_TEXT = (220, 220, 240)
    
    CRITTER_COLORS = {
        "green": (50, 220, 100),  # Standard
        "blue": (80, 150, 255),   # Bonus
        "red": (255, 70, 70)      # Penalty
    }
    
    CRITTER_PROB = {"green": 0.65, "blue": 0.2, "red": 0.15}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.critters_caught = 0
        self.missed_critters = 0
        
        self.catcher_pos_x = 0
        self.critters = []
        self.particles = []
        
        self.base_fall_speed = 0.2  # 1 cell per 5 steps
        self.current_fall_speed = self.base_fall_speed
        self.base_spawn_rate = 30 # steps
        self.current_spawn_rate = self.base_spawn_rate
        self.spawn_timer = 0
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.critters_caught = 0
        self.missed_critters = 0
        
        self.catcher_pos_x = self.GRID_COLS // 2
        self.critters.clear()
        self.particles.clear()
        
        self.current_fall_speed = self.base_fall_speed
        self.current_spawn_rate = self.base_spawn_rate
        self.spawn_timer = self.current_spawn_rate
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack action
        movement = action[0]

        if not self.game_over:
            # --- Handle Player Input ---
            if movement == 3:  # Left
                self.catcher_pos_x = max(0, self.catcher_pos_x - 1)
            elif movement == 4:  # Right
                self.catcher_pos_x = min(self.GRID_COLS - 1, self.catcher_pos_x + 1)

            # --- Update Game Logic ---
            self._update_spawner()
            reward += self._update_critters()
            self._update_particles()
            self._update_difficulty()
            
            self.steps += 1

        # --- Check Termination Conditions ---
        terminated = False
        if self.critters_caught >= self.WIN_CATCHES:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.missed_critters >= self.MAX_MISSES:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.current_spawn_rate
            
            critter_type_str = self.np_random.choice(
                list(self.CRITTER_PROB.keys()), p=list(self.CRITTER_PROB.values())
            )
            
            new_critter = {
                "x": self.np_random.integers(0, self.GRID_COLS),
                "y": -1.0, # Start just off-screen
                "type": critter_type_str,
                "color": self.CRITTER_COLORS[critter_type_str]
            }
            self.critters.append(new_critter)
            
    def _update_critters(self):
        step_reward = 0
        critters_to_remove = []

        for critter in self.critters:
            critter["y"] += self.current_fall_speed
            
            if critter["y"] >= self.GRID_ROWS - 1:
                critters_to_remove.append(critter)
                
                if critter["x"] == self.catcher_pos_x:
                    # --- CATCH ---
                    self.critters_caught += 1
                    
                    # Base reward
                    if critter["type"] == "green": step_reward += 1
                    elif critter["type"] == "blue": step_reward += 2
                    elif critter["type"] == "red": step_reward -= 1
                    
                    # Last-moment bonus
                    if critter["y"] >= self.GRID_ROWS - 1:
                        step_reward += 5
                    
                    self.score += step_reward
                    # // Catch sound effect
                    self._create_particles(critter["x"], self.GRID_ROWS - 1, critter["color"], 20)
                else:
                    # --- MISS ---
                    if critter["type"] != "red": # Missing a red critter is good
                        self.missed_critters += 1
                        # // Miss sound effect
                        self._create_particles(critter["x"], self.GRID_ROWS - 0.5, self.CRITTER_COLORS["red"], 10, is_miss=True)

        for critter in critters_to_remove:
            if critter in self.critters:
                self.critters.remove(critter)
                
        return step_reward

    def _update_difficulty(self):
        difficulty_tiers = self.critters_caught // 10
        self.current_fall_speed = self.base_fall_speed + (difficulty_tiers * 0.05)
        self.current_spawn_rate = max(10, self.base_spawn_rate - (difficulty_tiers * 3))

    def _create_particles(self, grid_x, grid_y, color, count, is_miss=False):
        px = (grid_x + 0.5) * self.CELL_WIDTH
        py = (grid_y + 0.5) * self.CELL_HEIGHT
        
        for _ in range(count):
            if is_miss:
                angle = self.np_random.uniform(math.pi, 2 * math.pi) # Downward puff
                speed = self.np_random.uniform(1, 3)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                
            particle = {
                "x": px, "y": py,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "lifespan": self.np_random.integers(20, 40),
                "color": color
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.95
            p["vy"] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                particles_to_remove.append(p)
        
        self.particles = [p for p in self.particles if p not in particles_to_remove]
        
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
            "critters_caught": self.critters_caught,
            "missed_critters": self.missed_critters
        }
    
    def _grid_to_pixels(self, gx, gy):
        return int(gx * self.CELL_WIDTH), int(gy * self.CELL_HEIGHT)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)
            
        # Draw critters
        for critter in self.critters:
            px, py = self._grid_to_pixels(critter["x"], critter["y"])
            center_x = px + self.CELL_WIDTH // 2
            center_y = py + self.CELL_HEIGHT // 2
            radius = self.CELL_WIDTH // 3
            
            if critter["type"] == "green":
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, critter["color"])
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, critter["color"])
            elif critter["type"] == "blue":
                rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
                pygame.draw.rect(self.screen, critter["color"], rect, border_radius=3)
            elif critter["type"] == "red":
                points = [
                    (center_x, center_y - radius),
                    (center_x - radius, center_y + radius // 2),
                    (center_x + radius, center_y + radius // 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, critter["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, critter["color"])

        # Draw catcher
        catcher_px, catcher_py = self._grid_to_pixels(self.catcher_pos_x, self.GRID_ROWS - 1)
        catcher_rect = pygame.Rect(catcher_px + 4, catcher_py + self.CELL_HEIGHT // 2, self.CELL_WIDTH - 8, self.CELL_HEIGHT // 2 - 4)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 40))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p["x"]) - 2, int(p["y"]) - 2))

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives (misses) display
        lives_text = self.font_main.render("Lives:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.MAX_MISSES - self.missed_critters):
            heart_color = (200, 50, 50)
            center = (self.SCREEN_WIDTH - 90 + i * 25, 25)
            points = [
                (center[0], center[1] + 5),
                (center[0] - 10, center[1] - 5),
                (center[0] - 5, center[1] - 10),
                (center[0], center[1] - 5),
                (center[0] + 5, center[1] - 10),
                (center[0] + 10, center[1] - 5),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, heart_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, heart_color)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.critters_caught >= self.WIN_CATCHES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
                
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a dummy video driver for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # Run for a few steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Episode finished. Final Info: {info}")
            obs, info = env.reset()
    
    env.close()
    print("Environment test run completed.")