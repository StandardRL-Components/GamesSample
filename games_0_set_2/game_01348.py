
# Generated: 2025-08-27T16:50:32.721078
# Source Brief: brief_01348.md
# Brief Index: 1348

        
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

    # Short, user-facing control string
    user_guide = "Controls: ←→ to move the catcher."

    # Short, user-facing description of the game
    game_description = (
        "Catch falling fruit to score points. Reach 50 points to win, but miss 10 and you lose!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS = 10
        self.GRID_CELL_WIDTH = self.WIDTH // self.GRID_COLS
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50
        self.LOSE_MISSES = 10

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_CATCHER = (255, 200, 0)
        self.COLOR_CATCHER_FLASH = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.FRUIT_TYPES = [
            {"color": (255, 50, 50), "value": 1, "radius": 12},  # Red
            {"color": (50, 255, 50), "value": 2, "radius": 10},  # Green
            {"color": (80, 150, 255), "value": 3, "radius": 8},   # Blue
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 36)
            self.font_game_over = pygame.font.Font(None, 72)
        except FileNotFoundError:
            self.font_ui = pygame.font.SysFont("Arial", 30)
            self.font_game_over = pygame.font.SysFont("Arial", 66)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.caught_fruits_total = 0
        self.game_over = False
        self.game_won = False
        
        self.catcher_col = 0
        self.catcher_flash_timer = 0
        
        self.fruits = []
        self.particles = []
        
        self.fruit_spawn_timer = 0
        self.base_fruit_speed = 2.5
        self.current_fruit_speed = 0.0
        
        self.last_reward = 0.0

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.caught_fruits_total = 0
        self.game_over = False
        self.game_won = False
        
        self.catcher_col = self.GRID_COLS // 2
        self.catcher_flash_timer = 0
        
        self.fruits = []
        self.particles = []
        
        self.fruit_spawn_timer = 30  # Spawn first fruit after 1 second
        self.current_fruit_speed = self.base_fruit_speed
        
        self.last_reward = 0.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.01 # Small penalty for time passing
        
        if not self.game_over:
            # --- 1. Handle Player Action ---
            if movement == 3:  # Left
                self.catcher_col = max(0, self.catcher_col - 1)
            elif movement == 4:  # Right
                self.catcher_col = min(self.GRID_COLS - 1, self.catcher_col + 1)
            
            # --- 2. Update Game Logic ---
            self._update_catcher()
            reward += self._update_fruits()
            self._update_particles()
            self._spawn_fruits()
            self._update_difficulty()

        # --- 3. Check for Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        self.last_reward = reward
        self.steps += 1
        
        # --- 4. Return Gymnasium 5-tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_catcher(self):
        if self.catcher_flash_timer > 0:
            self.catcher_flash_timer -= 1

    def _update_fruits(self):
        step_reward = 0
        fruits_to_remove = []
        catcher_rect = pygame.Rect(
            self.catcher_col * self.GRID_CELL_WIDTH,
            self.HEIGHT - 20,
            self.GRID_CELL_WIDTH,
            20
        )

        for fruit in self.fruits:
            fruit["pos"][1] += self.current_fruit_speed
            
            fruit_rect = pygame.Rect(
                fruit["pos"][0] - fruit["type"]["radius"],
                fruit["pos"][1] - fruit["type"]["radius"],
                fruit["type"]["radius"] * 2,
                fruit["type"]["radius"] * 2
            )

            # Check for catch
            if catcher_rect.colliderect(fruit_rect):
                # sfx: catch_fruit
                fruits_to_remove.append(fruit)
                self.score += fruit["type"]["value"]
                self.caught_fruits_total += 1
                self.catcher_flash_timer = 5 # Flash for 5 frames
                
                # Calculate reward
                catch_reward = 1.0 + fruit["type"]["value"] # Base reward + value
                
                # Risky catch bonus
                fruit_center_x = fruit["pos"][0]
                catcher_edge_threshold = self.GRID_CELL_WIDTH * 0.15
                if (fruit_center_x < catcher_rect.left + catcher_edge_threshold or
                    fruit_center_x > catcher_rect.right - catcher_edge_threshold):
                    catch_reward += 1.0 # Bonus for risky catch
                
                step_reward += catch_reward
                self._create_particles(fruit["pos"], fruit["type"]["color"])

            # Check for miss
            elif fruit["pos"][1] > self.HEIGHT + fruit["type"]["radius"]:
                # sfx: miss_fruit
                fruits_to_remove.append(fruit)
                self.missed_fruits += 1
                step_reward -= 5.0 # Penalty for missing

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return step_reward

    def _spawn_fruits(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            fruit_type = random.choice(self.FRUIT_TYPES)
            spawn_col = random.randint(0, self.GRID_COLS - 1)
            x_pos = (spawn_col * self.GRID_CELL_WIDTH) + (self.GRID_CELL_WIDTH / 2)
            
            self.fruits.append({
                "pos": [x_pos, -fruit_type["radius"]],
                "type": fruit_type,
            })
            
            # Reset timer with some randomness
            self.fruit_spawn_timer = max(15, 60 - self.caught_fruits_total) + random.randint(-5, 5)

    def _update_difficulty(self):
        # Increase speed every 10 fruits
        speed_increase_factor = self.caught_fruits_total // 10
        self.current_fruit_speed = self.base_fruit_speed + (speed_increase_factor * 0.25)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]
        
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_won = True
            return True
        if self.missed_fruits >= self.LOSE_MISSES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            x = i * self.GRID_CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, (*p["color"], alpha)
                )

        # Draw fruits
        for fruit in self.fruits:
            pos_x, pos_y = int(fruit["pos"][0]), int(fruit["pos"][1])
            radius = fruit["type"]["radius"]
            color = fruit["type"]["color"]
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, color)

        # Draw catcher
        catcher_color = self.COLOR_CATCHER_FLASH if self.catcher_flash_timer > 0 else self.COLOR_CATCHER
        catcher_rect = pygame.Rect(
            self.catcher_col * self.GRID_CELL_WIDTH,
            self.HEIGHT - 20,
            self.GRID_CELL_WIDTH,
            20
        )
        pygame.draw.rect(self.screen, catcher_color, catcher_rect, border_radius=4)
        
    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses display
        miss_text = self.font_ui.render(f"MISSES: {self.missed_fruits}/{self.LOSE_MISSES}", True, self.COLOR_UI_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(miss_text, miss_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
            "caught_fruits": self.caught_fruits_total,
            "game_over": self.game_over,
        }

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    # To run in a window, we need to create a screen.
    # This is for testing/visualization purposes only.
    # The environment itself is headless.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Fruit Catcher")
        
        obs, info = env.reset()
        done = False
        
        # Use a simple agent that moves randomly
        action = env.action_space.sample()
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # --- Simple Keyboard Agent for Human Play ---
            keys = pygame.key.get_pressed()
            move_action = 0 # no-op
            if keys[pygame.K_LEFT]:
                move_action = 3
            elif keys[pygame.K_RIGHT]:
                move_action = 4
            
            # The action space requires all 3 parts
            action = np.array([move_action, 0, 0])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Draw the observation from the environment to the screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
            if done:
                print(f"Game Over. Final Score: {info['score']}")
                pygame.time.wait(2000) # Wait 2 seconds before closing
                
    finally:
        env.close()