
# Generated: 2025-08-28T02:55:31.948086
# Source Brief: brief_04618.md
# Brief Index: 4618

        
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
        "Controls: ← to move left, → to move right. Catch the falling fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Move your catcher to grab falling fruit and score points. "
        "Missing too many fruits will end the game. Achieve the target score to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 32
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE

        self.MAX_STEPS = 2000
        self.WIN_CATCHES = 30
        self.MAX_MISSES = 5
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_CATCHER = (66, 135, 245)
        self.COLOR_CATCHER_OUTLINE = (150, 200, 255)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_SHADOW = (10, 10, 20)
        self.COLOR_MISS_ICON = (255, 80, 80)
        self.FRUIT_TYPES = {
            "red": {"color": (255, 70, 70), "value": 1},
            "yellow": {"color": (255, 220, 50), "value": 2},
            "green": {"color": (50, 220, 100), "value": 3},
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.catcher_grid_x = 0
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.missed_fruits = 0
        self.caught_fruits = 0
        self.fall_speed = 0.0
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 0
        self.np_random = None

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.catcher_grid_x = self.GRID_WIDTH // 2
        self.fruits = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.missed_fruits = 0
        self.caught_fruits = 0
        
        self.fall_speed = 2.0
        self.fruit_spawn_rate = 60 # frames
        self.fruit_spawn_timer = self.fruit_spawn_rate // 2

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30) # Maintain 30 FPS

        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 3=left, 4=right

            # --- Calculate Movement Reward ---
            # Find closest fruit to determine if move is good
            prev_dist = self._get_closest_fruit_dist()
            
            # --- Update Game Logic ---
            # 1. Player Movement
            if movement == 3:  # Left
                self.catcher_grid_x -= 1
            elif movement == 4:  # Right
                self.catcher_grid_x += 1
            self.catcher_grid_x = np.clip(self.catcher_grid_x, 0, self.GRID_WIDTH - 1)

            # Add movement reward after moving
            new_dist = self._get_closest_fruit_dist()
            if new_dist < prev_dist:
                reward += 0.1
            elif new_dist > prev_dist:
                reward -= 0.1

            # 2. Fruit Spawning
            self.fruit_spawn_timer -= 1
            if self.fruit_spawn_timer <= 0:
                self._spawn_fruit()
                self.fruit_spawn_timer = self.fruit_spawn_rate

            # 3. Fruit Movement & Collision
            fruits_caught_this_step = 0
            for fruit in self.fruits[:]:
                fruit["y"] += self.fall_speed

                # Check for catch
                catcher_y = self.SCREEN_HEIGHT - self.GRID_SIZE
                if (catcher_y < fruit["y"] < catcher_y + self.GRID_SIZE and
                    fruit["grid_x"] == self.catcher_grid_x):
                    
                    # --- Event: Fruit Caught ---
                    # SFX: catch_sound.play()
                    self.score += fruit["value"]
                    self.caught_fruits += 1
                    reward += 1.0 * fruit["value"] # Reward based on fruit value
                    fruits_caught_this_step += 1
                    
                    self._create_particles(fruit["x"], fruit["y"], fruit["color"], 20, "sparkle")
                    self.fruits.remove(fruit)

                    # Difficulty scaling
                    if self.caught_fruits > 0 and self.caught_fruits % 10 == 0:
                        self.fall_speed += 0.5
                        self.fruit_spawn_rate = max(20, self.fruit_spawn_rate - 5)

                # Check for miss
                elif fruit["y"] > self.SCREEN_HEIGHT:
                    # --- Event: Fruit Missed ---
                    # SFX: miss_sound.play()
                    self.missed_fruits += 1
                    reward -= 5.0
                    self._create_particles(fruit["x"], self.SCREEN_HEIGHT - 5, self.COLOR_MISS_ICON, 15, "splash")
                    self.fruits.remove(fruit)
            
            # Combo reward
            if fruits_caught_this_step > 1:
                reward += 2.0
                # SFX: combo_sound.play()

        # 4. Update Particles
        self._update_particles()
        
        # 5. Check for Termination
        self.steps += 1
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_closest_fruit_dist(self):
        if not self.fruits:
            return self.SCREEN_WIDTH
        catcher_x = (self.catcher_grid_x + 0.5) * self.GRID_SIZE
        min_dist = float('inf')
        for fruit in self.fruits:
            dist = abs(fruit['x'] - catcher_x)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_fruit(self):
        grid_x = self.np_random.integers(0, self.GRID_WIDTH)
        fruit_type_name = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
        fruit_info = self.FRUIT_TYPES[fruit_type_name]
        
        self.fruits.append({
            "grid_x": grid_x,
            "x": (grid_x + 0.5) * self.GRID_SIZE,
            "y": -self.GRID_SIZE / 2,
            "type": fruit_type_name,
            "color": fruit_info["color"],
            "value": fruit_info["value"],
            "radius": self.GRID_SIZE * 0.4
        })

    def _create_particles(self, x, y, color, count, p_type):
        for _ in range(count):
            if p_type == "sparkle":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                lifespan = self.np_random.integers(15, 30)
            elif p_type == "splash":
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                speed = self.np_random.uniform(2, 6)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                lifespan = self.np_random.integers(20, 40)

            self.particles.append({
                "x": x, "y": y, "vx": vx, "vy": vy,
                "lifespan": lifespan, "max_lifespan": lifespan,
                "color": color, "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.2  # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        reward = 0
        if self.caught_fruits >= self.WIN_CATCHES:
            self.game_over = True
            self.win = True
            reward = 50.0
            # SFX: win_jingle.play()
            return True, reward
        if self.missed_fruits >= self.MAX_MISSES:
            self.game_over = True
            self.win = False
            reward = -50.0
            # SFX: lose_sound.play()
            return True, reward
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            # No terminal reward for timeout, just ends.
            return True, reward
        return False, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw fruits
        for fruit in self.fruits:
            x, y, r = int(fruit["x"]), int(fruit["y"]), int(fruit["radius"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit["color"])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (p["x"] - p["size"], p-["y"] - p["size"]))
        
        # Draw catcher
        catcher_rect = pygame.Rect(
            self.catcher_grid_x * self.GRID_SIZE,
            self.SCREEN_HEIGHT - self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE // 2
        )
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_OUTLINE, catcher_rect, width=2, border_radius=4)

    def _render_ui(self):
        # --- Helper to draw text with shadow ---
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_UI_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # --- Score Display ---
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_medium, self.COLOR_UI_TEXT, (15, 10))

        # --- Missed Fruits Display ---
        miss_text = "MISSES:"
        text_surf = self.font_small.render(miss_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (15, 50))
        
        for i in range(self.MAX_MISSES):
            x = 15 + text_surf.get_width() + 10 + i * 25
            y = 50 + text_surf.get_height() // 2
            
            if i < self.missed_fruits:
                # Draw a filled red 'X'
                pygame.draw.line(self.screen, self.COLOR_MISS_ICON, (x-7, y-7), (x+7, y+7), 3)
                pygame.draw.line(self.screen, self.COLOR_MISS_ICON, (x-7, y+7), (x+7, y-7), 3)
            else:
                # Draw an empty gray circle
                pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_GRID)
        
        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "caught_fruits": self.caught_fruits,
            "missed_fruits": self.missed_fruits,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    # This part is for human testing and visualization
    # It is not part of the core Gym environment
    
    # Re-initialize pygame for display
    pygame.display.init()
    pygame.display.set_caption("Fruit Catcher")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Map keyboard inputs to actions
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # no-op, no space, no shift
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate for human play
        env.clock.tick(30)

    env.close()