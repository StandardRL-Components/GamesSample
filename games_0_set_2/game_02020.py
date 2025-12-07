
# Generated: 2025-08-27T18:59:20.267159
# Source Brief: brief_02020.md
# Brief Index: 2020

        
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
        "Controls: Use ← and → to move the basket left and right to catch the falling blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling objects in a basket for points. Catch 20 to win, but miss 5 and you lose! The objects fall faster as you score more points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
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
        
        # Fonts and Colors
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BASKET = (255, 255, 255)
        self.COLOR_UI = (200, 200, 200)
        self.OBJECT_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]
        
        # Game constants
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 12
        self.OBJECT_SIZE = 18
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        self.WIN_SCORE = 20
        self.LOSE_MISSES = 5
        self.INITIAL_FALL_SPEED = 2.0
        self.SPAWN_INTERVAL = 45 # Spawn new object every 1.5s
        
        # Initialize state variables
        self.basket_pos = None
        self.objects = None
        self.particles = None
        self.score = None
        self.missed_count = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.base_fall_speed = None
        self.object_spawn_timer = None
        
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_pos = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.BASKET_WIDTH // 2,
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        
        self.objects = []
        self.particles = []
        
        self.score = 0
        self.missed_count = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.base_fall_speed = self.INITIAL_FALL_SPEED
        self.object_spawn_timer = self.SPAWN_INTERVAL
        
        self._spawn_object()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Continuous Reward Calculation (Before Move) ---
        dist_before = self._get_distance_to_nearest_object()

        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos.x += self.BASKET_SPEED
            
        self.basket_pos.x = np.clip(self.basket_pos.x, 0, self.SCREEN_WIDTH - self.BASKET_WIDTH)

        # --- Continuous Reward Calculation (After Move) ---
        dist_after = self._get_distance_to_nearest_object()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 1.0  # Moved closer
            else:
                reward -= 0.1 # Moved away or stayed same distance

        # --- Game Logic Update ---
        self.steps += 1
        self._update_objects()
        self._update_particles()
        
        # --- Spawning ---
        self.object_spawn_timer -= 1
        if self.object_spawn_timer <= 0:
            self._spawn_object()
            self.object_spawn_timer = self.SPAWN_INTERVAL

        # --- Collision and Reward Events ---
        new_objects = []
        for obj in self.objects:
            if self.basket_pos.colliderect(obj["rect"]):
                # sfx: catch
                self.score += 1
                reward += 10
                self._create_particles(obj["rect"].center, obj["color"])
                
                # Increase difficulty
                if self.score % 5 == 0:
                    self.base_fall_speed += 0.5
            elif obj["rect"].top > self.SCREEN_HEIGHT:
                # sfx: miss
                self.missed_count += 1
                reward -= 5
            else:
                new_objects.append(obj)
        self.objects = new_objects
        
        # --- Termination Check ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            self.game_won = True
            reward += 100
        elif self.missed_count >= self.LOSE_MISSES:
            terminated = True
            self.game_over = True
            self.game_won = False
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_won = False # Timeout is a loss
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_distance_to_nearest_object(self):
        if not self.objects:
            return None
        
        # Find the object that is lowest on the screen (closest to being caught)
        lowest_object = min(self.objects, key=lambda o: self.basket_pos.top - o["rect"].bottom)
        
        return abs(self.basket_pos.centerx - lowest_object["rect"].centerx)

    def _spawn_object(self):
        x_pos = self.np_random.integers(self.OBJECT_SIZE, self.SCREEN_WIDTH - self.OBJECT_SIZE)
        color = self.np_random.choice(len(self.OBJECT_COLORS))
        
        obj = {
            "rect": pygame.Rect(x_pos, -self.OBJECT_SIZE, self.OBJECT_SIZE, self.OBJECT_SIZE),
            "color": self.OBJECT_COLORS[color],
            "y_float": float(-self.OBJECT_SIZE)
        }
        self.objects.append(obj)
        
    def _update_objects(self):
        for obj in self.objects:
            obj["y_float"] += self.base_fall_speed
            obj["rect"].y = int(obj["y_float"])

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": velocity,
                "color": color,
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _render_game(self):
        # Draw objects
        for obj in self.objects:
            pygame.draw.rect(self.screen, obj["color"], obj["rect"])
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color_with_alpha = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color_with_alpha
            )
            
        # Draw basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_pos, border_radius=3)
    
    def _render_ui(self):
        score_text = f"Score: {self.score}/{self.WIN_SCORE}"
        miss_text = f"Missed: {self.missed_count}/{self.LOSE_MISSES}"
        
        score_surf = self.font_main.render(score_text, True, self.COLOR_UI)
        miss_surf = self.font_main.render(miss_text, True, self.COLOR_UI)
        
        score_pos = (self.SCREEN_WIDTH // 2 - score_surf.get_width() - 20, 10)
        miss_pos = (self.SCREEN_WIDTH // 2 + 20, 10)
        
        self.screen.blit(score_surf, score_pos)
        self.screen.blit(miss_surf, miss_pos)
        
        if self.game_over:
            if self.game_won:
                end_text = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text = "GAME OVER"
                end_color = (255, 100, 100)
            
            end_surf = self.font_game_over.render(end_text, True, end_color)
            end_pos = (
                self.SCREEN_WIDTH // 2 - end_surf.get_width() // 2,
                self.SCREEN_HEIGHT // 2 - end_surf.get_height() // 2
            )
            self.screen.blit(end_surf, end_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_count": self.missed_count,
            "fall_speed": self.base_fall_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of how the environment works
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = env.SCREEN_WIDTH, env.SCREEN_HEIGHT
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Catch the Falling Objects")
    
    terminated = False
    running = True
    total_reward = 0
    
    # Set a higher FPS for smoother human play
    clock = pygame.time.Clock()
    
    while running:
        # --- Human Input to Action Mapping ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Print info
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Control the frame rate for human play
        clock.tick(30)
        
    env.close()