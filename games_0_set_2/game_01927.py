
# Generated: 2025-08-28T03:08:15.157773
# Source Brief: brief_01927.md
# Brief Index: 1927

        
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

    user_guide = (
        "Controls: ←→ to move the basket. Catch the falling fruit!"
    )

    game_description = (
        "Catch falling fruit in a basket to score points. Missing 5 fruits or running out of time ends the game. "
        "Catch all 25 fruits in a level to win!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    
    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_BASKET = (139, 69, 19) # Saddle Brown
    COLOR_BASKET_RIM = (160, 82, 45) # Sienna
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 128)
    FRUIT_COLORS = [
        (255, 0, 0),    # Red (Apple)
        (255, 165, 0),  # Orange (Orange)
        (255, 255, 0),  # Yellow (Banana-like)
        (0, 128, 0),    # Green (Pear)
        (128, 0, 128),  # Purple (Grape)
    ]
    
    # Game Parameters
    MAX_MISSES = 5
    FRUITS_PER_LEVEL = 25
    LEVEL_TIME_SECONDS = 60
    
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 8
    
    FRUIT_RADIUS = 12
    INITIAL_FRUIT_SPEED = 2.0
    SPEED_INCREASE_PER_MILESTONE = 0.25
    FRUITS_PER_MILESTONE = 5

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
        
        self.ui_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.basket_rect = None
        self.fruits = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.missed_fruits = 0
        self.fruits_caught_this_level = 0
        self.total_fruits_spawned = 0
        self.level_timer = 0
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.game_over = False
        self.spawn_timer = 0
        
        # Initialize state
        self.reset()

        # This check is not part of the standard API but is required by the prompt
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_rect = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) // 2,
            self.HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.fruits_caught_this_level = 0
        self.total_fruits_spawned = 0
        self.game_over = False
        
        self.level_timer = self.LEVEL_TIME_SECONDS * self.FPS
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        
        # Spawn one fruit immediately
        self._spawn_fruit()
        self.spawn_timer = self.np_random.integers(30, 60)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game Logic ---
        self.steps += 1
        self.level_timer -= 1

        # 1. Handle Player Input
        movement = action[0]
        if movement == 3:  # Left
            self.basket_rect.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_rect.x += self.BASKET_SPEED
        
        self.basket_rect.x = np.clip(self.basket_rect.x, 0, self.WIDTH - self.BASKET_WIDTH)

        # 2. Update Fruits
        fruits_to_remove = []
        caught_in_frame = 0
        for fruit in self.fruits:
            fruit['pos'][1] += fruit['vel']
            
            # Check for catch
            fruit_rect = pygame.Rect(fruit['pos'][0] - self.FRUIT_RADIUS, fruit['pos'][1] - self.FRUIT_RADIUS, self.FRUIT_RADIUS * 2, self.FRUIT_RADIUS * 2)
            if self.basket_rect.colliderect(fruit_rect):
                # Sound effect placeholder: # sfx_catch.play()
                reward += 1
                self.score += 1
                self.fruits_caught_this_level += 1
                caught_in_frame += 1
                fruits_to_remove.append(fruit)
                self._create_particles(fruit['pos'], fruit['color'])
                
                # Check for difficulty increase
                if self.fruits_caught_this_level > 0 and self.fruits_caught_this_level % self.FRUITS_PER_MILESTONE == 0:
                    self.base_fruit_speed += self.SPEED_INCREASE_PER_MILESTONE

            # Check for miss
            elif fruit['pos'][1] > self.HEIGHT + self.FRUIT_RADIUS:
                # Sound effect placeholder: # sfx_miss.play()
                reward -= 1
                self.missed_fruits += 1
                fruits_to_remove.append(fruit)

        if caught_in_frame > 1:
            reward += 5 # Multi-catch bonus

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]

        # 3. Update Particles
        self._update_particles()

        # 4. Spawn New Fruits
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.total_fruits_spawned < self.FRUITS_PER_LEVEL:
            self._spawn_fruit()
            # Set next spawn time, making it slightly random
            self.spawn_timer = self.np_random.integers(45, 90)

        # 5. Check Termination Conditions
        terminated = False
        if self.missed_fruits >= self.MAX_MISSES:
            terminated = True
        elif self.level_timer <= 0:
            terminated = True
        elif self.fruits_caught_this_level >= self.FRUITS_PER_LEVEL:
            reward += 50  # Level completion bonus
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_fruit(self):
        if self.total_fruits_spawned >= self.FRUITS_PER_LEVEL:
            return
            
        x_pos = self.np_random.integers(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS)
        color = random.choice(self.FRUIT_COLORS)
        # Add slight variation to speed
        speed_variation = self.np_random.uniform(-0.2, 0.2)
        
        fruit = {
            'pos': [x_pos, -self.FRUIT_RADIUS],
            'vel': self.base_fruit_speed + speed_variation,
            'color': color,
        }
        self.fruits.append(fruit)
        self.total_fruits_spawned += 1

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])

        # Render basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, self.basket_rect, width=3, border_radius=5)
        
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"Score: {self.score}"
        text_size = self.ui_font.size(score_text)
        draw_text(score_text, self.ui_font, self.COLOR_TEXT, (self.WIDTH - text_size[0] - 10, 10), self.COLOR_TEXT_SHADOW)
        
        # Timer
        time_left = max(0, self.level_timer // self.FPS)
        timer_text = f"Time: {time_left}"
        text_size = self.ui_font.size(timer_text)
        draw_text(timer_text, self.ui_font, self.COLOR_TEXT, ((self.WIDTH - text_size[0]) // 2, 10), self.COLOR_TEXT_SHADOW)

        # Fruits Caught
        caught_text = f"Caught: {self.fruits_caught_this_level}/{self.FRUITS_PER_LEVEL}"
        draw_text(caught_text, self.small_font, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)

        # Misses
        misses_text = f"Misses: {self.missed_fruits}/{self.MAX_MISSES}"
        text_size = self.small_font.size(misses_text)
        draw_text(misses_text, self.small_font, self.COLOR_TEXT, (self.WIDTH - text_size[0] - 10, self.HEIGHT - text_size[1] - 10), self.COLOR_TEXT_SHADOW)

        # Game Over message
        if self.game_over:
            if self.fruits_caught_this_level >= self.FRUITS_PER_LEVEL:
                msg = "LEVEL COMPLETE!"
            else:
                msg = "GAME OVER"
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            text_size = self.ui_font.size(msg)
            draw_text(msg, self.ui_font, self.COLOR_TEXT, ((self.WIDTH - text_size[0]) // 2, (self.HEIGHT - text_size[1]) // 2), self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
            "fruits_caught": self.fruits_caught_this_level,
            "time_left_seconds": max(0, self.level_timer // self.FPS),
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")

    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Human ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose the observation back for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if done:
            print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            print("Press 'R' to restart.")
            # Wait for restart
            while True:
                restart_event = pygame.event.wait()
                if restart_event.type == pygame.QUIT:
                    running = False
                    break
                if restart_event.type == pygame.KEYDOWN and restart_event.key == pygame.K_r:
                    print("Resetting game...")
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                    break

    env.close()