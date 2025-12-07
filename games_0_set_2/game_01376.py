
# Generated: 2025-08-27T16:56:17.161596
# Source Brief: brief_01376.md
# Brief Index: 1376

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move the basket left and right to catch the falling fruit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch the falling fruit! Move your basket to catch as many as you can. "
        "Win by catching 50 fruits, but lose if you miss 5."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 5
        self.MAX_STEPS = 1500 # Increased to allow more time to win

        # Player constants
        self.BASKET_WIDTH = 100
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 12

        # Fruit constants
        self.FRUIT_RADIUS = 12
        self.INITIAL_FRUIT_SPEED = 2.0
        self.FRUIT_SPAWN_PROB = 0.04

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BASKET = (200, 220, 230)
        self.COLOR_BASKET_OUTLINE = (150, 170, 180)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PROGRESS_BAR = (50, 205, 50)
        self.COLOR_PROGRESS_BAR_BG = (60, 60, 60)
        self.FRUIT_TYPES = {
            0: {"name": "apple", "color": (220, 30, 30)},
            1: {"name": "banana", "color": (255, 225, 50)},
            2: {"name": "orange", "color": (255, 140, 0)},
            3: {"name": "grape", "color": (128, 0, 128)},
            4: {"name": "lime", "color": (50, 205, 50)},
        }

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 64, bold=True)

        # Initialize state variables
        self.basket_pos = None
        self.fruits = None
        self.particles = None
        self.score = None
        self.steps = None
        self.caught_fruits = None
        self.missed_fruits = None
        self.game_over = None
        self.base_fruit_speed = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.basket_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - self.BASKET_HEIGHT * 2)
        self.fruits = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.caught_fruits = 0
        self.missed_fruits = 0
        self.game_over = False
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED

        # Spawn a few initial fruits
        for _ in range(3):
            self._spawn_fruit(initial_spawn=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.clock.tick(self.FPS)
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement = action[0]

        # 1. Handle player movement and calculate penalty
        reward += self._handle_movement(movement)
        
        # 2. Update game state
        self._update_fruits()
        self._update_particles()
        self._spawn_new_fruits()

        # 3. Check for catches and misses
        reward += self._check_collections()

        # 4. Check for game termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        if terminated:
            self.game_over = True

        self.steps += 1
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement):
        # Move basket based on action
        if movement == 3:  # Left
            self.basket_pos.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos.x += self.BASKET_SPEED

        # Clamp basket position to screen bounds
        self.basket_pos.x = max(self.BASKET_WIDTH // 2, self.basket_pos.x)
        self.basket_pos.x = min(self.WIDTH - self.BASKET_WIDTH // 2, self.basket_pos.x)

        # Calculate movement penalty
        penalty = 0
        is_fruit_nearby = any(fruit['pos'].y < self.HEIGHT * 0.8 for fruit in self.fruits)
        if movement in [3, 4] and not is_fruit_nearby:
            penalty = -0.2
        return penalty

    def _update_fruits(self):
        for fruit in self.fruits:
            fruit['pos'].y += fruit['speed']

    def _update_particles(self):
        for particle in self.particles:
            particle['pos'] += particle['vel']
            particle['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _spawn_new_fruits(self):
        if self.np_random.random() < self.FRUIT_SPAWN_PROB:
            self._spawn_fruit()

    def _spawn_fruit(self, initial_spawn=False):
        fruit_type_id = self.np_random.integers(0, len(self.FRUIT_TYPES))
        fruit_info = self.FRUIT_TYPES[fruit_type_id]
        
        x_pos = self.np_random.uniform(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS)
        y_pos = -self.FRUIT_RADIUS if not initial_spawn else self.np_random.uniform(-self.HEIGHT*0.5, -self.FRUIT_RADIUS)
        
        speed = self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5)

        self.fruits.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'type': fruit_type_id,
            'color': fruit_info['color'],
            'speed': max(1.0, speed)
        })

    def _check_collections(self):
        reward = 0
        basket_rect = pygame.Rect(
            self.basket_pos.x - self.BASKET_WIDTH // 2,
            self.basket_pos.y - self.BASKET_HEIGHT // 2,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )

        remaining_fruits = []
        for fruit in self.fruits:
            fruit_rect = pygame.Rect(
                fruit['pos'].x - self.FRUIT_RADIUS,
                fruit['pos'].y - self.FRUIT_RADIUS,
                self.FRUIT_RADIUS * 2,
                self.FRUIT_RADIUS * 2
            )
            
            # Check for catch
            if basket_rect.colliderect(fruit_rect):
                # SFX: Catch sound
                self.caught_fruits += 1
                
                # Check for risky catch
                risky_margin = self.BASKET_WIDTH * 0.15
                if fruit['pos'].x < basket_rect.left + risky_margin or \
                   fruit['pos'].x > basket_rect.right - risky_margin:
                    catch_reward = 2.0
                else:
                    catch_reward = 1.0
                
                reward += catch_reward
                self.score += catch_reward
                self._create_particles(fruit['pos'], fruit['color'])
                
                # Increase difficulty
                if self.caught_fruits > 0 and self.caught_fruits % 10 == 0:
                    self.base_fruit_speed += 0.5

            # Check for miss
            elif fruit['pos'].y > self.HEIGHT + self.FRUIT_RADIUS:
                # SFX: Miss sound
                self.missed_fruits += 1
                reward -= 1.0
                self.score -= 1.0
            else:
                remaining_fruits.append(fruit)
        
        self.fruits = remaining_fruits
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        
        if self.caught_fruits >= self.WIN_CONDITION:
            # SFX: Win jingle
            terminated = True
            terminal_reward = 100.0
        elif self.missed_fruits >= self.LOSS_CONDITION:
            # SFX: Lose sound
            terminated = True
            terminal_reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return terminated, terminal_reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'color': color,
                'max_lifespan': lifespan,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_fruits()
        self._render_particles()
        self._render_basket()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "caught_fruits": self.caught_fruits,
            "missed_fruits": self.missed_fruits,
        }

    def _render_basket(self):
        basket_rect = pygame.Rect(0, 0, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        basket_rect.center = (int(self.basket_pos.x), int(self.basket_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_OUTLINE, basket_rect, width=3, border_radius=5)

    def _render_fruits(self):
        for fruit in self.fruits:
            pos = (int(fruit['pos'].x), int(fruit['pos'].y))
            color = fruit['color']
            radius = self.FRUIT_RADIUS

            # Use gfxdraw for anti-aliased shapes
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'].x) - 2, int(p['pos'].y) - 2))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_main.render(f"MISSED: {self.missed_fruits}/{self.LOSS_CONDITION}", True, self.COLOR_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(miss_text, miss_rect)

        # Progress bar
        progress = self.caught_fruits / self.WIN_CONDITION
        bar_width = self.WIDTH - 20
        bar_height = 15
        
        bg_rect = pygame.Rect(10, self.HEIGHT - bar_height - 10, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, bg_rect, border_radius=4)
        
        fill_rect = pygame.Rect(10, self.HEIGHT - bar_height - 10, int(bar_width * progress), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, fill_rect, border_radius=4)
        
        progress_text = self.font_main.render(f"{self.caught_fruits}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        progress_text_rect = progress_text.get_rect(center=bg_rect.center)
        self.screen.blit(progress_text, progress_text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        if self.caught_fruits >= self.WIN_CONDITION:
            msg = "YOU WIN!"
            color = (100, 255, 100)
        else:
            msg = "GAME OVER"
            color = (255, 100, 100)
            
        text_surf = self.font_game_over.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        
        # print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    action = np.array([0, 0, 0]) # No-op
    
    # Create a window to display the game
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        # Player controls
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action[0] = movement
        
        # Step the environment
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Check for game over
        if done:
            # In a real game, you might show a "Press R to restart" message
            pass

    env.close()