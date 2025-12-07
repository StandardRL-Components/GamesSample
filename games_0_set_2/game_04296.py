
# Generated: 2025-08-28T01:58:41.870642
# Source Brief: brief_04296.md
# Brief Index: 4296

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for Fruit
class Fruit:
    """Represents a single falling fruit with properties for physics and rendering."""
    def __init__(self, x, speed, fruit_type, np_random):
        self.x = x
        self.y = -20.0
        self.speed = speed
        self.type = fruit_type
        self.size = 20
        self.rotation = np_random.uniform(0, 360)
        self.rotation_speed = np_random.uniform(-2, 2)
        self.id = np_random.integers(1_000_000) # For tracking rewards

        # Define fruit colors
        colors = {
            "apple": (255, 50, 50),
            "orange": (255, 165, 0),
            "lemon": (255, 255, 50),
            "lime": (50, 205, 50),
            "grape": (128, 0, 128),
        }
        self.color = colors[self.type]
        self.stem_color = (139, 69, 19)

    def update(self):
        """Updates the fruit's position and rotation."""
        self.y += self.speed
        self.rotation = (self.rotation + self.rotation_speed) % 360

    def draw(self, surface):
        """Renders the fruit onto the given Pygame surface."""
        # Draw main fruit body (anti-aliased)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size // 2, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size // 2, self.color)

        # Draw a little stem/leaf to show rotation
        angle_rad = math.radians(self.rotation)
        stem_end_x = int(self.x + (self.size / 2) * math.cos(angle_rad))
        stem_end_y = int(self.y + (self.size / 2) * math.sin(angle_rad))
        pygame.draw.line(surface, self.stem_color, (int(self.x), int(self.y)), (stem_end_x, stem_end_y), 2)


# Helper class for Particle
class Particle:
    """Represents a visual effect particle for successful catches."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.integers(20, 40)
        self.size = np_random.integers(3, 7)

    def update(self):
        """Updates the particle's position, size, and lifespan."""
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        """Renders the particle if it's still alive."""
        if self.lifespan > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the basket. Hold Space to prepare a catch. "
        "Catch the fruit when it's inside the basket!"
    )

    game_description = (
        "Catch falling fruit in your basket to score points. "
        "The fruit falls faster as you catch more. Don't miss too many!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 10
        self.CATCHER_WIDTH = 100
        self.CATCHER_HEIGHT = 40
        self.CATCHER_SPEED = 8.0

        # Colors
        self.COLOR_BG = (135, 206, 250) # Light Sky Blue
        self.COLOR_CATCHER = (139, 69, 19) # Saddle Brown
        self.COLOR_CATCHER_RIM = (160, 82, 45) # Sienna
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 74)
        self.font_small = pygame.font.Font(None, 36)
        
        # State variables are initialized in reset()
        self.catcher_x = 0
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.fruits_caught = 0
        self.fruit_speed = 0.0
        self.fruit_spawn_timer = 0
        self.game_over = False
        self.win = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.catcher_x = self.WIDTH / 2
        self.fruits = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = self.LOSS_CONDITION
        self.fruits_caught = 0
        
        self.fruit_speed = 2.0
        self.fruit_spawn_timer = 0

        self.game_over = False
        self.win = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Continuous Reward for moving towards the lowest fruit ---
        dist_before_move = self._get_dist_to_lowest_fruit()

        # --- Handle Actions ---
        movement, space_held, _ = self._unpack_action(action)
        self._move_catcher(movement)

        dist_after_move = self._get_dist_to_lowest_fruit()
        if dist_before_move is not None and dist_after_move is not None:
            # Reward is positive if distance decreases
            reward += (dist_before_move - dist_after_move) * 0.01

        # --- Game Logic ---
        self._update_fruits()
        self._update_particles()
        self._spawn_fruit()

        # --- Catching/Missing Logic ---
        catch_reward, miss_penalty = self._handle_collisions(space_held)
        reward += catch_reward
        reward += miss_penalty
        
        # --- Check Termination ---
        terminated = False
        if self.fruits_caught >= self.WIN_CONDITION:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100 # Win bonus
        elif self.lives <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(space_held),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _unpack_action(self, action):
        movement = action[0]  # 0-4
        space_held = action[1] == 1
        shift_held = action[2] == 1
        return movement, space_held, shift_held

    def _get_dist_to_lowest_fruit(self):
        lowest_fruit = self._find_lowest_fruit()
        if lowest_fruit:
            return abs(self.catcher_x - lowest_fruit.x)
        return None

    def _move_catcher(self, movement):
        # 3 is left, 4 is right
        if movement == 3:
            self.catcher_x -= self.CATCHER_SPEED
        elif movement == 4:
            self.catcher_x += self.CATCHER_SPEED
        
        self.catcher_x = np.clip(self.catcher_x, self.CATCHER_WIDTH / 2, self.WIDTH - self.CATCHER_WIDTH / 2)

    def _update_fruits(self):
        for fruit in self.fruits:
            fruit.update()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            spawn_x = self.np_random.uniform(20, self.WIDTH - 20)
            fruit_types = ["apple", "orange", "lemon", "lime", "grape"]
            fruit_type = self.np_random.choice(fruit_types)
            self.fruits.append(Fruit(spawn_x, self.fruit_speed, fruit_type, self.np_random))
            # Vary spawn rate for more dynamic gameplay
            self.fruit_spawn_timer = self.np_random.integers(40, 80)

    def _handle_collisions(self, space_held):
        catch_reward = 0
        miss_penalty = 0
        
        fruits_to_remove = []
        for fruit in self.fruits:
            # Miss condition
            if fruit.y > self.HEIGHT + fruit.size:
                self.lives -= 1
                miss_penalty -= 1 # Miss event penalty
                fruits_to_remove.append(fruit)
                # Sound: miss_sound.play()
                continue
            
            # Catch condition
            is_over_catcher = abs(fruit.x - self.catcher_x) < self.CATCHER_WIDTH / 2
            is_at_catcher_height = abs(fruit.y - (self.HEIGHT - self.CATCHER_HEIGHT / 2)) < fruit.size + 5
            
            if space_held and is_over_catcher and is_at_catcher_height:
                self.fruits_caught += 1
                self.score += 10 # 10 points per fruit
                catch_reward += 1 # Catch event reward
                fruits_to_remove.append(fruit)
                self._create_particles(fruit.x, fruit.y, fruit.color)
                # Sound: catch_sound.play()
                
                # Increase difficulty
                if self.fruits_caught > 0 and self.fruits_caught % 5 == 0:
                    self.fruit_speed += 0.05
        
        if fruits_to_remove:
            self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
            
        return catch_reward, miss_penalty

    def _find_lowest_fruit(self):
        if not self.fruits:
            return None
        return max(self.fruits, key=lambda f: f.y)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            self.particles.append(Particle(x, y, color, self.np_random))

    def _get_observation(self, space_held=False):
        self.screen.fill(self.COLOR_BG)
        self._render_game(space_held)
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, space_held):
        # Render fruits
        for fruit in self.fruits:
            fruit.draw(self.screen)
            
        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render catcher
        catcher_rect = pygame.Rect(
            self.catcher_x - self.CATCHER_WIDTH / 2,
            self.HEIGHT - self.CATCHER_HEIGHT,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_RIM, catcher_rect, width=4, border_radius=5)
        
        # Render catch "aura" when space is held
        if space_held:
            aura_rect = catcher_rect.inflate(10, 10)
            aura_surface = pygame.Surface(aura_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(aura_surface, (255, 255, 0, 80), aura_surface.get_rect(), border_radius=10)
            self.screen.blit(aura_surface, aura_rect.topleft)

    def _render_ui(self):
        # Helper to draw text with shadow for readability
        def draw_text(text, font, color, x, y, align="topleft"):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            
            if align == "topleft":
                text_rect.topleft = (x, y)
            elif align == "topright":
                text_rect.topright = (x, y)
            elif align == "center":
                text_rect.center = (x, y)

            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Score and Lives
        draw_text(f"Score: {self.score}", self.font_small, self.COLOR_TEXT, 10, 10)
        draw_text(f"Lives: {self.lives}", self.font_small, self.COLOR_TEXT, self.WIDTH - 10, 10, align="topright")
        
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            draw_text(message, self.font_large, color, self.WIDTH / 2, self.HEIGHT / 2, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fruits_caught": self.fruits_caught,
        }
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    
    # --- Headless random agent test ---
    print("--- Starting headless mode test ---")
    os.environ['SDL_VIDEODRIVER'] = 'dummy' 
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)

    terminated = False
    total_reward = 0
    for i in range(200):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 50 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
        if terminated:
            print("Episode finished after {} steps.".format(i+1))
            print("Final Info:", info)
            break
    env.close()

    # --- Interactive human-playable mode ---
    print("\n--- Starting interactive mode ---")
    if 'SDL_VIDEODRIVER' in os.environ:
        del os.environ['SDL_VIDEODRIVER']
    pygame.quit() 

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # no-op
        space_action = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, 0] # shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    env.close()