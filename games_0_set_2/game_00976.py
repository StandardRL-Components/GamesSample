
# Generated: 2025-08-27T15:23:44.677643
# Source Brief: brief_00976.md
# Brief Index: 976

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the catcher. Catch the falling fruit to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in this fast-paced arcade game. Miss three and it's game over!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_LIVES = 3
        self.WIN_CATCH_COUNT = 50
        self.MAX_STEPS = 3000

        # --- Colors ---
        self.COLOR_BG_TOP = (20, 30, 40)
        self.COLOR_BG_BOTTOM = (40, 60, 80)
        self.COLOR_CATCHER = (255, 255, 0) # Bright Yellow
        self.COLOR_CATCHER_OUTLINE = (200, 200, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN = (0, 255, 128)
        self.COLOR_LOSE = (255, 50, 50)
        self.FRUIT_COLORS = {
            "apple": (255, 60, 60),
            "banana": (255, 230, 60),
            "strawberry": (255, 100, 180),
        }
        self.PARTICLE_COLORS = {
            "apple": (255, 120, 120),
            "banana": (255, 240, 120),
            "strawberry": (255, 150, 200),
        }

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 64, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        
        self.catcher_x = 0
        self.catcher_width = 100
        self.catcher_height = 20
        self.catcher_speed = 12

        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 60 # frames
        self.fruit_fall_speed = 0
        self.fruits_caught_total = 0

        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng()
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False
        
        self.catcher_x = self.WIDTH // 2 - self.catcher_width // 2
        
        self.fruits = []
        self.particles = []
        
        self.fruit_spawn_timer = 10 # Start spawning quickly
        self.fruit_spawn_rate = 60
        self.fruit_fall_speed = 3.0
        self.fruits_caught_total = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is already over, do nothing but return the final state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0=none, 3=left, 4=right
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        reward = 0

        # --- Continuous Reward (move towards/away from lowest fruit) ---
        if self.fruits:
            lowest_fruit = min(self.fruits, key=lambda f: self.HEIGHT - f['rect'].y)
            fruit_center_x = lowest_fruit['rect'].centerx
            catcher_center_x = self.catcher_x + self.catcher_width / 2
            
            if movement == 3: # Moving left
                reward += 1.0 if fruit_center_x < catcher_center_x else -0.1
            elif movement == 4: # Moving right
                reward += 1.0 if fruit_center_x > catcher_center_x else -0.1
        
        # --- Update Catcher Position ---
        if movement == 3:
            self.catcher_x -= self.catcher_speed
        elif movement == 4:
            self.catcher_x += self.catcher_speed
        
        # Clamp catcher to screen bounds
        self.catcher_x = max(0, min(self.catcher_x, self.WIDTH - self.catcher_width))

        # --- Update Game Logic ---
        event_reward = self._update_fruits()
        reward += event_reward
        self._update_particles()
        self._spawn_fruit_if_needed()
        
        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            self.win = False
            reward -= 100  # Terminal penalty
            # Sound: game_over.wav
        elif self.fruits_caught_total >= self.WIN_CATCH_COUNT:
            self.game_over = True
            self.win = True
            reward += 100  # Terminal bonus
            # Sound: win.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_fruits(self):
        """Updates fruits, handles collisions/misses, and returns event-based rewards."""
        event_reward = 0
        catcher_rect = pygame.Rect(self.catcher_x, self.HEIGHT - self.catcher_height - 10, self.catcher_width, self.catcher_height)

        for fruit in self.fruits[:]:
            fruit['rect'].y += self.fruit_fall_speed

            # Check for catch
            if catcher_rect.colliderect(fruit['rect']):
                event_reward += 10
                self.score += 1 # Score is just number of fruits caught
                self.fruits_caught_total += 1
                self._create_particles(fruit['rect'].center, fruit['type'])
                self.fruits.remove(fruit)
                # Sound: catch.wav

                # Increase difficulty every 10 fruits
                if self.fruits_caught_total > 0 and self.fruits_caught_total % 10 == 0:
                    self.fruit_fall_speed += 0.5
                    self.fruit_spawn_rate = max(20, self.fruit_spawn_rate - 5)
            
            # Check for miss
            elif fruit['rect'].top > self.HEIGHT:
                event_reward -= 10
                self.lives -= 1
                self.fruits.remove(fruit)
                # Sound: miss.wav
        
        return event_reward

    def _spawn_fruit_if_needed(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self.fruit_spawn_timer = self.np_random.integers(self.fruit_spawn_rate - 10, self.fruit_spawn_rate + 10)

            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            size = self.np_random.integers(20, 31)
            x_pos = self.np_random.integers(0, self.WIDTH - size)
            
            self.fruits.append({
                'rect': pygame.Rect(x_pos, -size, size, size),
                'type': fruit_type,
            })
    
    def _create_particles(self, pos, fruit_type):
        color = self.PARTICLE_COLORS[fruit_type]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life, 'color': color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_fruits()
        self._render_catcher()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT
            pygame.draw.line(self.screen, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))
            
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(6 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), size, (*p['color'], alpha))

    def _render_fruits(self):
        for fruit in self.fruits:
            color = self.FRUIT_COLORS[fruit['type']]
            pygame.draw.ellipse(self.screen, color, fruit['rect'])
            shine_rect = fruit['rect'].copy()
            shine_rect.width = max(0, shine_rect.width // 3)
            shine_rect.height = max(0, shine_rect.height // 3)
            shine_rect.move_ip(5, 5)
            pygame.draw.ellipse(self.screen, (255, 255, 255, 100), shine_rect)

    def _render_catcher(self):
        catcher_rect = pygame.Rect(self.catcher_x, self.HEIGHT - self.catcher_height - 10, self.catcher_width, self.catcher_height)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_OUTLINE, catcher_rect.inflate(4, 4), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=8)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            msg_color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(msg_text, True, msg_color)
            pos_x = self.WIDTH // 2 - game_over_surf.get_width() // 2
            pos_y = self.HEIGHT // 2 - game_over_surf.get_height() // 2
            self.screen.blit(game_over_surf, (pos_x, pos_y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fruits_caught": self.fruits_caught_total,
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    running = True
    while running:
        action = env.action_space.sample()
        action[0] = 0 # Default to no movement
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_r]:
             obs, info = env.reset()
             terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()