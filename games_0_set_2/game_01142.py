
# Generated: 2025-08-27T16:09:57.599786
# Source Brief: brief_01142.md
# Brief Index: 1142

        
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

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move the basket."
    )

    # User-facing game description
    game_description = (
        "Catch falling fruit in your basket to score points. Miss too many and it's game over!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 30
    LOSE_MISSES = 5
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (144, 238, 144) # Light Green
    COLOR_GROUND = (139, 69, 19) # Saddle Brown
    COLOR_CATCHER = (160, 82, 45) # Sienna
    COLOR_CATCHER_RIM = (139, 69, 19) # Saddle Brown
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 128)
    FRUIT_COLORS = {
        'small': (220, 20, 60),  # Crimson
        'medium': (255, 165, 0), # Orange
        'large': (255, 215, 0),  # Gold
    }
    FRUIT_PROPERTIES = {
        'small': {'radius': 8, 'points': 1},
        'medium': {'radius': 12, 'points': 2},
        'large': {'radius': 16, 'points': 3},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_end = pygame.font.SysFont("Arial", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.missed_count = 0
        self.game_over = False
        self.game_won = False
        
        self.catcher_pos = 0
        self.catcher_width = 80
        self.catcher_height = 20
        self.catcher_speed = 10

        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = 0
        self.base_fruit_speed = 2.0
        self.current_fruit_speed = 2.0
        self.fruits_caught_for_speedup = 0

        self.np_random = None

        # This call will fail if the implementation is incorrect
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.missed_count = 0
        self.game_over = False
        self.game_won = False
        
        self.catcher_pos = self.SCREEN_WIDTH // 2
        
        self.fruits.clear()
        self.particles.clear()
        
        self.fruit_spawn_timer = self.np_random.integers(30, 60)
        self.current_fruit_speed = self.base_fruit_speed
        self.fruits_caught_for_speedup = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.1  # Small penalty per frame to encourage action

        if not self.game_over:
            # --- Handle Actions ---
            movement = action[0]
            if movement == 3:  # Left
                self.catcher_pos -= self.catcher_speed
            elif movement == 4:  # Right
                self.catcher_pos += self.catcher_speed

            # Clamp catcher position to screen bounds
            self.catcher_pos = max(self.catcher_width // 2, min(self.catcher_pos, self.SCREEN_WIDTH - self.catcher_width // 2))

            # --- Update Game State ---
            self._spawn_fruit()
            reward += self._update_fruits()
            self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.game_won:
                reward += 100
            else:
                reward += -100

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            fruit_type = self.np_random.choice(list(self.FRUIT_PROPERTIES.keys()))
            props = self.FRUIT_PROPERTIES[fruit_type]
            x_pos = self.np_random.integers(props['radius'], self.SCREEN_WIDTH - props['radius'])
            
            self.fruits.append({
                'rect': pygame.Rect(x_pos - props['radius'], -props['radius']*2, props['radius']*2, props['radius']*2),
                'type': fruit_type,
                'color': self.FRUIT_COLORS[fruit_type],
                'points': props['points'],
                'radius': props['radius']
            })
            # sfx: spawn
            self.fruit_spawn_timer = self.np_random.integers(int(30 * (self.base_fruit_speed / self.current_fruit_speed)), int(90 * (self.base_fruit_speed / self.current_fruit_speed)))


    def _update_fruits(self):
        step_reward = 0
        catcher_rect = pygame.Rect(self.catcher_pos - self.catcher_width // 2, self.SCREEN_HEIGHT - self.catcher_height - 5, self.catcher_width, self.catcher_height)

        for fruit in self.fruits[:]:
            fruit['rect'].y += self.current_fruit_speed
            
            # Check for catch
            if fruit['rect'].colliderect(catcher_rect):
                self.score += fruit['points']
                step_reward += fruit['points']
                self.fruits.remove(fruit)
                # sfx: catch
                self._spawn_particles(fruit['rect'].center, fruit['color'], 20, 3)

                # Update difficulty
                self.fruits_caught_for_speedup += 1
                if self.fruits_caught_for_speedup >= 10:
                    self.current_fruit_speed += 0.5
                    self.fruits_caught_for_speedup = 0
                continue

            # Check for miss
            if fruit['rect'].top > self.SCREEN_HEIGHT:
                self.missed_count += 1
                self.fruits.remove(fruit)
                # sfx: miss
                self._spawn_particles((fruit['rect'].centerx, self.SCREEN_HEIGHT - 5), self.COLOR_GROUND, 15, 2)
        
        return step_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_won = True
            return True
        if self.missed_count >= self.LOSE_MISSES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed": self.missed_count,
        }

    def _render_background(self):
        # Draw gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - 5, self.SCREEN_WIDTH, 5))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        # Draw fruits
        for fruit in self.fruits:
            pygame.gfxdraw.aacircle(self.screen, fruit['rect'].centerx, fruit['rect'].centery, fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, fruit['rect'].centerx, fruit['rect'].centery, fruit['radius'], fruit['color'])
            
        # Draw catcher
        catcher_rect = pygame.Rect(self.catcher_pos - self.catcher_width // 2, self.SCREEN_HEIGHT - self.catcher_height - 5, self.catcher_width, self.catcher_height)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_RIM, catcher_rect, width=3, border_radius=5)


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, anchor="topleft"):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surface = font.render(text, True, color)
            
            shadow_rect = shadow_surface.get_rect()
            text_rect = text_surface.get_rect()

            setattr(shadow_rect, anchor, (pos[0] + 2, pos[1] + 2))
            setattr(text_rect, anchor, pos)
            
            self.screen.blit(shadow_surface, shadow_rect)
            self.screen.blit(text_surface, text_rect)

        # Draw score and misses
        draw_text(f"Score: {self.score} / {self.WIN_SCORE}", self.font_ui, self.COLOR_TEXT, (10, 10))
        draw_text(f"Misses: {self.missed_count} / {self.LOSE_MISSES}", self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 10, 10), anchor="topright")

        # Draw game over/win message
        if self.game_over:
            if self.game_won:
                draw_text("YOU WIN!", self.font_end, (255, 255, 100), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), anchor="center")
            else:
                draw_text("GAME OVER", self.font_end, (255, 100, 100), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), anchor="center")

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    while running:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering for Human ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to play again.")

        clock.tick(30) # Match the intended FPS

    env.close()