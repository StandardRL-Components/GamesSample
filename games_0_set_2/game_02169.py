
# Generated: 2025-08-28T03:56:17.707457
# Source Brief: brief_02169.md
# Brief Index: 2169

        
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
        "Controls: Use ← and → arrow keys to move the catcher. "
        "Catch the falling fruit to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a top-down arcade game. "
        "Reach the target score of 100 before you miss 20 fruits."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # Visuals
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        self.COLOR_BG_TOP = (135, 206, 250)  # Sky Blue
        self.COLOR_BG_BOTTOM = (70, 130, 180) # Steel Blue
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_CATCHER_BODY = (139, 69, 19) # Saddle Brown
        self.COLOR_CATCHER_LIP = (160, 82, 45) # Sienna
        self.FRUIT_TYPES = [
            {"color": (255, 0, 0), "radius": 12},      # Apple
            {"color": (255, 165, 0), "radius": 13},   # Orange
            {"color": (255, 255, 0), "radius": 11},   # Lemon
        ]
        
        # Game constants
        self.CATCHER_WIDTH = 100
        self.CATCHER_HEIGHT = 20
        self.CATCHER_SPEED = 10
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.LOSE_MISSES = 20
        self.FRUIT_SPAWN_PROB = 0.04
        self.INITIAL_FRUIT_SPEED = 2.0
        
        # Initialize state variables
        self.catcher_x = 0
        self.score = 0
        self.misses = 0
        self.steps = 0
        self.game_over = False
        self.fruits = []
        self.particles = []
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.np_random = None

        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.catcher_x = self.W / 2 - self.CATCHER_WIDTH / 2
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.fruits = []
        self.particles = []
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Small penalty per frame to encourage faster completion

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # 1. Handle player input
            self._handle_input(movement)
            
            # 2. Update game objects and collect event rewards
            catch_rewards, miss_penalties = self._update_fruits()
            reward += catch_rewards + miss_penalties
            self._update_particles()
            
            # 3. Spawn new fruits
            self._spawn_fruit()
            
            # 4. Check for termination
            self.steps += 1
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                if self.score >= self.WIN_SCORE:
                    reward += 100 # Win bonus
                elif self.misses >= self.LOSE_MISSES:
                    reward += -100 # Lose penalty
        else:
            # If game is over, do nothing but allow rendering
            terminated = True
            reward = 0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.catcher_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_x += self.CATCHER_SPEED
        
        # Clamp catcher position to screen bounds
        self.catcher_x = max(0, min(self.W - self.CATCHER_WIDTH, self.catcher_x))

    def _update_fruits(self):
        catch_rewards = 0
        miss_penalties = 0
        
        catcher_rect = pygame.Rect(self.catcher_x, self.H - self.CATCHER_HEIGHT, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        
        for fruit in self.fruits[:]:
            fruit['y'] += fruit['speed']
            
            fruit_rect = pygame.Rect(fruit['x'] - fruit['radius'], fruit['y'] - fruit['radius'], fruit['radius']*2, fruit['radius']*2)

            # Check for catch
            if catcher_rect.colliderect(fruit_rect):
                self.fruits.remove(fruit)
                self.score += 1
                
                # Base reward for catch
                catch_rewards += 1.0
                # Bonus for risky catch
                if fruit['y'] > self.H * 0.9:
                    catch_rewards += 2.0
                
                # Visual/Audio feedback
                self._create_particles(fruit['x'], fruit['y'])
                # # Sound placeholder: pygame.mixer.Sound('catch.wav').play()
                
                # Difficulty progression
                if self.score > 0 and self.score % 50 == 0:
                    self.base_fruit_speed += 0.5

            # Check for miss
            elif fruit['y'] > self.H + fruit['radius']:
                self.fruits.remove(fruit)
                self.misses += 1
                # # Sound placeholder: pygame.mixer.Sound('miss.wav').play()

        return catch_rewards, miss_penalties

    def _spawn_fruit(self):
        if self.np_random.random() < self.FRUIT_SPAWN_PROB:
            fruit_type = self.np_random.choice(len(self.FRUIT_TYPES))
            fruit_info = self.FRUIT_TYPES[fruit_type]
            
            self.fruits.append({
                'x': self.np_random.integers(fruit_info['radius'], self.W - fruit_info['radius']),
                'y': -fruit_info['radius'],
                'speed': self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5),
                'color': fruit_info['color'],
                'radius': fruit_info['radius']
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, x, y):
        for _ in range(20):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-3, 1),
                'lifespan': self.np_random.integers(15, 30),
                'color': (255, 255, self.np_random.integers(100, 255)),
                'radius': self.np_random.uniform(1, 4)
            })
            
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.misses >= self.LOSE_MISSES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a smooth gradient from top to bottom
        for y in range(self.H):
            interp = y / self.H
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))

    def _render_game_elements(self):
        # Draw fruits
        for fruit in self.fruits:
            pygame.gfxdraw.aacircle(self.screen, int(fruit['x']), int(fruit['y']), fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(fruit['x']), int(fruit['y']), fruit['radius'], fruit['color'])

        # Draw catcher (basket)
        catcher_rect = pygame.Rect(self.catcher_x, self.H - self.CATCHER_HEIGHT, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_BODY, catcher_rect, border_bottom_left_radius=8, border_bottom_right_radius=8)
        lip_rect = pygame.Rect(self.catcher_x - 5, self.H - self.CATCHER_HEIGHT - 5, self.CATCHER_WIDTH + 10, 10)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_LIP, lip_rect, border_radius=5)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            # Pygame doesn't handle alpha well on base surfaces without flags, so we fake it by not drawing when invisible
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color[:3])


    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render misses
        miss_label_text = self.font_small.render("Misses:", True, self.COLOR_TEXT)
        self.screen.blit(miss_label_text, (10, 50))
        for i in range(self.LOSE_MISSES):
            color = (100, 100, 100) if i >= self.misses else (220, 20, 60) # Gray if not missed, Red if missed
            pygame.gfxdraw.filled_circle(self.screen, 100 + i * 15, 62, 5, color)
            pygame.gfxdraw.aacircle(self.screen, 100 + i * 15, 62, 5, color)
            
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(message, True, (255, 255, 0))
            text_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.W, env.H))
    
    # Game loop
    running = True
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control frame rate

    env.close()