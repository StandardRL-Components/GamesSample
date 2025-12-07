
# Generated: 2025-08-28T05:54:50.303793
# Source Brief: brief_02773.md
# Brief Index: 2773

        
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

    user_guide = "Controls: ←→ to move the basket."
    game_description = "Catch falling fruit in a basket to score points in a fast-paced, top-down arcade game."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (144, 238, 144)  # Light Green
    COLOR_BASKET = (139, 69, 19)  # Saddle Brown
    COLOR_BASKET_BORDER = (101, 51, 14)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    COLOR_MISS_ICON = (255, 50, 50)

    FRUIT_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    # Game Parameters
    WIN_CONDITION_CAUGHT = 50
    LOSE_CONDITION_MISSED = 10
    MAX_STEPS = 10000
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 10  # pixels per frame
    
    INITIAL_FRUIT_SPEED = 2.0
    INITIAL_SPAWN_RATE = 1.0  # fruits per second
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Pre-render background gradient for performance
        self.background = self._create_gradient_background()
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.basket_pos = [0, 0]
        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = 0.0
        self.current_fruit_speed = 0.0
        self.current_spawn_rate = 0.0

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fruits_caught = 0
        self.fruits_missed = 0
        
        self.basket_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.BASKET_HEIGHT * 1.5]
        self.fruits = []
        self.particles = []
        
        self.fruit_spawn_timer = 0.0
        self.current_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.1  # Continuous penalty for time passing
        
        if not self.game_over:
            # --- 1. Handle Input ---
            self._handle_input(action)

            # --- 2. Update Game State ---
            self._update_fruits()
            self._update_particles()
            self._spawn_fruit()

            # --- 3. Handle Collisions and Events ---
            catch_reward, missed_count = self._handle_collisions_and_misses()
            reward += catch_reward
            self.fruits_missed += missed_count
            
            # --- 4. Update Difficulty ---
            self._update_difficulty()
            
        # --- 5. Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Only add terminal reward once
            self.game_over = True
            if self.fruits_caught >= self.WIN_CONDITION_CAUGHT:
                reward += 100  # Win bonus
            elif self.fruits_missed >= self.LOSE_CONDITION_MISSED:
                reward -= 100  # Loss penalty
        
        self.steps += 1
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        if movement == 3:  # Left
            self.basket_pos[0] -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos[0] += self.BASKET_SPEED
            
        # Clamp basket position to screen bounds
        self.basket_pos[0] = max(self.BASKET_WIDTH / 2, min(self.SCREEN_WIDTH - self.BASKET_WIDTH / 2, self.basket_pos[0]))

    def _update_fruits(self):
        for fruit in self.fruits:
            fruit['pos'][1] += fruit['speed']
            fruit['pos'][0] += fruit['drift']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_fruit(self):
        self.fruit_spawn_timer += 1 / self.FPS
        if self.fruit_spawn_timer >= 1.0 / self.current_spawn_rate:
            self.fruit_spawn_timer = 0
            
            new_fruit = {
                'pos': [random.uniform(20, self.SCREEN_WIDTH - 20), -20],
                'speed': self.current_fruit_speed * random.uniform(0.9, 1.2),
                'drift': random.uniform(-0.5, 0.5),
                'color': random.choice(self.FRUIT_COLORS),
                'radius': random.randint(10, 15),
            }
            self.fruits.append(new_fruit)
            # sfx: fruit_spawn.wav

    def _handle_collisions_and_misses(self):
        reward = 0
        missed_count = 0
        basket_rect = pygame.Rect(
            self.basket_pos[0] - self.BASKET_WIDTH / 2,
            self.basket_pos[1] - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )

        for fruit in self.fruits[:]:
            fruit_pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            
            # Check for catch
            if basket_rect.collidepoint(fruit_pos):
                reward += 1  # Base reward for catching
                self.fruits_caught += 1
                
                # Bonus for edge catches
                dist_from_center = abs(fruit_pos[0] - self.basket_pos[0])
                if dist_from_center > self.BASKET_WIDTH * 0.4:
                    reward += 5
                
                self.score += int(reward)
                self._create_particles(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)
                # sfx: catch.wav
                continue

            # Check for miss
            if fruit['pos'][1] > self.SCREEN_HEIGHT:
                missed_count += 1
                self.fruits.remove(fruit)
                # sfx: miss.wav
        
        return reward, missed_count

    def _update_difficulty(self):
        # Increase speed every 10 fruits
        speed_updates = self.fruits_caught // 10
        self.current_fruit_speed = self.INITIAL_FRUIT_SPEED + speed_updates * 0.5
        
        # Increase spawn rate every 20 fruits
        spawn_updates = self.fruits_caught // 20
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE + spawn_updates * 0.1

    def _check_termination(self):
        win = self.fruits_caught >= self.WIN_CONDITION_CAUGHT
        loss = self.fruits_missed >= self.LOSE_CONDITION_MISSED
        timeout = self.steps >= self.MAX_STEPS
        return win or loss or timeout

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "fruits_caught": self.fruits_caught, "fruits_missed": self.fruits_missed}

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': random.randint(20, 40),
                'color': color
            })

    def _render_game(self):
        self._render_fruits()
        self._render_basket()
        self._render_particles()

    def _render_fruits(self):
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = fruit['radius']
            color = fruit['color']
            
            # Draw anti-aliased filled circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            
            # Add a small highlight for a 3D effect
            highlight_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 4, (255, 255, 255, 128))

    def _render_basket(self):
        basket_rect = pygame.Rect(
            self.basket_pos[0] - self.BASKET_WIDTH / 2,
            self.basket_pos[1] - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_BORDER, basket_rect, width=3, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (*p['color'], alpha)
            size = max(1, int(p['life'] / 10))
            
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_ui(self):
        # Render score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, self.font_small, (20, 20))
        
        # Render missed fruits icons
        for i in range(self.fruits_missed):
            x = self.SCREEN_WIDTH - 30 - (i * 25)
            y = 30
            pygame.draw.line(self.screen, self.COLOR_MISS_ICON, (x - 8, y - 8), (x + 8, y + 8), 4)
            pygame.draw.line(self.screen, self.COLOR_MISS_ICON, (x - 8, y + 8), (x + 8, y - 8), 4)
            
        # Render game over message
        if self.game_over:
            if self.fruits_caught >= self.WIN_CONDITION_CAUGHT:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            self._draw_text(msg, self.font_large, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), color=color, center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        # Draw shadow
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        # Draw text
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # --- Pygame window for human play testing ---
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are no-ops
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()