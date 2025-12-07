
# Generated: 2025-08-27T18:49:23.978491
# Source Brief: brief_01964.md
# Brief Index: 1964

        
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
    user_guide = "Controls: ←→ to move the basket horizontally to catch falling items."

    # Must be a short, user-facing description of the game:
    game_description = "Catch falling items in a moving basket for points, aiming for a high score before missing too many."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 20, 50)
    COLOR_BASKET = (255, 255, 255)
    COLOR_TEXT = (255, 255, 0)
    ITEM_COLORS = [
        (255, 87, 34),   # Deep Orange
        (3, 169, 244),   # Light Blue
        (76, 175, 80),   # Green
        (255, 235, 59),  # Yellow
        (156, 39, 176),  # Purple
    ]

    # Game Parameters
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 12
    ITEM_RADIUS = 12
    INITIAL_FALL_SPEED = 3.0
    FALL_SPEED_INCREASE = 0.5
    MAX_ITEMS_ON_SCREEN = 7
    ITEM_SPAWN_INTERVAL = 20 # frames

    # Win/Loss Conditions
    WIN_CONDITION_CATCHES = 25
    LOSE_CONDITION_MISSES = 5
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # State variables are initialized in reset()
        self.basket_x = 0
        self.items = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.missed_items = 0
        self.items_caught = 0
        self.consecutive_catches = 0
        self.item_fall_speed = 0
        self.item_spawn_timer = 0
        self.game_over = False
        self.game_outcome_message = ""
        self.combo_display_timer = 0
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.basket_x = self.SCREEN_WIDTH // 2 - self.BASKET_WIDTH // 2
        self.items = []
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.missed_items = 0
        self.items_caught = 0
        self.consecutive_catches = 0
        self.item_fall_speed = self.INITIAL_FALL_SPEED
        self.item_spawn_timer = self.ITEM_SPAWN_INTERVAL
        self.game_over = False
        self.game_outcome_message = ""
        self.combo_display_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            
            # --- Continuous Movement Reward ---
            nearest_item = self._find_nearest_item()
            dist_before = float('inf')
            if nearest_item:
                dist_before = abs((self.basket_x + self.BASKET_WIDTH / 2) - nearest_item['pos'][0])

            # --- Update Game Logic ---
            # 1. Move Basket
            if movement == 3:  # Left
                self.basket_x -= self.BASKET_SPEED
            elif movement == 4:  # Right
                self.basket_x += self.BASKET_SPEED
            self.basket_x = np.clip(self.basket_x, 0, self.SCREEN_WIDTH - self.BASKET_WIDTH)
            
            if nearest_item:
                dist_after = abs((self.basket_x + self.BASKET_WIDTH / 2) - nearest_item['pos'][0])
                if dist_after < dist_before:
                    reward += 1.0
                elif dist_after > dist_before:
                    reward -= 0.1

            # 2. Update Items
            items_to_remove = []
            for item in self.items:
                item['pos'][1] += self.item_fall_speed

                # Check for catch
                basket_rect = pygame.Rect(self.basket_x, self.SCREEN_HEIGHT - self.BASKET_HEIGHT, self.BASKET_WIDTH, self.BASKET_HEIGHT)
                item_rect = pygame.Rect(item['pos'][0] - self.ITEM_RADIUS, item['pos'][1] - self.ITEM_RADIUS, self.ITEM_RADIUS * 2, self.ITEM_RADIUS * 2)
                
                if basket_rect.colliderect(item_rect):
                    # --- Catch Event ---
                    # SFX: Catch sound
                    items_to_remove.append(item)
                    self.items_caught += 1
                    
                    catch_reward = 10 + (5 * self.consecutive_catches)
                    reward += catch_reward
                    self.score += catch_reward

                    self.consecutive_catches += 1
                    self.combo_display_timer = self.FPS * 1.5 # Display combo for 1.5 seconds
                    
                    self._create_catch_particles(item['pos'], item['color'])
                    
                    # Increase difficulty every 5 catches
                    if self.items_caught > 0 and self.items_caught % 5 == 0:
                        self.item_fall_speed += self.FALL_SPEED_INCREASE
                
                # Check for miss
                elif item['pos'][1] > self.SCREEN_HEIGHT:
                    # --- Miss Event ---
                    # SFX: Miss sound
                    items_to_remove.append(item)
                    self.missed_items += 1
                    self.consecutive_catches = 0
                    reward -= 1.0

            for item in items_to_remove:
                self.items.remove(item)

            # 3. Spawn New Items
            self.item_spawn_timer -= 1
            if self.item_spawn_timer <= 0 and len(self.items) < self.MAX_ITEMS_ON_SCREEN:
                self._spawn_item()
                self.item_spawn_timer = self.ITEM_SPAWN_INTERVAL + self.np_random.integers(-5, 6)

            # 4. Update Particles
            self._update_particles()
            if self.combo_display_timer > 0:
                self.combo_display_timer -= 1

        # 5. Check Termination Conditions
        if self.items_caught >= self.WIN_CONDITION_CATCHES:
            terminated = True
            self.game_over = True
            self.game_outcome_message = "YOU WIN!"
            reward += 100
        elif self.missed_items >= self.LOSE_CONDITION_MISSES:
            terminated = True
            self.game_over = True
            self.game_outcome_message = "GAME OVER"
            reward -= 100
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            self.game_outcome_message = "TIME UP"
            reward -= 100 # Penalize for timeout

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "items_caught": self.items_caught,
            "missed_items": self.missed_items,
            "consecutive_catches": self.consecutive_catches,
        }

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], alpha)
                )

        # Render items
        for item in self.items:
            pos = (int(item['pos'][0]), int(item['pos'][1]))
            color = item['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, color)

        # Render basket
        basket_rect = pygame.Rect(self.basket_x, self.SCREEN_HEIGHT - self.BASKET_HEIGHT, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=3)
        # Add a slight glow effect to the basket
        pygame.draw.rect(self.screen, (255, 255, 255, 50), basket_rect.inflate(6, 6), border_radius=5)


    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_medium.render(f"Missed: {self.missed_items}/{self.LOSE_CONDITION_MISSES}", True, self.COLOR_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(miss_text, miss_rect)
        
        # Catches
        catch_text = self.font_medium.render(f"Caught: {self.items_caught}/{self.WIN_CONDITION_CATCHES}", True, self.COLOR_TEXT)
        catch_rect = catch_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 40))
        self.screen.blit(catch_text, catch_rect)

        # Combo display
        if self.consecutive_catches > 1 and self.combo_display_timer > 0:
            combo_text = self.font_small.render(f"{self.consecutive_catches}x Combo!", True, self.COLOR_TEXT)
            alpha = min(255, int(255 * (self.combo_display_timer / (self.FPS * 0.5)))) # Fade out in last 0.5s
            combo_text.set_alpha(alpha)
            combo_pos_x = self.basket_x + self.BASKET_WIDTH / 2
            combo_pos_y = self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 30
            combo_rect = combo_text.get_rect(center=(combo_pos_x, combo_pos_y))
            self.screen.blit(combo_text, combo_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_outcome_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _spawn_item(self):
        x_pos = self.np_random.integers(self.ITEM_RADIUS, self.SCREEN_WIDTH - self.ITEM_RADIUS)
        y_pos = -self.ITEM_RADIUS
        color = self.ITEM_COLORS[self.np_random.integers(0, len(self.ITEM_COLORS))]
        self.items.append({'pos': [x_pos, y_pos], 'color': color})

    def _create_catch_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': velocity,
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(15, 30), # frames
                'max_life': 30,
            })

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                particles_to_remove.append(p)
        for p in particles_to_remove:
            self.particles.remove(p)

    def _find_nearest_item(self):
        if not self.items:
            return None
        basket_center_x = self.basket_x + self.BASKET_WIDTH / 2
        nearest_item = min(
            self.items, 
            key=lambda item: abs(item['pos'][0] - basket_center_x) + item['pos'][1] # Prioritize horizontal closeness and lower items
        )
        return nearest_item

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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Falling Items")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # Action is always [movement, space, shift]
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()