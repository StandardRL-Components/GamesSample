
# Generated: 2025-08-27T14:48:17.552076
# Source Brief: brief_00792.md
# Brief Index: 792

        
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
        "Controls: Use arrow keys (↑↓←→) to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Dodge bombs and catch falling fruit in this fast-paced, top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_SPEED = 10
    PLAYER_WIDTH = 80
    PLAYER_HEIGHT = 20
    INITIAL_LIVES = 3
    WIN_SCORE = 50
    MAX_STEPS = 1500
    ITEM_RADIUS = 12
    BOMB_PROBABILITY = 0.22
    MIN_SPAWN_INTERVAL = 18
    MAX_SPAWN_INTERVAL = 30
    INITIAL_FALL_SPEED = 2.5
    FALL_SPEED_INCREASE = 0.2
    DIFFICULTY_INTERVAL = 100
    COMBO_TARGET = 5
    COMBO_REWARD = 5.0
    
    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_BASKET = (139, 69, 19)
    COLOR_BASKET_RIM = (160, 82, 45)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BOMB = (50, 50, 50)
    COLOR_SKULL = (220, 220, 220)
    FRUIT_TYPES = {
        "apple": {"color": (220, 20, 60)},
        "lemon": {"color": (255, 235, 59)},
        "lime": {"color": (139, 195, 74)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables to be defined in reset
        self.items = []
        self.particles = []
        self.player_pos = [0, 0]
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.item_spawn_timer = 0
        self.current_fall_speed = 0
        self.combo_counter = 0
        self.screen_shake = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 40]
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.items = []
        self.particles = []
        self.item_spawn_timer = self.np_random.integers(self.MIN_SPAWN_INTERVAL, self.MAX_SPAWN_INTERVAL)
        self.current_fall_speed = self.INITIAL_FALL_SPEED
        self.combo_counter = 0
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0
        terminated = False
        
        if not self.game_over:
            self._handle_input(action)
            self._update_items()
            reward += self._handle_collisions()
            self._update_particles()

            self.steps += 1
            if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
                self.current_fall_speed += self.FALL_SPEED_INCREASE
            
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                if self.win:
                    reward += 100.0 # Win bonus
                else:
                    reward -= 100.0 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_WIDTH // 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH // 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.SCREEN_HEIGHT // 2, self.SCREEN_HEIGHT - self.PLAYER_HEIGHT)

    def _update_items(self):
        # Move existing items
        for item in self.items[:]:
            item['pos'][1] += item['speed']
            if item['pos'][1] > self.SCREEN_HEIGHT + self.ITEM_RADIUS:
                self.items.remove(item)
                if item['type'] != 'bomb':
                    self.combo_counter = 0

        # Spawn new items
        self.item_spawn_timer -= 1
        if self.item_spawn_timer <= 0:
            self._spawn_item()
            self.item_spawn_timer = self.np_random.integers(self.MIN_SPAWN_INTERVAL, self.MAX_SPAWN_INTERVAL)

    def _spawn_item(self):
        x_pos = self.np_random.integers(self.ITEM_RADIUS, self.SCREEN_WIDTH - self.ITEM_RADIUS)
        speed_multiplier = self.np_random.uniform(0.9, 1.2)
        
        if self.np_random.random() < self.BOMB_PROBABILITY:
            item_type = 'bomb'
            color = self.COLOR_BOMB
        else:
            item_type = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
            color = self.FRUIT_TYPES[item_type]['color']

        self.items.append({
            'pos': [x_pos, -self.ITEM_RADIUS],
            'type': item_type,
            'color': color,
            'speed': self.current_fall_speed * speed_multiplier
        })

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH // 2, self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for item in self.items[:]:
            item_rect = pygame.Rect(item['pos'][0] - self.ITEM_RADIUS, item['pos'][1] - self.ITEM_RADIUS, self.ITEM_RADIUS * 2, self.ITEM_RADIUS * 2)
            if player_rect.colliderect(item_rect):
                if item['type'] == 'bomb':
                    # sound: explosion.wav
                    self.lives -= 1
                    reward -= 1.0
                    self.combo_counter = 0
                    self.screen_shake = 10
                    self.items.remove(item)
                else: # Fruit
                    # sound: catch.wav
                    self.score += 1
                    reward += 1.0
                    self.combo_counter += 1
                    if self.combo_counter > 0 and self.combo_counter % self.COMBO_TARGET == 0:
                        # sound: combo.wav
                        reward += self.COMBO_REWARD
                        self.combo_counter = 0
                    self._create_particles(item['pos'], item['color'])
                    self.items.remove(item)
        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            return True
        if self.lives <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-5, 5)
            render_offset[1] = self.np_random.integers(-5, 5)

        self.screen.fill(self.COLOR_BG)
        self._render_game(render_offset)
        self._render_particles(render_offset)
        self._render_ui(render_offset)
        
        if self.game_over:
            self._render_game_over_overlay()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Render player basket
        basket_rect = pygame.Rect(
            offset[0] + self.player_pos[0] - self.PLAYER_WIDTH // 2,
            offset[1] + self.player_pos[1],
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, basket_rect, width=3, border_radius=5)

        # Render items
        for item in self.items:
            pos = (int(offset[0] + item['pos'][0]), int(offset[1] + item['pos'][1]))
            if item['type'] == 'bomb':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, self.COLOR_BOMB)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, self.COLOR_BOMB)
                # Draw skull
                skull_color = self.COLOR_SKULL
                pygame.draw.circle(self.screen, skull_color, (pos[0], pos[1] - 2), 3)
                pygame.draw.circle(self.screen, skull_color, (pos[0] - 4, pos[1] + 3), 2)
                pygame.draw.circle(self.screen, skull_color, (pos[0] + 4, pos[1] + 3), 2)
            else: # Fruit
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, item['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, item['color'])
                stem_color = (34, 139, 34)
                pygame.draw.line(self.screen, stem_color, (pos[0], pos[1] - self.ITEM_RADIUS), (pos[0] + 2, pos[1] - self.ITEM_RADIUS - 5), 2)

    def _render_ui(self, offset):
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (offset[0] + 10, offset[1] + 10))

        for i in range(self.lives):
            pos = (self.SCREEN_WIDTH - 30 - i * 35 + offset[0], 25 + offset[1])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_BOMB)
            pygame.draw.circle(self.screen, self.COLOR_SKULL, (pos[0], pos[1]), 2)

    def _render_game_over_overlay(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _create_particles(self, pos, color):
        for _ in range(15):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'color': color,
                'life': self.np_random.integers(10, 20)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = (int(offset[0] + p['pos'][0]), int(offset[1] + p['pos'][1]))
            size = max(1, int(p['life'] / 4))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()