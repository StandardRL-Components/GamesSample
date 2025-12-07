
# Generated: 2025-08-28T03:02:48.869949
# Source Brief: brief_04646.md
# Brief Index: 4646

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to fly up, ↓ to fly down. Avoid the trees and collect all the coins before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Soar through a procedurally generated forest, dodging trees and collecting coins to reach the target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLAYER = (255, 255, 0) # Yellow
    COLOR_PLAYER_OUTLINE = (218, 165, 32) # Goldenrod
    COLOR_TREE_LEAVES = (0, 100, 0) # Dark Green
    COLOR_TREE_TRUNK = (139, 69, 19) # Saddle Brown
    COLOR_COIN = (255, 215, 0) # Gold
    COLOR_COIN_OUTLINE = (218, 165, 32) # Goldenrod
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Screen dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game parameters
    FPS = 30
    WIN_SCORE = 50
    TOTAL_TIME = 180  # seconds
    MAX_STEPS = TOTAL_TIME * FPS

    # Player physics
    GRAVITY = 0.4
    LIFT_FORCE = -8
    DOWN_FORCE = 1.5
    MAX_VEL_Y = 10
    PLAYER_X_POS = 100
    PLAYER_SIZE = (30, 20)

    # World mechanics
    SCROLL_SPEED = 4
    TREE_WIDTH = 80
    MIN_TREE_GAP = 120 # Min vertical gap between trees
    MAX_TREE_GAP = 180 # Max vertical gap between trees
    TREE_SPAWN_INTERVAL = 300 # pixels

    # Difficulty progression
    DIFFICULTY_INTERVAL_SECONDS = 30
    DIFFICULTY_INTERVAL_STEPS = DIFFICULTY_INTERVAL_SECONDS * FPS
    TREE_DENSITY_INCREASE = 1.05 # Multiplier
    COIN_SPAWN_RATE_INCREASE = 1.02 # Multiplier

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.world_scroll = 0
        self.trees = []
        self.coins = []
        self.particles = []
        self.time_left = 0
        self.next_tree_spawn = 0
        self.current_tree_gap_min = self.MIN_TREE_GAP
        self.current_tree_gap_max = self.MAX_TREE_GAP
        self.coin_spawn_chance = 0.5
        self.last_difficulty_increase_step = 0

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_over = False
        self.steps = 0
        self.score = 0
        self.player_pos = [self.PLAYER_X_POS, self.SCREEN_HEIGHT // 2]
        self.player_vel_y = 0
        self.world_scroll = 0
        self.time_left = self.TOTAL_TIME

        self.trees = []
        self.coins = []
        self.particles = []
        
        # Reset difficulty
        self.current_tree_gap_min = self.MIN_TREE_GAP
        self.current_tree_gap_max = self.MAX_TREE_GAP
        self.coin_spawn_chance = 0.5 # Starting chance
        self.last_difficulty_increase_step = 0

        # Initial obstacle generation
        self.next_tree_spawn = self.SCREEN_WIDTH
        for i in range(5): # pre-populate the screen and beyond
            self._spawn_obstacle_pair(self.SCREEN_WIDTH + i * self.TREE_SPAWN_INTERVAL)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            movement = action[0]
            
            # 1. Update Player Physics
            if movement == 1: # Up
                self.player_vel_y = self.LIFT_FORCE
                # sfx: flap_wing.wav
            elif movement == 2: # Down
                self.player_vel_y += self.DOWN_FORCE
            
            self.player_vel_y += self.GRAVITY
            self.player_vel_y = np.clip(self.player_vel_y, -self.MAX_VEL_Y, self.MAX_VEL_Y)
            self.player_pos[1] += self.player_vel_y
            
            # Clamp player to screen
            self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE[1])

            # 2. Update World
            self.world_scroll += self.SCROLL_SPEED
            self.time_left -= 1 / self.FPS
            self.steps += 1

            self._update_entities()
            self._update_difficulty()
            
            # 3. Collision Detection & Rewards
            player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
            
            # Coins
            collected_indices = player_rect.collidelistall(self.coins)
            if collected_indices:
                for i in sorted(collected_indices, reverse=True):
                    self.score += 1
                    reward += 1.0
                    # sfx: coin_collect.wav
                    coin_center = self.coins[i].center
                    self._spawn_particles(coin_center, self.COLOR_COIN, 20)
                    del self.coins[i]
            
            # Trees
            if player_rect.collidelist(self.trees) != -1:
                self.game_over = True
                reward = -100.0
                # sfx: crash.wav
                self._spawn_particles(player_rect.center, self.COLOR_PLAYER, 50)
            else:
                reward += 0.1 # Survival reward
                
                # Near-miss penalty
                near_miss_detected = False
                for tree in self.trees:
                    if tree.right > player_rect.left and tree.left < player_rect.right:
                        is_top_tree = tree.top == 0
                        if is_top_tree:
                           gap_top_y = tree.bottom
                           if abs(player_rect.top - gap_top_y) < 20:
                               near_miss_detected = True; break
                        else: # is bottom tree
                           gap_bottom_y = tree.top
                           if abs(player_rect.bottom - gap_bottom_y) < 20:
                               near_miss_detected = True; break
                if near_miss_detected:
                    reward -= 5.0

        # 4. Check Termination Conditions
        terminated = self.game_over
        if self.score >= self.WIN_SCORE:
            reward = 100.0
            terminated = True
            self.game_over = True # To show win message
        elif self.time_left <= 0:
            reward = -50.0
            terminated = True
            self.game_over = True # To show time out message
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            reward = -50.0 # Time out penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_entities(self):
        # Move trees and coins based on world scroll
        for tree in self.trees:
            tree.x -= self.SCROLL_SPEED
        for coin in self.coins:
            coin.x -= self.SCROLL_SPEED
        
        # Despawn off-screen entities
        self.trees = [t for t in self.trees if t.right > 0]
        self.coins = [c for c in self.coins if c.right > 0]

        # Spawn new trees
        last_tree_x = max([t.x for t in self.trees]) if self.trees else 0
        if last_tree_x < self.SCREEN_WIDTH:
             self._spawn_obstacle_pair(last_tree_x + self.TREE_SPAWN_INTERVAL)

        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0] - self.SCROLL_SPEED
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_difficulty(self):
        if self.steps > self.last_difficulty_increase_step + self.DIFFICULTY_INTERVAL_STEPS:
            self.last_difficulty_increase_step = self.steps
            
            self.current_tree_gap_min = max(self.MIN_TREE_GAP / 1.5, self.current_tree_gap_min / self.TREE_DENSITY_INCREASE)
            self.current_tree_gap_max = max(self.MIN_TREE_GAP, self.current_tree_gap_max / self.TREE_DENSITY_INCREASE)
            self.coin_spawn_chance = min(0.9, self.coin_spawn_chance * self.COIN_SPAWN_RATE_INCREASE)

    def _spawn_obstacle_pair(self, x_pos):
        gap_size = self.np_random.integers(int(self.current_tree_gap_min), int(self.current_tree_gap_max) + 1)
        gap_y = self.np_random.integers(40, self.SCREEN_HEIGHT - 40 - gap_size)

        top_height = gap_y
        bottom_y = gap_y + gap_size
        bottom_height = self.SCREEN_HEIGHT - bottom_y

        self.trees.append(pygame.Rect(x_pos, 0, self.TREE_WIDTH, top_height))
        self.trees.append(pygame.Rect(x_pos, bottom_y, self.TREE_WIDTH, bottom_height))

        if self.np_random.random() < self.coin_spawn_chance:
            coin_y = gap_y + gap_size // 2
            self.coins.append(pygame.Rect(x_pos + self.TREE_WIDTH // 2 - 10, coin_y - 10, 20, 20))

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 3 + 2,
                'color': color,
                'life': self.np_random.integers(15, 30)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw trees
        for tree in self.trees:
            pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, tree)
            leaves_rect = tree.copy().inflate(20, 20)
            if tree.y == 0: leaves_rect.bottom = tree.bottom
            else: leaves_rect.top = tree.top
            pygame.draw.rect(self.screen, self.COLOR_TREE_LEAVES, leaves_rect, border_radius=5)

        # Draw coins
        for coin in self.coins:
            anim_phase = (self.steps + coin.x) % self.FPS / self.FPS
            width_multiplier = abs(math.cos(anim_phase * 2 * math.pi))
            anim_rect = coin.copy()
            anim_rect.width = max(2, int(coin.width * width_multiplier))
            anim_rect.centerx = coin.centerx
            pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)
            pygame.draw.ellipse(self.screen, self.COLOR_COIN_OUTLINE, anim_rect, width=2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Draw player
        if not self.game_over or self.score >= self.WIN_SCORE:
            player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
            pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)
            pygame.draw.ellipse(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2)
            eye_pos = (int(player_rect.centerx + 5), int(player_rect.centery - 3))
            pygame.draw.circle(self.screen, (0,0,0), eye_pos, 2)
            
            wing_angle = math.sin(self.steps * 0.8) * 0.4
            wing_center_y = player_rect.centery
            wing_points = [
                (player_rect.centerx - 5, wing_center_y),
                (player_rect.centerx - 15, wing_center_y - 10 * math.cos(wing_angle) - 2),
                (player_rect.centerx - 15, wing_center_y + 10 * math.cos(wing_angle) + 2)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, wing_points)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER_OUTLINE, wing_points, width=2)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        score_text = self.font_ui.render(f"COINS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        time_str = f"TIME: {max(0, int(self.time_left)):03d}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 5))
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self.score >= self.WIN_SCORE: msg, color = "YOU WIN!", self.COLOR_COIN
            elif self.time_left <= 0: msg, color = "TIME UP!", (255, 100, 100)
            else: msg, color = "GAME OVER", (255, 100, 100)
                
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "time_left": self.time_left }

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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Forest Flyer")
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        action = [movement, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and (env.game_over):
                obs, info = env.reset()
                total_reward = 0

        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if (env.game_over):
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}. Press 'R' to reset.")
        
        clock.tick(env.FPS)
        
    env.close()