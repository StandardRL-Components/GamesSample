
# Generated: 2025-08-27T20:03:16.013379
# Source Brief: brief_02334.md
# Brief Index: 2334

        
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
        "Controls: Arrow keys to move. Collect 20 yellow gems to win, but watch out for the red enemies!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Collect gems for points while dodging patrolling enemies. Lose all your health and it's game over."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_INVINCIBLE = (150, 255, 150)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART_FULL = (255, 0, 80)
        self.COLOR_HEART_EMPTY = (50, 50, 60)

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 72)

        # Game constants
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 4.0
        self.GEM_SIZE = 12
        self.ENEMY_SIZE = 10
        self.ENEMY_SPEED = 1.5
        self.MAX_HEALTH = 3
        self.WIN_GEM_COUNT = 20
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time
        self.INVINCIBILITY_DURATION = 60 # 2 seconds at 30fps

        # Initialize state variables
        self.player_pos = None
        self.player_health = 0
        self.invincibility_timer = 0
        self.gems = []
        self.enemies = []
        self.gems_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.rng = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize all game state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.MAX_HEALTH
        self.invincibility_timer = 0
        
        self.gems_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self._spawn_gems(5)
        self._spawn_enemies(4)
        
        return self._get_observation(), self._get_info()

    def _spawn_gems(self, num_gems):
        self.gems = []
        for _ in range(num_gems):
            self.gems.append(
                np.array([
                    self.rng.integers(self.GEM_SIZE, self.WIDTH - self.GEM_SIZE),
                    self.rng.integers(self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE)
                ], dtype=np.float32)
            )

    def _spawn_enemies(self, num_enemies):
        self.enemies = []
        for i in range(num_enemies):
            patrol_w = self.rng.integers(100, 250)
            patrol_h = self.rng.integers(100, 250)
            patrol_x = self.rng.integers(self.ENEMY_SIZE, self.WIDTH - patrol_w - self.ENEMY_SIZE)
            patrol_y = self.rng.integers(self.ENEMY_SIZE, self.HEIGHT - patrol_h - self.ENEMY_SIZE)
            
            start_pos = np.array([patrol_x, patrol_y], dtype=np.float32)
            direction = self.rng.choice([[1, 0], [-1, 0], [0, 1], [0, -1]])

            self.enemies.append({
                "pos": start_pos,
                "patrol_rect": pygame.Rect(patrol_x, patrol_y, patrol_w, patrol_h),
                "dir": np.array(direction, dtype=np.float32)
            })
    
    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- REWARD CALCULATION SETUP ---
        reward = 0
        old_player_pos = self.player_pos.copy()

        # --- UPDATE GAME LOGIC ---
        self._handle_player_movement(movement)
        self._update_enemies()
        
        # Continuous rewards
        reward += self._calculate_distance_rewards(old_player_pos)
        
        # Event-based rewards and state changes
        reward += self._handle_collisions()

        if not self.gems:
            self._spawn_gems(5)
            # SFX: Gem respawn chime

        self.steps += 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        # --- CHECK TERMINATION ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100  # Loss penalty
            # SFX: Game over stinger
        elif self.gems_collected >= self.WIN_GEM_COUNT:
            terminated = True
            self.game_over = True
            self.win_condition = True
            reward += 100  # Win bonus
            # SFX: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # Clamp reward to specified range for non-terminal steps
        if not terminated:
            reward = np.clip(reward, -20, 10)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        direction = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            direction[1] = -1
        elif movement == 2:  # Down
            direction[1] = 1
        elif movement == 3:  # Left
            direction[0] = -1
        elif movement == 4:  # Right
            direction[0] = 1
        
        self.player_pos += direction * self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["pos"] += enemy["dir"] * self.ENEMY_SPEED
            
            # Patrol boundary logic
            if enemy["pos"][0] <= enemy["patrol_rect"].left or enemy["pos"][0] >= enemy["patrol_rect"].right:
                enemy["dir"][0] *= -1
            if enemy["pos"][1] <= enemy["patrol_rect"].top or enemy["pos"][1] >= enemy["patrol_rect"].bottom:
                enemy["dir"][1] *= -1
            
            # Clamp position to prevent escaping patrol box due to speed
            enemy["pos"][0] = np.clip(enemy["pos"][0], enemy["patrol_rect"].left, enemy["patrol_rect"].right)
            enemy["pos"][1] = np.clip(enemy["pos"][1], enemy["patrol_rect"].top, enemy["patrol_rect"].bottom)

    def _calculate_distance_rewards(self, old_player_pos):
        reward = 0
        # Reward for moving closer to the nearest gem
        if self.gems:
            old_dists_gem = [np.linalg.norm(old_player_pos - gem_pos) for gem_pos in self.gems]
            new_dists_gem = [np.linalg.norm(self.player_pos - gem_pos) for gem_pos in self.gems]
            if min(new_dists_gem) < min(old_dists_gem):
                reward += 1.0

        # Penalty for moving closer to the nearest enemy
        if self.enemies:
            old_dists_enemy = [np.linalg.norm(old_player_pos - e["pos"]) for e in self.enemies]
            new_dists_enemy = [np.linalg.norm(self.player_pos - e["pos"]) for e in self.enemies]
            if min(new_dists_enemy) < min(old_dists_enemy):
                reward -= 0.1
        
        return reward

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )

        # Player-Gem collision
        gems_to_remove = []
        for i, gem_pos in enumerate(self.gems):
            gem_rect = pygame.Rect(
                gem_pos[0] - self.GEM_SIZE / 2, gem_pos[1] - self.GEM_SIZE / 2,
                self.GEM_SIZE, self.GEM_SIZE
            )
            if player_rect.colliderect(gem_rect):
                gems_to_remove.append(i)
                self.gems_collected += 1
                self.score += 10
                reward += 10
                # SFX: Gem collect sound

        for i in sorted(gems_to_remove, reverse=True):
            del self.gems[i]

        # Player-Enemy collision
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player_pos - enemy["pos"])
                if dist < (self.PLAYER_SIZE / 2 + self.ENEMY_SIZE):
                    self.player_health -= 1
                    self.invincibility_timer = self.INVINCIBILITY_DURATION
                    reward -= 20
                    # SFX: Player hurt sound
                    break # Only take one hit per frame
        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Gems
        for gem_pos in self.gems:
            points = [
                (gem_pos[0], gem_pos[1] - self.GEM_SIZE),
                (gem_pos[0] + self.GEM_SIZE, gem_pos[1]),
                (gem_pos[0], gem_pos[1] + self.GEM_SIZE),
                (gem_pos[0] - self.GEM_SIZE, gem_pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)

        # Render Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY)

        # Render Player
        player_color = self.COLOR_PLAYER
        if self.invincibility_timer > 0:
            # Flicker when invincible
            if (self.invincibility_timer // 3) % 2 == 0:
                player_color = self.COLOR_PLAYER_INVINCIBLE
        
        player_rect = pygame.Rect(
            int(self.player_pos[0] - self.PLAYER_SIZE / 2),
            int(self.player_pos[1] - self.PLAYER_SIZE / 2),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, player_color, player_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        for i in range(self.MAX_HEALTH):
            heart_pos = (self.WIDTH - (i + 1) * 25, 15)
            color = self.COLOR_HEART_FULL if i < self.player_health else self.COLOR_HEART_EMPTY
            points = [
                (heart_pos[0], heart_pos[1] + 4),
                (heart_pos[0] - 8, heart_pos[1] - 2),
                (heart_pos[0] - 4, heart_pos[1] - 8),
                (heart_pos[0], heart_pos[1] - 4),
                (heart_pos[0] + 4, heart_pos[1] - 8),
                (heart_pos[0] + 8, heart_pos[1] - 2),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Gem Count
        gem_count_text = self.font_ui.render(f"GEMS: {self.gems_collected} / {self.WIN_GEM_COUNT}", True, self.COLOR_UI_TEXT)
        text_rect = gem_count_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(gem_count_text, text_rect)

        # Game Over / Win Message
        if self.game_over:
            if self.win_condition:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "gems_collected": self.gems_collected,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Gem Collector")
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    
    terminated = False
    running = True
    total_reward = 0

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    print("--- Game Reset ---")

        if not terminated:
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
            if terminated:
                print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose the numpy array from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()