
# Generated: 2025-08-28T02:11:56.794367
# Source Brief: brief_04370.md
# Brief Index: 4370

        
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

    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the flag!"
    )

    game_description = (
        "A fast-paced pixel-art platformer. Jump, run, and collect coins to "
        "reach the flag while avoiding enemies and pitfalls."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.LEVEL_WIDTH_TILES = 150
        self.LEVEL_WIDTH_PX = self.LEVEL_WIDTH_TILES * self.TILE_SIZE

        # Colors
        self.COLOR_BG = (135, 206, 235)  # Light Sky Blue
        self.COLOR_PLAYER = (255, 69, 0)  # Bright Red-Orange
        self.COLOR_PLATFORM = (139, 69, 19)  # Brown
        self.COLOR_COIN = (255, 215, 0)  # Gold
        self.COLOR_ENEMY = (0, 0, 139)  # Dark Blue
        self.COLOR_FLAG_POLE = (192, 192, 192)  # Silver
        self.COLOR_FLAG = (220, 20, 60)  # Crimson
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        
        # Physics constants
        self.GRAVITY = 0.6
        self.PLAYER_JUMP_STRENGTH = -11
        self.PLAYER_MAX_SPEED_X = 5
        self.PLAYER_ACCEL = 0.5
        self.PLAYER_FRICTION = -0.15

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 60)

        # Initialize state variables
        self.player = {}
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.particles = []
        self.flag_rect = None
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.max_x_progress = 0
        self.base_enemy_speed = 1.0

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        """Creates the platforms, coins, and enemies for the level."""
        self.platforms = []
        self.coins = []
        self.enemies = []

        # Ground floor
        self.platforms.append(pygame.Rect(0, self.HEIGHT - self.TILE_SIZE, self.LEVEL_WIDTH_PX, self.TILE_SIZE))

        # Generate procedural platforms
        last_x = 0
        last_y = self.HEIGHT - self.TILE_SIZE * 3
        for i in range(1, self.LEVEL_WIDTH_TILES - 10):
            if i > last_x + self.np_random.integers(3, 7):
                width = self.np_random.integers(3, 10) * self.TILE_SIZE
                height = last_y + self.np_random.integers(-3, 4) * self.TILE_SIZE
                height = np.clip(height, self.TILE_SIZE * 5, self.HEIGHT - self.TILE_SIZE * 4)
                
                platform_rect = pygame.Rect(i * self.TILE_SIZE, height, width, self.TILE_SIZE)
                self.platforms.append(platform_rect)

                # Add coins on top of the platform
                for j in range(width // self.TILE_SIZE // 2):
                    if self.np_random.random() < 0.7:
                        coin_x = platform_rect.x + j * self.TILE_SIZE * 2 + self.TILE_SIZE // 2
                        coin_y = platform_rect.y - self.TILE_SIZE
                        self.coins.append({'rect': pygame.Rect(coin_x, coin_y, self.TILE_SIZE // 2, self.TILE_SIZE // 2), 'anim_state': self.np_random.random() * math.pi * 2})

                # Add an enemy
                if width > 5 * self.TILE_SIZE and self.np_random.random() < 0.4:
                    enemy_x = platform_rect.x + self.TILE_SIZE
                    enemy_y = platform_rect.y - self.TILE_SIZE
                    self.enemies.append({
                        'rect': pygame.Rect(enemy_x, enemy_y, self.TILE_SIZE, self.TILE_SIZE),
                        'vx': 1,
                        'patrol_start': platform_rect.left,
                        'patrol_end': platform_rect.right - self.TILE_SIZE
                    })

                last_x = i + (width // self.TILE_SIZE)
                last_y = height

        # Flag at the end
        flag_x = self.LEVEL_WIDTH_PX - 5 * self.TILE_SIZE
        self.flag_rect = pygame.Rect(flag_x, self.HEIGHT - 6 * self.TILE_SIZE, self.TILE_SIZE, 5 * self.TILE_SIZE)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        self._generate_level()

        self.player = {
            'rect': pygame.Rect(self.TILE_SIZE * 3, self.HEIGHT - self.TILE_SIZE * 5, self.TILE_SIZE, self.TILE_SIZE * 2),
            'vx': 0,
            'vy': 0,
            'on_ground': False,
            'jump_requested': False,
            'last_jump_reward': 0
        }
        self.max_x_progress = self.player['rect'].x
        self.particles = []
        self.camera_x = 0

        return self._get_observation(), self._get_info()

    def _lose_life(self):
        self.lives -= 1
        self.player['rect'].topleft = (self.TILE_SIZE * 3, self.HEIGHT - self.TILE_SIZE * 5)
        self.player['vx'] = 0
        self.player['vy'] = 0
        # sfx: player_hit
        self._create_particles(self.player['rect'].center, self.COLOR_PLAYER, 20)
        if self.lives <= 0:
            self.game_over = True
            # sfx: game_over
        
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- 1. Handle Input ---
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -self.PLAYER_MAX_SPEED_X
        elif movement == 4: # Right
            target_vx = self.PLAYER_MAX_SPEED_X

        # Smooth acceleration
        if self.player['vx'] < target_vx:
            self.player['vx'] = min(target_vx, self.player['vx'] + self.PLAYER_ACCEL)
        elif self.player['vx'] > target_vx:
            self.player['vx'] = max(target_vx, self.player['vx'] - self.PLAYER_ACCEL)

        # Friction
        if target_vx == 0:
            if self.player['vx'] > 0:
                self.player['vx'] += self.PLAYER_FRICTION
                if self.player['vx'] < 0: self.player['vx'] = 0
            elif self.player['vx'] < 0:
                self.player['vx'] -= self.PLAYER_FRICTION
                if self.player['vx'] > 0: self.player['vx'] = 0
        
        if movement == 1 and self.player['on_ground']: # Jump
            self.player['vy'] = self.PLAYER_JUMP_STRENGTH
            self.player['on_ground'] = False
            # sfx: jump
            self._create_particles((self.player['rect'].midbottom), self.COLOR_PLATFORM, 5, -2)

        # --- 2. Update Physics & State ---
        self.steps += 1
        reward = 0
        
        # Apply gravity
        self.player['vy'] += self.GRAVITY
        
        # Move horizontally
        self.player['rect'].x += int(self.player['vx'])
        
        # Horizontal collision
        for plat in self.platforms:
            if self.player['rect'].colliderect(plat):
                if self.player['vx'] > 0:
                    self.player['rect'].right = plat.left
                    self.player['vx'] = 0
                elif self.player['vx'] < 0:
                    self.player['rect'].left = plat.right
                    self.player['vx'] = 0
        
        # Move vertically
        self.player['rect'].y += int(self.player['vy'])
        
        # Vertical collision
        self.player['on_ground'] = False
        landed_on_platform = None
        for plat in self.platforms:
            if self.player['rect'].colliderect(plat):
                if self.player['vy'] > 0: # Landing on a platform
                    self.player['rect'].bottom = plat.top
                    self.player['vy'] = 0
                    if not self.player['on_ground']: # Just landed
                        landed_on_platform = plat
                        self.player['on_ground'] = True
                        # sfx: land
                        self._create_particles(self.player['rect'].midbottom, self.COLOR_PLATFORM, 3, -1)
                elif self.player['vy'] < 0: # Hitting head
                    self.player['rect'].top = plat.bottom
                    self.player['vy'] = 0

        # --- 3. Calculate Rewards & Check Interactions ---
        
        # Reward for progress
        progress = self.player['rect'].x
        if progress > self.max_x_progress:
            reward += (progress - self.max_x_progress) * 0.1
            self.max_x_progress = progress
        
        # Penalty for standing still
        if abs(self.player['vx']) < 0.1:
            reward -= 0.02

        # Reward for jump type
        if landed_on_platform:
            if landed_on_platform.width <= 2 * self.TILE_SIZE:
                reward += 10  # Risky jump
            else:
                reward -= 0.2 # Safe jump
        
        # Coin collection
        for coin in self.coins[:]:
            if self.player['rect'].colliderect(coin['rect']):
                self.coins.remove(coin)
                self.score += 10
                reward += 1
                # sfx: coin_collect
                self._create_particles(coin['rect'].center, self.COLOR_COIN, 10)

        # Enemy updates and collision
        enemy_speed = self.base_enemy_speed + 0.05 * (self.steps // 500)
        for enemy in self.enemies:
            enemy['rect'].x += enemy['vx'] * enemy_speed
            if enemy['rect'].left < enemy['patrol_start'] or enemy['rect'].right > enemy['patrol_end']:
                enemy['vx'] *= -1
            if self.player['rect'].colliderect(enemy['rect']):
                reward -= 5
                self._lose_life()
                break # Only process one hit per frame

        # Falling off screen
        if self.player['rect'].top > self.HEIGHT:
            reward -= 100 # Large penalty for falling
            self._lose_life()

        # Reaching the flag
        if self.player['rect'].colliderect(self.flag_rect):
            self.score += 1000
            reward += 100
            self.game_over = True
            # sfx: victory
        
        # Termination check
        terminated = self.game_over or self.lives <= 0 or self.steps >= 2000
        if terminated and self.lives <= 0 and self.player['rect'].top <= self.HEIGHT:
             # If terminated due to lives lost but not by falling
             pass # Already handled by _lose_life and enemy collision reward

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update camera
        self.camera_x = self.player['rect'].x - self.WIDTH // 3
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH_PX - self.WIDTH))
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_pos": (self.player['rect'].x, self.player['rect'].y),
            "max_progress": self.max_x_progress,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            draw_rect = plat.move(-self.camera_x, 0)
            if draw_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect)
        
        # Draw coins
        for coin in self.coins:
            coin['anim_state'] += 0.2
            anim_width = max(1, int(self.TILE_SIZE//2 * abs(math.sin(coin['anim_state']))))
            draw_rect = coin['rect'].copy()
            draw_rect.width = anim_width
            draw_rect.centerx = coin['rect'].centerx
            draw_rect.move_ip(-self.camera_x, 0)
            if draw_rect.colliderect(self.screen.get_rect()):
                 pygame.draw.ellipse(self.screen, self.COLOR_COIN, draw_rect)

        # Draw enemies
        for enemy in self.enemies:
            draw_rect = enemy['rect'].move(-self.camera_x, 0)
            if draw_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, draw_rect)
        
        # Draw flag
        draw_rect = self.flag_rect.move(-self.camera_x, 0)
        if draw_rect.colliderect(self.screen.get_rect()):
            pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, draw_rect)
            flag_poly = [
                (draw_rect.left + 2, draw_rect.top),
                (draw_rect.left + self.TILE_SIZE * 2, draw_rect.top + self.TILE_SIZE / 2),
                (draw_rect.left + 2, draw_rect.top + self.TILE_SIZE)
            ]
            pygame.gfxdraw.aapolygon(self.screen, flag_poly, self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, flag_poly, self.COLOR_FLAG)

        # Draw player
        player_draw_rect = self.player['rect'].move(-self.camera_x, 0)
        # Simple bobbing animation when running
        if self.player['on_ground'] and abs(self.player['vx']) > 0.1:
            player_draw_rect.y += math.sin(self.steps * 0.5) * 2
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect)
        
        # Draw particles
        for p in self.particles:
            pos = [p['pos'][0] - self.camera_x, p['pos'][1]]
            size = max(0, int(p['life'] / p['max_life'] * 4))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_ui)
        
        # Lives
        lives_text = f"LIVES: {self.lives}"
        text_width = self.font_ui.size(lives_text)[0]
        self._draw_text(lives_text, (self.WIDTH - text_width - 15, 10), self.font_ui)

        # Game Over message
        if self.game_over:
            msg = "VICTORY!" if self.lives > 0 else "GAME OVER"
            text_width, text_height = self.font_game_over.size(msg)
            pos = (self.WIDTH // 2 - text_width // 2, self.HEIGHT // 2 - text_height // 2)
            self._draw_text(msg, pos, self.font_game_over)
    
    def _draw_text(self, text, pos, font):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(surface, pos)

    def _create_particles(self, pos, color, count, y_vel_mod=0):
        for _ in range(count):
            life = self.np_random.integers(10, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-4, 0) + y_vel_mod],
                'life': life,
                'max_life': life,
                'color': color,
            })

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a window for manual play
    pygame.display.set_caption("Manual Play")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    total_steps = 0
    
    while not done:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        # Action 2 (down) is unused in this game
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {total_steps}")
    env.close()