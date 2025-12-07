
# Generated: 2025-08-28T04:31:05.449456
# Source Brief: brief_05270.md
# Brief Index: 5270

        
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
        "Controls: ←→ to move, ↑ to jump. Collect coins and reach the flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced procedural platformer. Jump across gaps, collect coins for points, and reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH = 640
        self.HEIGHT = 400
        self.LEVEL_LENGTH = 12000 # Pixel length of the level
        self.CHUNK_SIZE = 1500 # How many pixels of level to generate at a time

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_BG_TILE = (20, 30, 50)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_PLATFORM = (120, 130, 150)
        self.COLOR_PLATFORM_RISKY = (255, 100, 180)
        self.COLOR_FLAG_POLE = (200, 200, 200)
        self.COLOR_FLAG = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Physics and game constants
        self.FPS = 30
        self.GRAVITY = 0.6
        self.PLAYER_SIZE = 20
        self.PLAYER_JUMP_STRENGTH = 11
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.85
        self.MAX_VEL_X = 6
        self.MAX_EPISODE_STEPS = 1800 # 60 seconds at 30fps

        # Initialize state variables
        self.player = {}
        self.platforms = []
        self.coins = []
        self.particles = deque()
        self.stars = []
        self.camera_x = 0
        self.last_generated_x = 0
        self.flag_pos = (0, 0)
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.reward_this_step = 0
        self.timer = self.MAX_EPISODE_STEPS

        # Player state
        self.player = {
            'x': 100.0, 'y': 200.0,
            'vx': 0.0, 'vy': 0.0,
            'on_ground': False,
            'squash': 0.0
        }

        # Procedural generation
        self.platforms = []
        self.coins = []
        self.particles.clear()
        self.last_generated_x = 0
        self._generate_level()

        # Camera
        self.camera_x = 0

        # Background
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'x': self.np_random.uniform(0, self.LEVEL_LENGTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'depth': self.np_random.uniform(0.1, 0.8) # For parallax
            })

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Starting platform
        start_platform = pygame.Rect(20, self.HEIGHT - 80, 200, 40)
        self.platforms.append({'rect': start_platform, 'type': 'safe', 'moving': False})
        self.last_generated_x = start_platform.right

        # Generate chunks until the level is complete
        while self.last_generated_x < self.LEVEL_LENGTH - self.WIDTH:
            self._generate_chunk()

        # End platform and flag
        end_platform = pygame.Rect(self.LEVEL_LENGTH, self.HEIGHT - 80, 200, 40)
        self.platforms.append({'rect': end_platform, 'type': 'safe', 'moving': False})
        self.flag_pos = (self.LEVEL_LENGTH + 50, self.HEIGHT - 140)

    def _generate_chunk(self):
        current_x = self.last_generated_x
        last_platform = self.platforms[-1]['rect']
        
        difficulty = 1.0 + (self.steps / 1500.0)

        while current_x < self.last_generated_x + self.CHUNK_SIZE:
            gap = self.np_random.uniform(40, 100) * min(difficulty, 1.8)
            plat_y = last_platform.y + self.np_random.uniform(-60, 60)
            plat_y = np.clip(plat_y, self.PLAYER_SIZE * 4, self.HEIGHT - self.PLAYER_SIZE * 2)

            is_risky = self.np_random.random() < 0.2
            is_moving = not is_risky and self.np_random.random() < (0.15 * difficulty)

            if is_risky:
                plat_width = self.np_random.uniform(25, 40)
                ptype = 'risky'
            else:
                plat_width = self.np_random.uniform(80, 200) / min(difficulty, 1.5)
                ptype = 'safe'

            new_platform_rect = pygame.Rect(
                last_platform.right + gap, plat_y, max(25, plat_width), 20
            )

            moving_props = {}
            if is_moving:
                moving_props = {
                    'moving': True,
                    'speed': self.np_random.uniform(0.5, 1.5) + (self.steps / 300 * 0.05),
                    'range': self.np_random.uniform(40, 100),
                    'start_y': plat_y,
                    'direction': 1
                }
            else:
                moving_props = {'moving': False}

            self.platforms.append({'rect': new_platform_rect, 'type': ptype, **moving_props})
            
            # Add coins
            if not is_risky and self.np_random.random() < 0.7:
                num_coins = self.np_random.integers(1, 5)
                for i in range(num_coins):
                    coin_x = new_platform_rect.x + (new_platform_rect.width / (num_coins + 1)) * (i + 1)
                    coin_y = new_platform_rect.y - 40
                    self.coins.append(pygame.Rect(coin_x, coin_y, 12, 12))

            last_platform = new_platform_rect
            current_x = new_platform_rect.right
        
        self.last_generated_x = current_x

    def step(self, action):
        self.reward_this_step = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_physics()
            self._handle_collisions()
            self._update_game_state()

        reward = self.reward_this_step

        if self._check_termination():
            terminated = True
            if self.victory:
                reward += 100 # Goal-oriented reward
                # sfx: victory fanfare
            elif self.player['y'] > self.HEIGHT + self.PLAYER_SIZE:
                reward -= 10 # Penalty for falling
                # sfx: fall whistle
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        # Horizontal movement
        if movement == 3: # Left
            self.player['vx'] -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player['vx'] += self.PLAYER_ACCEL
        
        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vy'] = -self.PLAYER_JUMP_STRENGTH
            self.player['on_ground'] = False
            self.player['squash'] = -0.4 # Stretch on jump
            # sfx: jump
            for _ in range(10):
                self.particles.append(self._create_particle(self.player['x'], self.player['y'] + self.PLAYER_SIZE / 2, (1,1,1)))


    def _update_physics(self):
        # Update moving platforms
        for p in self.platforms:
            if p.get('moving'):
                if self.steps > 0 and self.steps % 300 == 0:
                    p['speed'] += 0.05
                p['rect'].y += p['speed'] * p['direction']
                if abs(p['rect'].y - p['start_y']) > p['range']:
                    p['direction'] *= -1

        # Player physics
        # Friction
        if self.player['on_ground']:
            self.player['vx'] *= self.PLAYER_FRICTION
        else:
            self.player['vx'] *= 0.98 # Air friction

        self.player['vx'] = np.clip(self.player['vx'], -self.MAX_VEL_X, self.MAX_VEL_X)
        if abs(self.player['vx']) < 0.1: self.player['vx'] = 0

        # Update position
        self.player['x'] += self.player['vx']
        
        self.player['vy'] += self.GRAVITY
        self.player['y'] += self.player['vy']

        self.player['on_ground'] = False

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player['x'] - self.PLAYER_SIZE/2, self.player['y'] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Platforms
        for p in self.platforms:
            if player_rect.colliderect(p['rect']):
                # Check if player was above in the last frame and is moving down
                if self.player['vy'] > 0 and (player_rect.bottom - self.player['vy']) <= p['rect'].top:
                    if not self.player['on_ground']: # First frame of landing
                        # sfx: land
                        self.player['squash'] = 0.5 # Squash on land
                        if p['type'] == 'safe':
                            self.reward_this_step -= 0.2
                        elif p['type'] == 'risky':
                            self.reward_this_step += 2.0
                    
                    self.player['on_ground'] = True
                    self.player['vy'] = 0
                    self.player['y'] = p['rect'].top - self.PLAYER_SIZE / 2
                    
                    # Stick to moving platforms
                    if p.get('moving'):
                        self.player['y'] += p['speed'] * p['direction']

        # Coins
        new_coins = []
        for coin in self.coins:
            if player_rect.colliderect(coin):
                self.score += 1
                self.reward_this_step += 1.0
                # sfx: coin collect
                for _ in range(15):
                    self.particles.append(self._create_particle(coin.centerx, coin.centery, self.COLOR_COIN, life=15))
            else:
                new_coins.append(coin)
        self.coins = new_coins

        # Flag
        flag_rect = pygame.Rect(self.flag_pos[0], self.flag_pos[1], 10, 10)
        if player_rect.colliderect(flag_rect):
            self.victory = True
            self.game_over = True

    def _update_game_state(self):
        self.steps += 1
        self.timer -= 1
        self.reward_this_step += 0.01 # Small survival reward

        # Update camera
        self.camera_x = self.player['x'] - self.WIDTH / 3

        # Update squash effect
        if self.player['squash'] != 0:
            self.player['squash'] -= np.sign(self.player['squash']) * 0.1
            if abs(self.player['squash']) < 0.1:
                self.player['squash'] = 0

        # Update particles
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0:
                self.particles.append(p)
        
        # Add player trail
        if self.steps % 2 == 0 and abs(self.player['vx']) > 1:
            self.particles.append(self._create_particle(self.player['x'], self.player['y'], (0.5,0.5,0.5), life=8))

    def _check_termination(self):
        return (
            self.player['y'] > self.HEIGHT + self.PLAYER_SIZE * 2 or
            self.timer <= 0 or
            self.steps >= self.MAX_EPISODE_STEPS or
            self.victory
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background tiles
        tile_size = 50
        start_x = int(-self.camera_x % tile_size) - tile_size
        start_y = 0
        for x in range(start_x, self.WIDTH, tile_size):
            for y in range(start_y, self.HEIGHT, tile_size):
                if (int(x/tile_size) + int(y/tile_size)) % 2 == 0:
                    pygame.draw.rect(self.screen, self.COLOR_BG_TILE, (x, y, tile_size, tile_size))

        # Parallax stars
        for star in self.stars:
            screen_x = (star['x'] - self.camera_x * star['depth']) % self.WIDTH
            size = int(star['depth'] * 2)
            color_val = 50 + int(star['depth'] * 150)
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(screen_x), int(star['y']), size, size))

        # Platforms
        for p in self.platforms:
            screen_rect = p['rect'].move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            color = self.COLOR_PLATFORM_RISKY if p['type'] == 'risky' else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), screen_rect.move(0, 4), border_radius=3)

        # Coins
        for coin in self.coins:
            screen_rect = coin.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            
            # Spinning animation
            anim_phase = (self.steps + coin.x) * 0.2
            width_scale = abs(math.sin(anim_phase))
            
            w = int(coin.width * width_scale)
            h = coin.height
            x = int(screen_rect.centerx - w / 2)
            y = int(screen_rect.y)
            
            if w > 1:
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, (x, y, w, h))
                pygame.draw.ellipse(self.screen, tuple(c*0.7 for c in self.COLOR_COIN), (x, y, w, h), 1)

        # Flag
        fx, fy = self.flag_pos
        pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, (fx - self.camera_x, fy), (fx - self.camera_x, fy + 60), 3)
        flag_points = [
            (fx - self.camera_x, fy),
            (fx - self.camera_x + 30, fy + 15),
            (fx - self.camera_x, fy + 30)
        ]
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)

        # Particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = tuple(c * alpha for c in p['color'])
            size = int(p['size'] * alpha)
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['x'] - self.camera_x), int(p['y'])), size)

        # Player
        stretch_w = 1.0 - self.player['squash']
        stretch_h = 1.0 + self.player['squash']
        w = int(self.PLAYER_SIZE * stretch_w)
        h = int(self.PLAYER_SIZE * stretch_h)
        px = int(self.player['x'] - self.camera_x - w / 2)
        py = int(self.player['y'] - h / 2)
        
        player_rect_render = pygame.Rect(px, py, w, h)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_render, border_radius=4)
    
    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.timer / self.FPS)
        time_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player['x'],
            "player_y": self.player['y'],
            "time_left": max(0, self.timer / self.FPS)
        }

    def _create_particle(self, x, y, color_mod=(1,1,1), life=20):
        return {
            'x': x, 'y': y,
            'vx': self.np_random.uniform(-1.5, 1.5),
            'vy': self.np_random.uniform(-1.5, 1.5),
            'life': life, 'max_life': life,
            'color': (
                self.np_random.uniform(150, 255) * color_mod[0],
                self.np_random.uniform(150, 255) * color_mod[1],
                self.np_random.uniform(150, 255) * color_mod[2]
            ),
            'size': self.np_random.uniform(2, 5)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Procedural Platformer")

    terminated = False
    total_reward = 0
    
    # --- Human Controls ---
    # Map keyboard keys to MultiDiscrete actions
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    action = np.array([0, 0, 0])
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Update action based on key presses
        action[0] = 0 # No movement by default
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        # The environment already rendered to its internal surface in step()
        # We just need to blit it to the display surface
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            terminated = False

    env.close()