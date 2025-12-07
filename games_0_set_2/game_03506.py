
# Generated: 2025-08-27T23:33:29.681527
# Source Brief: brief_03506.md
# Brief Index: 3506

        
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
    user_guide = "Controls: ←→ to run, ↑ or Space to jump."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced procedural platformer. Jump across gaps, collect coins, and reach the flag at the end of the level. Watch out for long falls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRAVITY = 0.5
        self.PLAYER_JUMP_STRENGTH = -10.5
        self.PLAYER_MOVE_ACCEL = 1.2
        self.PLAYER_MAX_HSPEED = 5.0
        self.PLAYER_FRICTION = 0.85
        self.MAX_STEPS = 2500
        self.LEVEL_LENGTH = 6000 # in pixels
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 20

        # Colors
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_CLOUD = (240, 248, 255) # AliceBlue
        self.COLOR_PLAYER = (255, 69, 0)   # Bright Red-Orange
        self.COLOR_PLATFORM = (119, 136, 153) # LightSlateGray
        self.COLOR_PLATFORM_TOP = (149, 166, 183)
        self.COLOR_COIN = (255, 215, 0)    # Gold
        self.COLOR_FLAG_POLE = (192, 192, 192) # Silver
        self.COLOR_FLAG = (220, 20, 60)      # Crimson
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_PARTICLE_COIN = (255, 223, 0)
        self.COLOR_PARTICLE_JUMP = (210, 180, 140) # Tan
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.level_cleared = False
        
        self.player_pos = np.array([100.0, 250.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False
        self.last_jump_x = 0.0
        
        self.camera_x = 0.0
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []

        # Start platform
        current_x = 0
        self.platforms.append(pygame.Rect(-10, 300, 250, 100))

        while current_x < self.LEVEL_LENGTH:
            progress_ratio = current_x / self.LEVEL_LENGTH
            
            last_platform = self.platforms[-1]
            
            # Platform gap size
            avg_gap = 20 + 120 * progress_ratio
            gap = self.np_random.uniform(avg_gap * 0.8, avg_gap * 1.2)
            gap = min(gap, 180) # Max horizontal distance for a jump
            
            current_x += last_platform.width + gap
            
            # Platform width
            width = self.np_random.integers(80, 250)
            
            # Platform height
            max_dy = 100
            dy = self.np_random.uniform(-max_dy, max_dy)
            new_y = np.clip(last_platform.y + dy, 150, self.HEIGHT - 50)
            
            new_platform = pygame.Rect(current_x, new_y, width, self.HEIGHT - new_y)
            self.platforms.append(new_platform)
            
            # Coin generation
            if self.np_random.random() < 0.7:
                num_coins = self.np_random.integers(1, 5)
                for i in range(num_coins):
                    coin_x = new_platform.x + (new_platform.width / (num_coins + 1)) * (i + 1)
                    coin_y = new_platform.y - 40 - self.np_random.uniform(0, 30)
                    self.coins.append({'pos': np.array([coin_x, coin_y]), 'initial_y': coin_y, 'active': True})
                    
        last_platform = self.platforms[-1]
        self.end_flag_pos = np.array([last_platform.centerx, last_platform.y])

        # Generate background clouds
        self.clouds = []
        for _ in range(30):
            x = self.np_random.uniform(-self.WIDTH, self.LEVEL_LENGTH * 1.5)
            y = self.np_random.uniform(20, 150)
            size = self.np_random.uniform(40, 120)
            speed_multiplier = self.np_random.uniform(0.1, 0.5) 
            self.clouds.append({'x': x, 'y': y, 'size': size, 'speed': speed_multiplier})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # 1. Handle Input
        is_jumping = (movement == 1 or space_held)
        is_moving_left = (movement == 3)
        is_moving_right = (movement == 4)
        
        if is_moving_left: self.player_vel[0] -= self.PLAYER_MOVE_ACCEL
        if is_moving_right: self.player_vel[0] += self.PLAYER_MOVE_ACCEL
            
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_HSPEED, self.PLAYER_MAX_HSPEED)

        if is_jumping and self.is_grounded:
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.is_grounded = False
            self.last_jump_x = self.player_pos[0]
            self._spawn_particles(self.player_pos + np.array([10, 20]), 5, self.COLOR_PARTICLE_JUMP)
            # sfx: jump.wav

        # 2. Update Physics
        old_player_pos_x = self.player_pos[0]
        
        if not self.is_grounded:
            self.player_vel[1] += self.GRAVITY
            
        self.player_pos += self.player_vel
        self.player_pos[0] = max(self.player_pos[0], self.camera_x)

        # 3. Collision Detection & Resolution
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        on_ground_this_frame = False
        
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                prev_player_bottom = (self.player_pos[1] - self.player_vel[1]) + player_rect.height
                if self.player_vel[1] >= 0 and prev_player_bottom <= plat.top + 1:
                    self.player_pos[1] = plat.top - player_rect.height
                    self.player_vel[1] = 0
                    if not self.is_grounded:
                        # sfx: land.wav
                        self._spawn_particles(self.player_pos + np.array([10, 20]), 3, self.COLOR_PARTICLE_JUMP)
                        jump_dist = self.player_pos[0] - self.last_jump_x
                        if jump_dist > 120: reward -= 5.0 # Risky jump penalty
                    on_ground_this_frame = True
                    break
        self.is_grounded = on_ground_this_frame

        # 4. Game Events
        if self.player_pos[0] > old_player_pos_x:
            reward += (self.player_pos[0] - old_player_pos_x)
        
        for coin in self.coins:
            if coin['active'] and np.linalg.norm(player_rect.center - coin['pos']) < 25:
                coin['active'] = False
                self.score += 10
                reward += 10.0
                self._spawn_particles(coin['pos'], 10, self.COLOR_PARTICLE_COIN)
                # sfx: coin.wav
    
        if self.player_pos[1] > self.HEIGHT + 50:
            self.lives -= 1
            reward -= 100.0
            # sfx: fall.wav
            if self.lives > 0: self._respawn_player()
            else: self.game_over = True

        flag_rect = pygame.Rect(self.end_flag_pos[0], self.end_flag_pos[1]-50, 10, 50)
        if player_rect.colliderect(flag_rect):
            self.level_cleared = True
            self.game_over = True
            reward += 100.0
            self.score += 1000
            # sfx: victory.wav

        # 5. Update Camera & Particles
        self._update_camera()
        self._update_particles()
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             self.game_over = True
             reward -= 100.0 # Penalty for running out of time

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _respawn_player(self):
        safe_platforms = [p for p in self.platforms if p.right < self.player_pos[0] - 30]
        last_safe_platform = safe_platforms[-1] if safe_platforms else self.platforms[0]
            
        self.player_pos = np.array([last_safe_platform.centerx, last_safe_platform.top - 50.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False

    def _update_camera(self):
        target_cam_x = self.player_pos[0] - self.WIDTH / 3
        self.camera_x += (target_cam_x - self.camera_x) * 0.1
        self.camera_x = max(0, self.camera_x)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            vel = np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)])
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color, 'size': self.np_random.uniform(2, 5)})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifespan'] > 0 and p['size'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        cam_x = int(self.camera_x)
        
        self._render_background(cam_x)
        
        visible_platforms = [p for p in self.platforms if p.right > cam_x and p.left < cam_x + self.WIDTH]
        for plat in visible_platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, 0))
            top_rect = pygame.Rect(plat.x - cam_x, plat.y, plat.width, 5)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect)

        for coin in self.coins:
            if coin['active'] and coin['pos'][0] > cam_x and coin['pos'][0] < cam_x + self.WIDTH:
                bob = math.sin(self.steps * 0.1 + coin['pos'][0]) * 5
                spin = abs(math.sin(self.steps * 0.25 + coin['pos'][0] * 0.5))
                w, h = int(spin * 14) + 2, 14
                pos = (int(coin['pos'][0] - cam_x), int(coin['initial_y'] + bob))
                rect = pygame.Rect(pos[0] - w//2, pos[1] - h//2, w, h)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, rect)
                pygame.gfxdraw.aaellipse(self.screen, pos[0], pos[1], w//2, h//2, self.COLOR_COIN)

        if self.end_flag_pos[0] > cam_x and self.end_flag_pos[0] < cam_x + self.WIDTH:
            pole_rect = pygame.Rect(self.end_flag_pos[0] - cam_x - 2, self.end_flag_pos[1] - 50, 4, 50)
            pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, pole_rect)
            flag_y = self.end_flag_pos[1] - 50 + math.sin(self.steps * 0.1) * 3
            flag_points = [(self.end_flag_pos[0] - cam_x, flag_y), (self.end_flag_pos[0] - cam_x + 30, flag_y + 10), (self.end_flag_pos[0] - cam_x, flag_y + 20)]
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

        player_screen_pos_x = int(self.player_pos[0] - cam_x)
        player_screen_pos_y = int(self.player_pos[1])
        bob = math.sin(self.steps * 0.4) * 2 if self.is_grounded and abs(self.player_vel[0]) > 0.1 else 0
        player_rect = pygame.Rect(player_screen_pos_x, player_screen_pos_y + bob, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_x):
        for cloud in self.clouds:
            screen_x = cloud['x'] - (cam_x * cloud['speed'])
            if screen_x > -cloud['size'] and screen_x < self.WIDTH:
                pygame.gfxdraw.filled_ellipse(self.screen, int(screen_x), int(cloud['y']), int(cloud['size'] * 0.6), int(cloud['size'] * 0.3), self.COLOR_CLOUD)
                pygame.gfxdraw.filled_ellipse(self.screen, int(screen_x + cloud['size']*0.3), int(cloud['y']), int(cloud['size'] * 0.7), int(cloud['size'] * 0.4), self.COLOR_CLOUD)

    def _render_text(self, text, font, pos, color, shadow_color):
        text_surf_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        self._render_text(f"SCORE: {self.score}", self.font_medium, (20, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        lives_text = "LIVES: " + "♥ " * self.lives
        self._render_text(lives_text, self.font_medium, (self.WIDTH - 160, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        time_sec = self.steps / 30
        self._render_text(f"TIME: {time_sec:.1f}", self.font_medium, (self.WIDTH // 2 - 70, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            msg = "LEVEL CLEARED!" if self.level_cleared else "GAME OVER"
            self._render_text(msg, self.font_large, (self.WIDTH // 2 - self.font_large.size(msg)[0] // 2, self.HEIGHT // 2 - 50), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
            final_score_msg = f"Final Score: {self.score}"
            self._render_text(final_score_msg, self.font_medium, (self.WIDTH // 2 - self.font_medium.size(final_score_msg)[0] // 2, self.HEIGHT // 2), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_x": self.player_pos[0],
            "level_cleared": self.level_cleared,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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