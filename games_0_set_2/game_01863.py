
# Generated: 2025-08-28T02:56:57.086753
# Source Brief: brief_01863.md
# Brief Index: 1863

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move vertically. Press space for a short speed burst. "
        "Avoid red obstacles and collect yellow boosts!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling snail race. Dodge obstacles and use speed boosts "
        "to get to the finish line as quickly as possible. You lose after 5 hits."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WORLD_LENGTH = 6400  # 10 screens long
    MAX_STEPS = 2000
    MAX_HITS = 5
    
    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_HILLS_FAR = (140, 170, 140)
    COLOR_BG_HILLS_NEAR = (120, 160, 120)
    COLOR_TRACK = (100, 190, 80)
    COLOR_TRACK_DARK = (80, 150, 60)
    COLOR_SNAIL_SHELL = (255, 140, 0)
    COLOR_SNAIL_BODY = (240, 240, 220)
    COLOR_SNAIL_EYE = (0, 0, 0)
    COLOR_OBSTACLE = (220, 50, 50)
    COLOR_BOOST = (255, 223, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    # Snail Physics
    SNAIL_BASE_SPEED = 4.0
    SNAIL_BOOST_SPEED_MULTIPLIER = 2.0
    SNAIL_BURST_SPEED_ADDITION = 5.0
    SNAIL_VERTICAL_ACCEL = 1.0
    SNAIL_VERTICAL_DRAG = 0.9
    SNAIL_MAX_VERTICAL_SPEED = 10.0
    
    TRACK_TOP_Y = HEIGHT * 0.15
    TRACK_BOTTOM_Y = HEIGHT * 0.85
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 72)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snail_pos = [0, 0]
        self.snail_v_speed = 0.0
        self.snail_h_speed = 0.0
        self.boost_timer = 0
        self.burst_timer = 0
        self.burst_cooldown = 0
        self.camera_x = 0.0
        self.obstacle_hits = 0
        self.obstacles = []
        self.speed_boosts = []
        self.particles = []
        self.win_condition_met = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.snail_pos = [100, self.HEIGHT / 2]
        self.snail_v_speed = 0.0
        self.snail_h_speed = self.SNAIL_BASE_SPEED
        self.boost_timer = 0
        self.burst_timer = 0
        self.burst_cooldown = 0
        
        self.camera_x = 0
        self.obstacle_hits = 0
        
        self.obstacles = []
        self.speed_boosts = []
        self.particles = []

        for x in range(400, self.WORLD_LENGTH - self.WIDTH, 250):
            if self.np_random.random() < 0.8:
                obs_y = self.np_random.integers(int(self.TRACK_TOP_Y + 30), int(self.TRACK_BOTTOM_Y - 30))
                obs_speed_y = (self.np_random.random() - 0.5) * 2
                self.obstacles.append({
                    "pos": [x + self.np_random.integers(-50, 50), obs_y],
                    "size": self.np_random.integers(15, 30),
                    "speed_y": obs_speed_y
                })
            if self.np_random.random() < 0.25:
                boost_y = self.np_random.integers(int(self.TRACK_TOP_Y + 20), int(self.TRACK_BOTTOM_Y - 20))
                self.speed_boosts.append({
                    "pos": [x + self.np_random.integers(100, 150), boost_y],
                    "size": 12
                })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._update_timers()
        reward += self._handle_input(movement, space_held)
        self._update_physics()
        self._update_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                reward += 100
            else:
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        if self.boost_timer > 0: self.boost_timer -= 1
        if self.burst_timer > 0: self.burst_timer -= 1
        if self.burst_cooldown > 0: self.burst_cooldown -= 1

    def _handle_input(self, movement, space_held):
        if movement == 1: self.snail_v_speed -= self.SNAIL_VERTICAL_ACCEL
        if movement == 2: self.snail_v_speed += self.SNAIL_VERTICAL_ACCEL
        
        if space_held and self.burst_cooldown <= 0:
            self.burst_timer = 15 # 0.5s
            self.burst_cooldown = 60 # 2s cooldown
            # sfx_burst.wav
            for _ in range(20):
                self._create_particle(self.snail_pos, (100, 100, 255), 3, 20)
        return 0

    def _update_physics(self):
        self.snail_v_speed *= self.SNAIL_VERTICAL_DRAG
        self.snail_v_speed = np.clip(self.snail_v_speed, -self.SNAIL_MAX_VERTICAL_SPEED, self.SNAIL_MAX_VERTICAL_SPEED)
        self.snail_pos[1] += self.snail_v_speed
        self.snail_pos[1] = np.clip(self.snail_pos[1], self.TRACK_TOP_Y, self.TRACK_BOTTOM_Y)
        
        base_speed = self.SNAIL_BASE_SPEED + 0.05 * (self.steps // 200)
        boost_multiplier = self.SNAIL_BOOST_SPEED_MULTIPLIER if self.boost_timer > 0 else 1.0
        burst_addition = self.SNAIL_BURST_SPEED_ADDITION if self.burst_timer > 0 else 0.0
        self.snail_h_speed = base_speed * boost_multiplier + burst_addition
        
        self.snail_pos[0] += self.snail_h_speed
        self.camera_x = self.snail_pos[0] - 100

    def _update_entities(self):
        for obs in self.obstacles:
            obs['pos'][1] += obs['speed_y']
            if obs['pos'][1] < self.TRACK_TOP_Y + obs['size'] or obs['pos'][1] > self.TRACK_BOTTOM_Y - obs['size']:
                obs['speed_y'] *= -1
        
        self.obstacles = [o for o in self.obstacles if o['pos'][0] > self.camera_x - 50]
        self.speed_boosts = [b for b in self.speed_boosts if b['pos'][0] > self.camera_x - 50]

        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0.1 # Base reward for moving forward
        snail_rect = pygame.Rect(self.snail_pos[0] - 15, self.snail_pos[1] - 15, 30, 30)
        
        for obs in self.obstacles[:]:
            obs_rect = pygame.Rect(obs['pos'][0] - obs['size'], obs['pos'][1] - obs['size'], obs['size']*2, obs['size']*2)
            if snail_rect.colliderect(obs_rect):
                self.obstacles.remove(obs)
                self.obstacle_hits += 1
                reward -= 20
                # sfx_hit.wav
                for _ in range(30):
                    self._create_particle(self.snail_pos, self.COLOR_OBSTACLE, 4, 25)
        
        for boost in self.speed_boosts[:]:
            boost_rect = pygame.Rect(boost['pos'][0] - boost['size'], boost['pos'][1] - boost['size'], boost['size']*2, boost['size']*2)
            if snail_rect.colliderect(boost_rect):
                self.speed_boosts.remove(boost)
                self.boost_timer = 90 # 3 seconds
                reward += 5
                # sfx_boost.wav
                for _ in range(30):
                    self._create_particle(self.snail_pos, self.COLOR_BOOST, 3, 30)
        return reward

    def _check_termination(self):
        if self.snail_pos[0] >= self.WORLD_LENGTH:
            self.win_condition_met = True
            return True
        if self.obstacle_hits >= self.MAX_HITS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
        
    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "hits": self.obstacle_hits}

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_SKY)
        
        # Parallax hills
        for i in range(-1, self.WIDTH // 200 + 2):
            random.seed(i)
            h = random.randint(100, 150)
            x = (i * 200 - int(self.camera_x * 0.1)) % (self.WIDTH + 200) - 200
            pygame.gfxdraw.filled_ellipse(self.screen, int(x), int(self.HEIGHT - h / 2), 200, h, self.COLOR_BG_HILLS_FAR)
        for i in range(-1, self.WIDTH // 300 + 2):
            random.seed(i + 1000)
            h = random.randint(150, 200)
            x = (i * 300 - int(self.camera_x * 0.3)) % (self.WIDTH + 300) - 300
            pygame.gfxdraw.filled_ellipse(self.screen, int(x), int(self.HEIGHT - h / 2), 300, h, self.COLOR_BG_HILLS_NEAR)

        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_TOP_Y, self.WIDTH, self.HEIGHT))
        
        # Speed lines
        for i in range(10):
            y = self.TRACK_TOP_Y + i * (self.TRACK_BOTTOM_Y - self.TRACK_TOP_Y) / 9
            start_x = (0 - self.camera_x * 1.5) % 100 - 100
            while start_x < self.WIDTH:
                pygame.draw.line(self.screen, self.COLOR_TRACK_DARK, (start_x, y), (start_x + 30, y), 1)
                start_x += 100

    def _render_game_objects(self):
        # Finish line
        finish_x = self.WORLD_LENGTH - self.camera_x
        if finish_x < self.WIDTH + 50:
            for i in range(10):
                color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (finish_x, self.TRACK_TOP_Y + i * (self.HEIGHT/10), 20, self.HEIGHT/10))
                
        # Obstacles
        for obs in self.obstacles:
            x, y = int(obs['pos'][0] - self.camera_x), int(obs['pos'][1])
            size = int(obs['size'])
            if -size < x < self.WIDTH + size:
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_OBSTACLE)

        # Speed boosts
        for boost in self.speed_boosts:
            x, y = int(boost['pos'][0] - self.camera_x), int(boost['pos'][1])
            size = int(boost['size'])
            if -size < x < self.WIDTH + size:
                # Glow
                glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
                temp_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.COLOR_BOOST + (int(glow_alpha),), (size*2, size*2), int(size*1.5))
                self.screen.blit(temp_surf, (x - size*2, y - size*2))
                # Core
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_BOOST)
                pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_BOOST)

        # Particles
        for p in self.particles:
            x, y = int(p['pos'][0] - self.camera_x), int(p['pos'][1])
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(p['size']), color)
            
        self._draw_snail()

    def _draw_snail(self):
        x = int(self.snail_pos[0] - self.camera_x)
        y = int(self.snail_pos[1])

        # Boost/Burst glow
        if self.boost_timer > 0 or self.burst_timer > 0:
            alpha = 100 if (self.steps // 3) % 2 == 0 else 50
            glow_color = self.COLOR_BOOST if self.boost_timer > 0 else (100, 100, 255)
            temp_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color + (alpha,), (40, 40), 30)
            self.screen.blit(temp_surf, (x - 40, y - 40))

        # Body
        pygame.gfxdraw.filled_ellipse(self.screen, x - 5, y + 2, 25, 12, self.COLOR_SNAIL_BODY)
        pygame.gfxdraw.aaellipse(self.screen, x - 5, y + 2, 25, 12, self.COLOR_SNAIL_BODY)
        # Head
        pygame.gfxdraw.filled_circle(self.screen, x + 15, y - 5, 10, self.COLOR_SNAIL_BODY)
        pygame.gfxdraw.aacircle(self.screen, x + 15, y - 5, 10, self.COLOR_SNAIL_BODY)
        
        # Shell with simple spiral
        shell_rect = pygame.Rect(x - 25, y - 25, 30, 30)
        shell_outline_color = tuple(max(0, c-40) for c in self.COLOR_SNAIL_SHELL)
        pygame.draw.arc(self.screen, shell_outline_color, shell_rect.inflate(4, 4), math.pi / 2, math.pi * 2, 6)
        pygame.draw.arc(self.screen, self.COLOR_SNAIL_SHELL, shell_rect, math.pi / 2, math.pi * 2, 15)
        
        # Eye
        eye_x = x + 18
        eye_y_offset = np.clip(self.snail_v_speed * 0.5, -3, 3)
        eye_y = y - 8 + int(eye_y_offset)
        pygame.draw.circle(self.screen, (255,255,255), (eye_x, eye_y), 5)
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_EYE, (eye_x, eye_y), 3)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, pos, color, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Hits display
        draw_text("HITS:", self.font_small, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        for i in range(self.MAX_HITS):
            color = self.COLOR_OBSTACLE if i < self.obstacle_hits else (80, 80, 80)
            pygame.draw.circle(self.screen, color, (100 + i * 25, 25), 10)
            pygame.draw.circle(self.screen, (0,0,0), (100 + i * 25, 25), 10, 2)
            
        # Timer
        time_text = f"TIME: {self.steps / self.FPS:.1f}s"
        draw_text(time_text, self.font_small, (self.WIDTH - 180, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over / Finish message
        if self.game_over:
            message = "GAME OVER"
            if self.win_condition_met:
                message = "FINISH!"
            
            text_width, text_height = self.font_large.size(message)
            draw_text(message, self.font_large, (self.WIDTH // 2 - text_width // 2, self.HEIGHT // 2 - text_height // 2), 
                      self.COLOR_BOOST, self.COLOR_TEXT_SHADOW)
    
    def _create_particle(self, pos, color, speed_scale, life):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, speed_scale)
        vel = [math.cos(angle) * speed - self.snail_h_speed * 0.3, math.sin(angle) * speed]
        self.particles.append({
            'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(life // 2, life),
            'color': color, 'size': self.np_random.integers(2, 5)
        })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")