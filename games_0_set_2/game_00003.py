import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to jump up, ←→ to jump left/right, ↓ to boost down. "
        "Hold Shift for a fast dash. Press Space for a long jump."
    )

    game_description = (
        "Navigate a hopping spaceship through a vertically scrolling asteroid field. "
        "Dodge obstacles to score points and reach the top. Difficulty increases as you climb."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array", seed=None):
        super().__init__()
        self.np_random = np.random.default_rng(seed)

        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_HEIGHT = 20000
        self.FPS = 30

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_STAR_1 = (80, 80, 100)
        self.COLOR_STAR_2 = (120, 120, 140)

        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.player_pos = None
        self.player_vel = None
        self.player_size = 20
        self.jump_cooldown = 0
        self.last_move_dir = np.array([0, -1])
        
        self.obstacles = []
        self.particles = []
        self.stars = []
        
        self.camera_y = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.level = 1
        
        self.difficulty_tier = 0
        self.base_obstacle_speed = 2.0
        self.max_obstacles = 15


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.LEVEL_HEIGHT - self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0])
        self.jump_cooldown = 0
        self.last_move_dir = np.array([0, -1])

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.difficulty_tier = 0
        self.base_obstacle_speed = 2.0
        self.max_obstacles = 15

        self.camera_y = self.LEVEL_HEIGHT - self.HEIGHT
        self.obstacles = []
        self.particles = []
        self._init_stars()

        for _ in range(self.max_obstacles):
            self._spawn_obstacle(initial_spawn=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward

        self._handle_input(action)
        self._update_player()
        self._update_camera()
        self._update_obstacles()
        self._update_particles()
        
        terminated = self._check_collisions()
        
        reward += self._calculate_stagnation_penalty()
        reward += self._calculate_dodge_reward()
        
        if terminated:
            reward = -10.0
            self.game_over = True
            self._create_explosion(self.player_pos, 50)
        
        if self.player_pos[1] < self.player_size:
            reward = 100.0
            terminated = True
            self.game_over = True

        self.steps += 1
        truncated = False
        if self.steps >= 10000:
            truncated = True

        new_score = int((self.LEVEL_HEIGHT - self.player_pos[1]) / 10)
        self.score = max(self.score, new_score)
        self._update_difficulty()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return

        jump_applied = False
        
        if movement in [1, 2, 3, 4]:
            dirs = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}
            self.last_move_dir = np.array(dirs[movement])

        if shift_held:
            impulse = self.last_move_dir * 8.0
            self.player_vel += impulse
            self.jump_cooldown = 10
            jump_applied = True
        elif space_held:
            self.player_vel[1] -= 7.0
            self.jump_cooldown = 15
            jump_applied = True
        elif movement > 0:
            impulse = self.last_move_dir * 5.0
            if movement == 2: impulse *= 0.6
            self.player_vel += impulse
            self.jump_cooldown = 12
            jump_applied = True
        
        if jump_applied:
            self._create_jump_particles(self.player_pos)

    def _update_player(self):
        self.player_vel[1] += 0.3
        self.player_vel[1] = min(self.player_vel[1], 5)
        
        self.player_pos += self.player_vel
        self.player_vel[0] *= 0.95
        
        if self.player_pos[0] < self.player_size / 2 or self.player_pos[0] > self.WIDTH - self.player_size / 2:
            self.player_vel[0] *= -0.5
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size / 2, self.WIDTH - self.player_size / 2)
            
        player_screen_y = self.player_pos[1] - self.camera_y
        if player_screen_y > self.HEIGHT - self.player_size / 2:
            self.player_pos[1] = self.camera_y + self.HEIGHT - self.player_size / 2
            self.player_vel[1] = min(0, self.player_vel[1])

    def _update_camera(self):
        target_cam_y = self.player_pos[1] - self.HEIGHT / 2
        self.camera_y += (target_cam_y - self.camera_y) * 0.05
        # Clip camera to level bounds. The original min() call incorrectly
        # pinned the camera at the start, causing inevitable collisions.
        self.camera_y = np.clip(self.camera_y, 0, self.LEVEL_HEIGHT - self.HEIGHT)

    def _update_obstacles(self):
        # Remove obstacles that are below the screen view
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].bottom > self.camera_y]
        
        while len(self.obstacles) < self.max_obstacles:
            self._spawn_obstacle()
            
        for obs in self.obstacles:
            obs['rect'].y += obs['speed']
            
    def _spawn_obstacle(self, initial_spawn=False):
        w = self.np_random.integers(40, 120)
        h = self.np_random.integers(15, 30)
        x = self.np_random.integers(0, self.WIDTH - w)
        
        if initial_spawn:
            # On reset, spawn all obstacles above the player's starting position
            # to ensure a safe start and prevent immediate termination from falling.
            player_start_y = self.LEVEL_HEIGHT - self.HEIGHT / 2
            y_range_top = player_start_y - self.player_size
            y_range_bottom = player_start_y - self.HEIGHT * 2
            y = self.np_random.integers(y_range_bottom, y_range_top)
        else:
            # During gameplay, spawn new obstacles just above the camera's view.
            y_range_top = self.camera_y - 20
            y_range_bottom = self.camera_y - self.HEIGHT
            y = self.np_random.integers(y_range_bottom, y_range_top)
            
        speed = self.base_obstacle_speed + self.np_random.uniform(-0.5, 1.0)
        rect = pygame.Rect(x, y, w, h)
        
        self.obstacles.append({'rect': rect, 'speed': speed, 'dodged': False})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size / 2,
            self.player_pos[1] - self.player_size / 2,
            self.player_size, self.player_size
        )
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                return True
        return False
        
    def _calculate_stagnation_penalty(self):
        return -0.02 if abs(self.player_vel[0]) < 0.1 else 0.0

    def _calculate_dodge_reward(self):
        dodge_reward = 0
        player_bottom = self.player_pos[1] + self.player_size / 2
        for obs in self.obstacles:
            if not obs['dodged'] and player_bottom < obs['rect'].top:
                obs['dodged'] = True
                dist = abs(self.player_pos[0] - obs['rect'].centerx)
                if dist < obs['rect'].width / 2 + self.player_size * 2:
                    dodge_reward += 1.0
        return dodge_reward
        
    def _update_difficulty(self):
        current_tier = self.score // 500
        if current_tier > self.difficulty_tier:
            self.difficulty_tier = current_tier
            self.level = self.difficulty_tier + 1
            self.base_obstacle_speed += 0.05
            self.max_obstacles = min(40, int(15 * (1.05 ** self.difficulty_tier)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
        
    def _render_stars(self):
        for star in self.stars:
            x, y, z = star['pos']
            screen_y = (y - self.camera_y * z) % self.HEIGHT
            size = max(1, int(z * 2))
            color = self.COLOR_STAR_1 if z < 0.5 else self.COLOR_STAR_2
            pygame.draw.rect(self.screen, color, (int(x), int(screen_y), size, size))
            
    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_rect = obs['rect'].copy()
            screen_rect.y -= self.camera_y
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 150, 150), screen_rect, 2, border_radius=3)

    def _render_player(self):
        if self.game_over:
            return
            
        hop_scale = 1.0 - min(0.3, max(-0.2, self.player_vel[1] * 0.05))
        size = self.player_size * hop_scale
        
        player_screen_pos = self.player_pos - np.array([0, self.camera_y])
        x, y = int(player_screen_pos[0]), int(player_screen_pos[1])
        
        points = [
            (x, y - size * 0.8),
            (x - size * 0.5, y + size * 0.4),
            (x + size * 0.5, y + size * 0.4)
        ]
        
        glow_size = int(size * 1.8)
        glow_points = [
            (x, y - glow_size * 0.8),
            (x - glow_size * 0.5, y + glow_size * 0.4),
            (x + glow_size * 0.5, y + glow_size * 0.4)
        ]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, (*self.COLOR_PLAYER, 50))
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*self.COLOR_PLAYER, 50))
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'] - np.array([0, self.camera_y])
            life_ratio = p['life'] / p['max_life']
            size = max(0, int(p['size'] * life_ratio))
            if size > 0:
                color = tuple(c * life_ratio for c in p['color'])
                color_with_alpha = (*color, 255)
                try:
                    pygame.draw.rect(self.screen, color_with_alpha, (int(pos[0]), int(pos[1]), size, size))
                except TypeError: # Fallback for colors that might become invalid
                    pygame.draw.rect(self.screen, (255,255,255,100), (int(pos[0]), int(pos[1]), size, size))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))
        
        if self.game_over:
            over_text = self.font_main.render("GAME OVER", True, self.COLOR_OBSTACLE)
            over_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), over_rect.inflate(20,20))
            self.screen.blit(over_text, over_rect)

    def _init_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT), self.np_random.uniform(0.1, 0.6)],
            })
            
    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'color': random.choice([self.COLOR_OBSTACLE, (255, 150, 0), (255, 255, 255)]),
                'size': self.np_random.integers(2, 5)
            })
            
    def _create_jump_particles(self, pos):
        for _ in range(5):
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            vel += self.player_vel * 0.2
            life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pos.copy() + np.array([0, self.player_size * 0.4]),
                'vel': vel, 'life': life, 'max_life': life,
                'color': self.COLOR_PLAYER, 'size': self.np_random.integers(2, 4)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def close(self):
        pygame.quit()