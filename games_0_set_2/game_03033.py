
# Generated: 2025-08-28T06:47:19.253721
# Source Brief: brief_03033.md
# Brief Index: 3033

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (100, 149, 237)
    COLOR_BG_BOTTOM = (15, 23, 42)
    COLOR_PLAYER = (50, 205, 50)
    COLOR_PLATFORM = (240, 248, 255)
    COLOR_ENEMY = (255, 69, 0)
    COLOR_COIN = (255, 215, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GOAL = (144, 238, 144)
    
    # Game parameters
    PLATFORM_SIZE = (60, 15)
    PLAYER_SIZE = 16
    ENEMY_SIZE = 10 # Radius
    COIN_SIZE = 6 # Radius
    HOP_SPEED = 0.08 # Progress per frame
    MAX_STEPS = 2000

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.np_random = None

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.is_hopping = False
        self.hop_start_pos = None
        self.hop_target_pos = None
        self.hop_progress = 0.0
        self.current_platform_idx = 0
        self.highest_y = self.SCREEN_HEIGHT
        self.platforms = []
        self.enemies = []
        self.coins = []
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.is_hopping = False
        self.hop_progress = 0.0
        self.current_platform_idx = 0
        
        self._generate_platforms()
        start_platform_rect = self.platforms[0]['rect']
        self.player_pos = pygame.Vector2(start_platform_rect.center)
        self.highest_y = self.player_pos.y
        
        self._generate_enemies()
        self._generate_coins()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        self.steps += 1
        
        movement = action[0]
        
        # --- Handle Player Action (Hopping) ---
        if not self.is_hopping and movement != 0:
            target_idx = self._find_hop_target(movement)
            
            if target_idx is not None:
                self.is_hopping = True
                self.hop_progress = 0.0
                self.hop_start_pos = self.player_pos.copy()
                self.hop_target_pos = pygame.Vector2(self.platforms[target_idx]['rect'].center)
                self.current_platform_idx = target_idx
                
                if self.hop_target_pos.y < self.hop_start_pos.y:
                    reward += 0.1  # Upward hop
                else:
                    reward -= 0.2  # Downward or sideways hop
            else:
                self.game_over = True
                reward -= 100  # Penalty for falling
                # Sound: fall

        # --- Update Game State ---
        hop_completion_reward = self._update_player_hop()
        reward += hop_completion_reward
        self._update_enemies()
        self._update_particles()
        
        # --- Check for Events ---
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        
        # Coin Collection
        collected_coins = []
        for i, coin in enumerate(self.coins):
            if player_rect.colliderect(coin['rect']):
                collected_coins.append(i)
                self.score += 10
                reward += 1.0
                self._create_particles(coin['pos'], self.COLOR_COIN, 20)
                # Sound: coin collect

        for i in sorted(collected_coins, reverse=True):
            del self.coins[i]

        # Enemy Collision
        if not self.is_hopping:
            for enemy in self.enemies:
                dist = self.player_pos.distance_to(enemy['pos'])
                if dist < self.PLAYER_SIZE / 2 + self.ENEMY_SIZE:
                    self.game_over = True
                    reward -= 100
                    self._create_particles(self.player_pos, self.COLOR_ENEMY, 50)
                    # Sound: explosion
                    break
        
        # --- Check Termination Conditions ---
        if not self.is_hopping and self.platforms[self.current_platform_idx]['type'] == 'goal':
            self.game_over = True
            self.score += 1000
            reward += 100
            self._create_particles(self.player_pos, self.COLOR_GOAL, 100)
            # Sound: victory
        
        if self.player_pos.y > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms = []
        start_rect = pygame.Rect(
            self.SCREEN_WIDTH / 2 - self.PLATFORM_SIZE[0] / 2,
            self.SCREEN_HEIGHT - self.PLATFORM_SIZE[1] * 3,
            *self.PLATFORM_SIZE
        )
        self.platforms.append({'rect': start_rect, 'type': 'start'})
        
        last_rect = start_rect
        for _ in range(10): # Generate a path of 10 platforms
            px, py = last_rect.center
            ny = py - self.np_random.integers(40, 80)
            nx = px + self.np_random.integers(-120, 121)
            nx = np.clip(nx, self.PLATFORM_SIZE[0]/2, self.SCREEN_WIDTH - self.PLATFORM_SIZE[0]/2)
            
            if ny < 50: break
            
            new_rect = pygame.Rect(nx - self.PLATFORM_SIZE[0]/2, ny - self.PLATFORM_SIZE[1]/2, *self.PLATFORM_SIZE)
            self.platforms.append({'rect': new_rect, 'type': 'normal'})
            last_rect = new_rect

        px, py = last_rect.center
        ny = 30
        nx = np.clip(px + self.np_random.integers(-80, 81), self.PLATFORM_SIZE[0]/2, self.SCREEN_WIDTH - self.PLATFORM_SIZE[0]/2)
        goal_rect = pygame.Rect(nx - self.PLATFORM_SIZE[0]/2, ny, *self.PLATFORM_SIZE)
        self.platforms.append({'rect': goal_rect, 'type': 'goal'})
        
        for _ in range(15):
            rect = pygame.Rect(
                self.np_random.integers(0, self.SCREEN_WIDTH - self.PLATFORM_SIZE[0]),
                self.np_random.integers(50, self.SCREEN_HEIGHT - 50),
                *self.PLATFORM_SIZE
            )
            if not any(rect.colliderect(p['rect'].inflate(20, 20)) for p in self.platforms):
                 self.platforms.append({'rect': rect, 'type': 'normal'})

    def _generate_enemies(self):
        self.enemies = []
        for i, p in enumerate(self.platforms):
            if p['type'] == 'normal' and self.np_random.random() < 0.3 and i > 0:
                speed = 1.0 + (self.steps // 500) * 0.05
                speed = min(speed, 5.0)
                self.enemies.append({
                    'pos': pygame.Vector2(p['rect'].center),
                    'platform_idx': i,
                    'vel_x': self.np_random.choice([-speed, speed]),
                })

    def _generate_coins(self):
        self.coins = []
        for i, p in enumerate(self.platforms):
             if p['type'] == 'normal' and self.np_random.random() < 0.4 and i > 0:
                 if not any(e['platform_idx'] == i for e in self.enemies):
                    pos = pygame.Vector2(p['rect'].centerx, p['rect'].centery - self.PLATFORM_SIZE[1])
                    self.coins.append({
                        'pos': pos,
                        'rect': pygame.Rect(0, 0, self.COIN_SIZE * 2, self.COIN_SIZE * 2)
                    })
                    self.coins[-1]['rect'].center = pos

    def _find_hop_target(self, direction):
        px, py = self.player_pos
        best_target, min_dist_sq = None, float('inf')
        
        for i, p in enumerate(self.platforms):
            if i == self.current_platform_idx: continue
            pcx, pcy = p['rect'].center
            
            is_candidate = False
            if direction == 1 and pcy < py: is_candidate = True # Up
            elif direction == 2 and pcy > py: is_candidate = True # Down
            elif direction == 3 and pcx < px: is_candidate = True # Left
            elif direction == 4 and pcx > px: is_candidate = True # Right
            
            if is_candidate:
                dist_sq = self.player_pos.distance_squared_to(p['rect'].center)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_target = i
                elif dist_sq == min_dist_sq and best_target is not None and self.platforms[i]['rect'].centerx < self.platforms[best_target]['rect'].centerx:
                    best_target = i
        return best_target

    def _update_player_hop(self):
        reward = 0
        if not self.is_hopping: return reward
        
        self.hop_progress = min(1.0, self.hop_progress + self.HOP_SPEED)
        t = self.hop_progress
        
        if t >= 1.0:
            self.is_hopping = False
            self.player_pos = self.hop_target_pos.copy()
            
            if self.player_pos.y < self.highest_y:
                self.score += 50
                reward += 5.0
                self.highest_y = self.player_pos.y
            
            self._create_particles(self.player_pos, self.COLOR_PLATFORM, 10, is_shockwave=True)
            # Sound: land
        else:
            self.player_pos = self.hop_start_pos.lerp(self.hop_target_pos, t)
            arc_height = math.sin(t * math.pi) * 30
            self.player_pos.y -= arc_height
        return reward

    def _update_enemies(self):
        speed = 1.0 + (self.steps // 500) * 0.05
        speed = min(speed, 5.0)
        
        for enemy in self.enemies:
            platform_rect = self.platforms[enemy['platform_idx']]['rect']
            enemy['vel_x'] = math.copysign(speed, enemy['vel_x'])
            enemy['pos'].x += enemy['vel_x']
            
            if enemy['pos'].x < platform_rect.left + self.ENEMY_SIZE:
                enemy['pos'].x = platform_rect.left + self.ENEMY_SIZE
                enemy['vel_x'] *= -1
            elif enemy['pos'].x > platform_rect.right - self.ENEMY_SIZE:
                enemy['pos'].x = platform_rect.right - self.ENEMY_SIZE
                enemy['vel_x'] *= -1

    def _create_particles(self, pos, color, count, is_shockwave=False):
        for _ in range(count):
            if is_shockwave:
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.uniform(1, 3)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
                vel.y -= 1.5
                vel.scale_to_length(self.np_random.uniform(1, 4))
            
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifetime': self.np_random.integers(15, 30), 'color': color, 'is_shockwave': is_shockwave})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['is_shockwave']: p['vel'] *= 0.9
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_platforms()
        self._render_particles()
        self._render_coins()
        self._render_enemies()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        # Simplified gradient using a rect
        gradient_height = int(self.SCREEN_HEIGHT * 0.8)
        top_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, gradient_height)
        bottom_color = self.COLOR_BG_TOP
        top_color = tuple(int(c*0.5) for c in self.COLOR_BG_TOP) # Darker at very top
        
        for y in range(gradient_height):
            interp = y / gradient_height
            color = (
                int(top_color[0] * (1 - interp) + bottom_color[0] * interp),
                int(top_color[1] * (1 - interp) + bottom_color[1] * interp),
                int(top_color[2] * (1 - interp) + bottom_color[2] * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_platforms(self):
        for p in self.platforms:
            color = self.COLOR_GOAL if p['type'] == 'goal' else self.COLOR_PLATFORM
            shadow_rect = p['rect'].move(0, 4)
            shadow_color = (0, 0, 0, 50)
            # This requires a surface with alpha. We'll skip for performance.
            # pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=3)
            pygame.draw.rect(self.screen, color, p['rect'], border_radius=3)

    def _render_coins(self):
        for coin in self.coins:
            pos = (int(coin['pos'].x), int(coin['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.COIN_SIZE, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.COIN_SIZE, (255, 255, 255, 150))
            
    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, (255,255,255,100))

    def _render_player(self):
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        
        if self.is_hopping:
            t = self.hop_progress
            squash = 1.0 - 0.4 * math.sin(t * math.pi)
            stretch = 1.0 + 0.4 * math.sin(t * math.pi)
            player_rect.width = self.PLAYER_SIZE * squash
            player_rect.height = self.PLAYER_SIZE * stretch
            player_rect.center = self.player_pos
            
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, (255, 255, 255), player_rect, width=2, border_radius=3)

    def _render_particles(self):
        if not self.particles: return
        particle_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            alpha = max(0, int(255 * (p['lifetime'] / 30.0)))
            color = (*p['color'], alpha)
            size = max(1, int(p['lifetime'] / 6))
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(particle_surf, pos[0], pos[1], size, color)
        self.screen.blit(particle_surf, (0, 0))
                 
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

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