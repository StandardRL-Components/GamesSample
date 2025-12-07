
# Generated: 2025-08-27T23:38:36.694810
# Source Brief: brief_03534.md
# Brief Index: 3534

        
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
        "Controls: ↑ to fly up, ↓ to fly down. Avoid trees and collect coins!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Soar through a procedurally generated forest as a bird, dodging obstacles and collecting coins to reach the target score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 2000
    WIN_SCORE = 50

    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLAYER = (255, 255, 0) # Bright Yellow
    COLOR_COIN = (255, 215, 0)   # Gold
    COLOR_OBSTACLE_TRUNK = (139, 69, 19) # Saddle Brown
    COLOR_OBSTACLE_LEAVES = (34, 139, 34) # Forest Green
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128) # Semi-transparent black

    # Physics & Gameplay
    GRAVITY = 0.4
    LIFT = -1.2
    MAX_VEL_Y = 8
    INITIAL_SCROLL_SPEED = 4.0
    
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
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel_y = None
        self.player_rect = None
        self.scroll_speed = None
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.bg_layers = []
        self.obstacle_spawn_timer = 0
        self.coin_spawn_timer = 0
        self.coins_collected_since_speedup = 0
        self.prev_dist_to_coin = 0
        self.prev_dist_to_obstacle = 0
        self.win = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT / 2)
        self.player_vel_y = 0
        self.player_rect = pygame.Rect(0, 0, 24, 24)
        self.player_rect.center = self.player_pos

        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.coins_collected_since_speedup = 0

        self.obstacles = []
        self.coins = []
        self.particles = []
        self.bg_layers = [[] for _ in range(3)]

        self.obstacle_spawn_timer = 0
        self.coin_spawn_timer = 0

        # Pre-populate the world for a smooth start
        for i in range(self.SCREEN_WIDTH // 200 + 2):
            self._spawn_obstacle(x_pos=i * 300 + 500)
        for i in range(self.SCREEN_WIDTH // 300 + 2):
            self._spawn_coin_cluster(x_pos=i * 350 + 600)
        for i in range(self.SCREEN_WIDTH // 100 + 2):
            self._spawn_bg_elements(x_pos=i * 100)
        
        self.prev_dist_to_coin = self._get_dist_to_nearest_coin()
        self.prev_dist_to_obstacle = self._get_dist_to_nearest_obstacle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack action and update player
        movement = action[0]
        if movement == 1: # UP
            self.player_vel_y += self.LIFT
        elif movement == 2: # DOWN
            self.player_vel_y -= self.LIFT / 2
        
        self.player_vel_y += self.GRAVITY
        self.player_vel_y = np.clip(self.player_vel_y, -self.MAX_VEL_Y, self.MAX_VEL_Y)
        self.player_pos.y += self.player_vel_y
        self.player_pos.y = np.clip(self.player_pos.y, self.player_rect.height/2, self.SCREEN_HEIGHT-self.player_rect.height/2)
        self.player_rect.center = self.player_pos
        
        # 2. Update world state
        self._update_world_scroll()
        self._cleanup_offscreen_elements()
        self._spawn_new_elements()

        # 3. Handle collisions and events
        event_reward = self._handle_collisions()

        # 4. Update difficulty
        if self.coins_collected_since_speedup >= 5:
            self.scroll_speed += 0.05
            self.coins_collected_since_speedup = 0

        # 5. Calculate reward
        reward = self._calculate_reward(event_reward)
        
        # 6. Check termination conditions
        self.steps += 1
        if self.score >= self.WIN_SCORE:
            self.win = True
            event_reward += 100 # Add win bonus to final frame reward
        if self.steps >= self.MAX_STEPS and not self.win:
            self.game_over = True
        
        terminated = self.game_over or self.win
        if terminated and not self.game_over: # Win scenario
            # SFX: win_jingle.wav
            pass

        reward = event_reward + self._calculate_reward(0)
        
        self.clock.tick(30)
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_world_scroll(self):
        for obs in self.obstacles:
            obs['top'].x -= self.scroll_speed
            obs['bottom'].x -= self.scroll_speed
        for coin in self.coins:
            coin['rect'].x -= self.scroll_speed
        for p in self.particles:
            p['pos'].x -= self.scroll_speed
            p['pos'] += p['vel']
            p['life'] -= 1
        for i, layer in enumerate(self.bg_layers):
            speed_multiplier = 0.2 + 0.2 * i
            for bg_obj in layer:
                bg_obj.x -= self.scroll_speed * speed_multiplier

    def _cleanup_offscreen_elements(self):
        self.obstacles = [obs for obs in self.obstacles if obs['top'].right > 0]
        self.coins = [c for c in self.coins if c['rect'].right > 0]
        self.particles = [p for p in self.particles if p['life'] > 0]
        for i, layer in enumerate(self.bg_layers):
            self.bg_layers[i] = [bg_obj for bg_obj in layer if bg_obj.right > 0]

    def _spawn_new_elements(self):
        self.obstacle_spawn_timer -= self.scroll_speed
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = self.np_random.integers(280, 350)
        self.coin_spawn_timer -= self.scroll_speed
        if self.coin_spawn_timer <= 0:
            self._spawn_coin_cluster()
            self.coin_spawn_timer = self.np_random.integers(180, 250)
        if len(self.bg_layers[0]) < 10 and self.np_random.random() < 0.1:
            self._spawn_bg_elements()

    def _handle_collisions(self):
        event_reward = 0
        
        # Coin collection
        collided_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin['rect']):
                collided_coins.append(coin)
                self.score += 1
                self.coins_collected_since_speedup += 1
                event_reward += 10
                # SFX: coin_collect.wav
                self._create_coin_particles(coin['rect'].center)
        self.coins = [c for c in self.coins if c not in collided_coins]

        # Obstacle collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['top']) or self.player_rect.colliderect(obs['bottom']):
                self.game_over = True
                event_reward = -100
                # SFX: crash.wav
                self._create_crash_particles(self.player_rect.center)
                break
        return event_reward

    def _calculate_reward(self, event_reward):
        dist_to_coin = self._get_dist_to_nearest_coin()
        dist_to_obstacle = self._get_dist_to_nearest_obstacle()
        
        reward_coin_proximity = (self.prev_dist_to_coin - dist_to_coin) * 0.1
        reward_obstacle_proximity = (dist_to_obstacle - self.prev_dist_to_obstacle) * 0.01

        continuous_reward = np.clip(reward_coin_proximity + reward_obstacle_proximity, -0.5, 0.5)

        self.prev_dist_to_coin = dist_to_coin
        self.prev_dist_to_obstacle = dist_to_obstacle
        
        return event_reward + continuous_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render parallax background
        bg_colors = [(100, 150, 100, 100), (80, 120, 80, 150), (60, 90, 60, 200)] 
        for i, layer in reversed(list(enumerate(self.bg_layers))):
            for bg_obj in layer:
                pygame.draw.rect(self.screen, bg_colors[i], bg_obj)

        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_TRUNK, obs['top'].inflate(-obs['top'].width * 0.5, 0))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_TRUNK, obs['bottom'].inflate(-obs['bottom'].width * 0.5, 0))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_LEAVES, obs['top'])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_LEAVES, obs['bottom'])
            
        # Render coins
        for coin in self.coins:
            t = (pygame.time.get_ticks() / 200.0) + coin['anim_offset']
            bob = math.sin(t) * 3
            scale = 0.8 + 0.2 * abs(math.sin(t * 0.8))
            w, h = int(coin['rect'].width * scale), coin['rect'].height
            cx, cy = coin['rect'].centerx, coin['rect'].centery + int(bob)
            pygame.gfxdraw.filled_ellipse(self.screen, cx, cy, w//2, h//2, self.COLOR_COIN)
            pygame.gfxdraw.aaellipse(self.screen, cx, cy, w//2, h//2, self.COLOR_COIN)
            pygame.gfxdraw.filled_ellipse(self.screen, cx-w//4, cy-h//4, w//8, h//4, (255, 255, 150))

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['size'], p['size']))

        # Render player
        if not self.game_over:
            angle = self.player_vel_y * 2.5
            p1 = pygame.Vector2(12, 0).rotate(-angle) + self.player_pos
            p2 = pygame.Vector2(-12, 10).rotate(-angle) + self.player_pos
            p3 = pygame.Vector2(-12, -10).rotate(-angle) + self.player_pos
            points = [tuple(p1), tuple(p2), tuple(p3)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            eye_pos = pygame.Vector2(6, -2).rotate(-angle) + self.player_pos
            pygame.gfxdraw.filled_circle(self.screen, int(eye_pos.x), int(eye_pos.y), 2, (0,0,0))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        text_rect = score_text.get_rect(topleft=(10, 10))
        bg_rect = text_rect.inflate(20, 10)
        ui_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, bg_rect.topleft)
        self.screen.blit(score_text, text_rect)
        
        message = "YOU WIN!" if self.win else "GAME OVER" if self.game_over else ""
        if message:
            large_text = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            large_rect = large_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            bg_rect_large = large_rect.inflate(40, 20)
            ui_surf_large = pygame.Surface(bg_rect_large.size, pygame.SRCALPHA)
            ui_surf_large.fill(self.COLOR_UI_BG)
            self.screen.blit(ui_surf_large, bg_rect_large.topleft)
            self.screen.blit(large_text, large_rect)

    def _spawn_obstacle(self, x_pos=None):
        x = x_pos if x_pos is not None else self.SCREEN_WIDTH + 50
        gap_size = self.np_random.integers(150, 200)
        gap_y = self.np_random.integers(gap_size, self.SCREEN_HEIGHT - gap_size)
        top_height = gap_y - gap_size / 2
        bottom_y = gap_y + gap_size / 2
        obstacle_width = 80
        self.obstacles.append({
            'top': pygame.Rect(x, 0, obstacle_width, top_height),
            'bottom': pygame.Rect(x, bottom_y, obstacle_width, self.SCREEN_HEIGHT - bottom_y)
        })

    def _spawn_coin(self, x, y):
        self.coins.append({'rect': pygame.Rect(x, y, 16, 16), 'anim_offset': self.np_random.random() * 2 * math.pi})

    def _spawn_coin_cluster(self, x_pos=None):
        x = x_pos if x_pos is not None else self.SCREEN_WIDTH + 100
        start_y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
        for i in range(self.np_random.integers(2, 5)):
             self._spawn_coin(x + i * 30, start_y + self.np_random.integers(-20, 20))

    def _spawn_bg_elements(self, x_pos=None):
        for i, layer in enumerate(self.bg_layers):
            x = (x_pos if x_pos is not None else self.SCREEN_WIDTH) + self.np_random.integers(-50, 50)
            width = self.np_random.integers(20, 40) * (i + 1) * 0.8
            height = self.np_random.integers(50, 150) * (i + 1) * 0.8
            y = self.np_random.integers(self.SCREEN_HEIGHT - height, self.SCREEN_HEIGHT)
            layer.append(pygame.Rect(x, y, width, height))

    def _create_coin_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'size': self.np_random.integers(2, 5), 'color': self.COLOR_COIN})

    def _create_crash_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'size': self.np_random.integers(3, 7), 'color': self.COLOR_PLAYER})

    def _get_dist_to_nearest_coin(self):
        if not self.coins: return self.SCREEN_WIDTH
        player_center = self.player_pos
        future_coins = [c for c in self.coins if c['rect'].centerx > player_center.x]
        if not future_coins: return self.SCREEN_WIDTH
        return min(player_center.distance_to(c['rect'].center) for c in future_coins)

    def _get_dist_to_nearest_obstacle(self):
        if not self.obstacles: return self.SCREEN_WIDTH
        player_center = self.player_pos
        future_obs = [o for o in self.obstacles if o['top'].right > player_center.x]
        if not future_obs: return self.SCREEN_WIDTH
        
        closest_dist = float('inf')
        for obs in future_obs:
            gap_center_y = (obs['top'].bottom + obs['bottom'].top) / 2
            dist = player_center.distance_to((obs['top'].centerx, gap_center_y))
            if dist < closest_dist:
                closest_dist = dist
        return closest_dist
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()