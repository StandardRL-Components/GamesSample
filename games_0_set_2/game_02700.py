
# Generated: 2025-08-27T21:12:19.470620
# Source Brief: brief_02700.md
# Brief Index: 2700

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


# Helper class for particle effects
class Particle:
    """A simple particle for visual effects like explosions or sparkles."""
    def __init__(self, pos, color, rng, gravity=0.1, lifespan_range=(20, 40), vel_range=((-4, 4), (-6, 2))):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(rng.uniform(vel_range[0][0], vel_range[0][1]), rng.uniform(vel_range[1][0], vel_range[1][1]))
        self.color = color
        self.lifespan = rng.integers(lifespan_range[0], lifespan_range[1])
        self.max_lifespan = self.lifespan
        self.size = rng.integers(3, 7)
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.lifespan -= 1
        
    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, surface):
        if not self.is_dead():
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color_with_alpha = self.color + (alpha,)
            # Create a temporary surface for alpha blending to avoid modifying the main screen's alpha
            particle_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(particle_surf, (int(self.pos.x - self.size), int(self.pos.y - self.size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Press SPACE to jump. Hold SPACE for a longer jump. Avoid obstacles and pits."
    game_description = "A fast-paced side-scrolling arcade runner. Navigate a procedural course, collect coins, and try to reach the 100-meter finish line."
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    PIXELS_PER_METER = 30
    
    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_PLAYER = (255, 69, 0)
    COLOR_OBSTACLE = (80, 80, 80)
    COLOR_COIN = (255, 215, 0)
    COLOR_GROUND = (34, 139, 34)
    COLOR_GROUND_DARK = (21, 87, 21)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    # Physics & Game Rules
    GRAVITY = 1.0
    JUMP_VELOCITY = -18.0
    JUMP_GRAVITY_MULTIPLIER = 0.5
    PLAYER_X_POS = 100
    GROUND_Y = SCREEN_HEIGHT - 50
    TRACK_LENGTH_METERS = 100
    MAX_STEPS = 10000
    INITIAL_SPEED_MPS = 5.0
    MAX_SPEED_MPS = 15.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel_y = 0.0
        self.is_on_ground = True
        self.jump_key_was_pressed = False
        self.world_offset_m = 0.0
        self.running_speed_mps = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.obstacles = deque()
        self.coins = deque()
        self.pits = deque()
        self.particles = deque()
        self.bg_layers = {}
        self.last_feature_x_m = 0.0

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.GROUND_Y)
        self.player_vel_y = 0.0
        self.is_on_ground = True
        self.jump_key_was_pressed = False

        self.world_offset_m = 0.0
        self.running_speed_mps = self.INITIAL_SPEED_MPS

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.obstacles.clear()
        self.coins.clear()
        self.pits.clear()
        self.particles.clear()

        self.bg_layers = {
            'far': {'speed_ratio': 0.1, 'color': (50, 80, 100, 100), 'items': deque()},
            'mid': {'speed_ratio': 0.3, 'color': (100, 120, 80, 150), 'items': deque()}
        }
        self.last_feature_x_m = 10.0
        self._generate_world_chunk(initial=True)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- 1. Handle Action ---
        space_held = action[1] == 1
        
        jump_initiated = space_held and not self.jump_key_was_pressed and self.is_on_ground
        if jump_initiated:
            self.player_vel_y = self.JUMP_VELOCITY
            self.is_on_ground = False
            # sfx: jump

        self.jump_key_was_pressed = space_held

        # --- 2. Update Physics & World ---
        gravity_effect = self.GRAVITY
        if not self.is_on_ground and space_held:
            gravity_effect *= self.JUMP_GRAVITY_MULTIPLIER
        self.player_vel_y += gravity_effect
        self.player_pos.y += self.player_vel_y

        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            if not self.is_on_ground: # Landing
                # sfx: land
                self._create_particles((self.player_pos.x, self.player_pos.y + 10), (150, 75, 0), 5, gravity=0.2, lifespan_range=(10,20))
            self.is_on_ground = True

        progress_ratio = min(1.0, self.world_offset_m / self.TRACK_LENGTH_METERS)
        self.running_speed_mps = self.INITIAL_SPEED_MPS + progress_ratio * (self.MAX_SPEED_MPS - self.INITIAL_SPEED_MPS)
        
        delta_dist_m = self.running_speed_mps / self.FPS
        self.world_offset_m += delta_dist_m
        reward += delta_dist_m * 0.1

        # --- 3. Update World Elements ---
        self._generate_world_chunk()
        self._cull_world_elements()
        self._update_particles()
        
        # --- 4. Handle Collisions & Events ---
        player_rect = self._get_player_rect()

        # Obstacle collision
        for obs in self.obstacles:
            if self._get_obstacle_rect(obs).colliderect(player_rect):
                self.game_over = True
                reward = -100.0
                self._create_particles(player_rect.center, self.COLOR_PLAYER, 30)
                # sfx: crash
                break
        if self.game_over: return self._get_observation(), reward, True, False, self._get_info()

        # Coin collision
        for coin in list(self.coins):
            if self._get_coin_rect(coin).colliderect(player_rect):
                self.coins.remove(coin)
                self.score += 1
                reward += 1.0
                self._create_particles((self._world_to_screen_x(coin.x), coin.y), self.COLOR_COIN, 10, vel_range=((-2,2),(-4,0)))
                # sfx: coin_collect
        
        # Pitfall
        is_over_pit = False
        if self.is_on_ground:
            for pit in self.pits:
                pit_start_x = self._world_to_screen_x(pit['x_m'])
                pit_end_x = self._world_to_screen_x(pit['x_m'] + pit['w_m'])
                if pit_start_x < player_rect.centerx < pit_end_x:
                    is_over_pit = True
                    break
            if is_over_pit:
                self.game_over = True
                reward = -100.0
                # sfx: fall
        
        # --- 5. Check Termination Conditions ---
        if not self.game_over:
            if self.world_offset_m >= self.TRACK_LENGTH_METERS:
                self.game_over = True
                self.win = True
                reward = 100.0
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
        
        self.steps += 1
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "distance_m": self.world_offset_m}

    def _render_game(self):
        self.screen.fill(self.COLOR_SKY)
        self._render_background()
        self._render_ground_and_pits()
        self._render_world_elements()
        self._render_player()
        for p in self.particles: p.draw(self.screen)
        self._render_ui()

    def _render_background(self):
        for layer in self.bg_layers.values():
            for item in layer['items']:
                x = int(self._world_to_screen_x(item.x, layer['speed_ratio']))
                shape_surf = pygame.Surface((item.width, item.height), pygame.SRCALPHA)
                pygame.draw.ellipse(shape_surf, layer['color'], (0, 0, item.width, item.height))
                self.screen.blit(shape_surf, (x, int(item.y)))

    def _render_ground_and_pits(self):
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        pygame.draw.line(self.screen, self.COLOR_GROUND_DARK, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 5)
        
        for pit in self.pits:
            x = self._world_to_screen_x(pit['x_m'])
            w = pit['w_m'] * self.PIXELS_PER_METER
            if x + w > 0 and x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_SKY, (int(x), int(self.GROUND_Y), int(w) + 1, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_world_elements(self):
        for obs in self.obstacles:
            rect = self._get_obstacle_rect(obs)
            if rect.right > 0 and rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
                pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in self.COLOR_OBSTACLE), rect.inflate(-6, -6), border_radius=3)

        for coin in self.coins:
            rect = self._get_coin_rect(coin)
            if rect.right > 0 and rect.left < self.SCREEN_WIDTH:
                pulse = math.sin(self.steps * 0.3 + coin.x) * 2
                size = int(8 + pulse)
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, size, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, size, tuple(int(c*0.8) for c in self.COLOR_COIN))
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx-2, rect.centery-2, int(size*0.4), (255,255,150))

    def _render_player(self):
        rect = self._get_player_rect()
        body_bob = math.sin(self.world_offset_m * 2) * 3 if self.is_on_ground else 0
        
        leg_angle = math.sin(self.world_offset_m * 2.5) * 0.5
        if not self.is_on_ground:
            leg_angle = 0.3
        
        leg_len = 20
        leg1_end = (rect.centerx + math.cos(leg_angle) * leg_len, rect.bottom + math.sin(leg_angle) * leg_len - 10)
        leg2_end = (rect.centerx + math.cos(-leg_angle) * leg_len, rect.bottom + math.sin(-leg_angle) * leg_len - 10)
        
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (rect.centerx, rect.bottom-5), leg1_end, 5)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (rect.centerx, rect.bottom-5), leg2_end, 5)

        body_rect = pygame.Rect(0, 0, rect.width, rect.height)
        body_rect.midbottom = (rect.centerx, rect.bottom + body_bob)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=8)
        
        eye_pos = (body_rect.centerx + 5, body_rect.centery - 5)
        pygame.draw.circle(self.screen, (255,255,255), eye_pos, 4)
        pygame.draw.circle(self.screen, (0,0,0), eye_pos, 2)

    def _render_ui(self):
        dist_text = f"Distance: {int(self.world_offset_m)} / {self.TRACK_LENGTH_METERS} m"
        score_text = f"Coins: {self.score}"
        
        self._draw_text(dist_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(score_text, (self.SCREEN_WIDTH - 150, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _draw_text(self, text, pos, font, color, shadow_color):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _world_to_screen_x(self, world_x_m, speed_ratio=1.0):
        # Parallax formula to keep player at a fixed screen position
        return (world_x_m - self.world_offset_m) * self.PIXELS_PER_METER * speed_ratio + self.PLAYER_X_POS

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 18, self.player_pos.y - 38, 36, 38)
    
    def _get_obstacle_rect(self, obs):
        x = self._world_to_screen_x(obs['x_m'])
        w = obs['w_m'] * self.PIXELS_PER_METER
        return pygame.Rect(int(x), int(self.GROUND_Y - obs['h_px']), int(w), int(obs['h_px']))
        
    def _get_coin_rect(self, coin_pos):
        x = self._world_to_screen_x(coin_pos.x)
        return pygame.Rect(int(x - 8), int(coin_pos.y - 8), 16, 16)

    def _create_particles(self, pos, color, count, **kwargs):
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random, **kwargs))

    def _update_particles(self):
        for p in list(self.particles):
            p.update()
            if p.is_dead(): self.particles.remove(p)

    def _generate_world_chunk(self, initial=False):
        screen_end_x_m = self.world_offset_m + (self.SCREEN_WIDTH / self.PIXELS_PER_METER)
        
        while self.last_feature_x_m < screen_end_x_m + 50:
            if self.last_feature_x_m > self.TRACK_LENGTH_METERS: break
            
            min_gap_m = self.running_speed_mps * 0.7
            max_gap_m = self.running_speed_mps * 1.5
            gap_m = self.np_random.uniform(min_gap_m, max_gap_m)
            feature_start_x_m = self.last_feature_x_m + gap_m

            feature_type = self.np_random.choice(['obstacle', 'pit', 'coins'], p=[0.4, 0.2, 0.4])

            if feature_type == 'obstacle':
                width_m = self.np_random.uniform(1, 2)
                height_px = self.np_random.integers(30, 80)
                self.obstacles.append({'x_m': feature_start_x_m, 'w_m': width_m, 'h_px': height_px})
                self.last_feature_x_m = feature_start_x_m + width_m
            elif feature_type == 'pit' and not initial:
                width_m = self.np_random.uniform(2, 4)
                self.pits.append({'x_m': feature_start_x_m, 'w_m': width_m})
                self.last_feature_x_m = feature_start_x_m + width_m
            elif feature_type == 'coins' or (feature_type == 'pit' and initial):
                num_coins = self.np_random.integers(3, 8)
                pattern = self.np_random.choice(['line', 'arc'])
                coin_y = self.GROUND_Y - self.np_random.integers(50, 150)
                for i in range(num_coins):
                    coin_x_m = feature_start_x_m + i * 0.7
                    y_offset = 0
                    if pattern == 'arc' and num_coins > 1:
                        y_offset = math.sin(i / (num_coins - 1) * math.pi) * 50
                    self.coins.append(pygame.Vector2(coin_x_m, coin_y - y_offset))
                self.last_feature_x_m = feature_start_x_m + num_coins * 0.7
        
        for layer in self.bg_layers.values():
            while not layer['items'] or self._world_to_screen_x(layer['items'][-1].x, layer['speed_ratio']) < self.SCREEN_WIDTH:
                last_x = layer['items'][-1].x if layer['items'] else self.world_offset_m * layer['speed_ratio']
                item_x_world = last_x + self.np_random.uniform(200, 400) / layer['speed_ratio']
                item_y = self.np_random.uniform(50, self.GROUND_Y - 150)
                item_w = self.np_random.uniform(100, 200)
                item_h = self.np_random.uniform(50, 150)
                layer['items'].append(pygame.Rect(item_x_world, item_y, item_w, item_h))

    def _cull_world_elements(self):
        cull_x_m = self.world_offset_m - 20
        while self.obstacles and self.obstacles[0]['x_m'] + self.obstacles[0]['w_m'] < cull_x_m: self.obstacles.popleft()
        while self.coins and self.coins[0].x < cull_x_m: self.coins.popleft()
        while self.pits and self.pits[0]['x_m'] + self.pits[0]['w_m'] < cull_x_m: self.pits.popleft()
        
        for layer in self.bg_layers.values():
            cull_x_screen = -200
            while layer['items'] and self._world_to_screen_x(layer['items'][0].x + layer['items'][0].width, layer['speed_ratio']) < cull_x_screen:
                layer['items'].popleft()

    def validate_implementation(self):
        self.reset(seed=0)
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset(seed=0)
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import os
    # For some setups, the next line is needed to run without a display
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Arcade Runner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = np.array([0, 0, 0]) # [movement, space, shift]
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Distance: {info['distance_m']:.2f}, Total Reward: {total_reward:.2f}")
            total_reward = 0.0
            obs, info = env.reset(seed=random.randint(0, 10000))
            # Add a small delay before restarting
            pygame.time.wait(1000)
        
        clock.tick(GameEnv.FPS)
        
    env.close()