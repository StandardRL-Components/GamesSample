
# Generated: 2025-08-28T00:32:37.574420
# Source Brief: brief_03818.md
# Brief Index: 3818

        
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
        "Controls: Arrow keys to move the cursor. Press space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding the bombs. Reach 100 points to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CURSOR_SPEED = 15
    MAX_STEPS = 1000
    WIN_SCORE = 100
    MAX_BOMB_HITS = 3
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)
    COLOR_BG_BOTTOM = (25, 25, 112)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    
    FRUIT_TYPES = {
        'apple': {'color': (220, 20, 60), 'base_points': 1, 'size_range': (15, 20)},
        'lime': {'color': (50, 205, 50), 'base_points': 2, 'size_range': (10, 15)},
        'orange': {'color': (255, 165, 0), 'base_points': 3, 'size_range': (20, 25)},
    }
    BOMB_COLOR = (30, 30, 30)
    BOMB_SKULL_COLOR = (200, 200, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        # Initialize state variables
        self.cursor_pos = None
        self.objects = None
        self.slice_trails = None
        self.particles = None
        self.score = None
        self.bombs_hit = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.fall_speed_multiplier = None
        self.spawn_timer = None
        self.bg_surface = self._create_gradient_background()

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.objects = []
        self.slice_trails = []
        self.particles = []
        self.score = 0
        self.bombs_hit = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.fall_speed_multiplier = 1.0
        self.spawn_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        self.steps += 1
        
        # --- Handle Input ---
        self._handle_movement(movement)
        slice_action = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if not slice_action:
            reward -= 0.1 # Small penalty for not acting

        # --- Update Game Logic ---
        self._update_difficulty()
        self._update_spawner()
        self._update_objects()
        self._update_effects()
        
        # --- Handle Slicing ---
        if slice_action:
            # SFX: slice_whoosh.wav
            self.slice_trails.append({'points': [self.cursor_pos.copy()], 'life': 10})
            reward += self._perform_slice()

        # --- Handle Off-screen Objects ---
        reward += self._cull_offscreen_objects()

        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        if self.bombs_hit >= self.MAX_BOMB_HITS:
            reward -= 50
            terminated = True
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 50 == 0:
            self.fall_speed_multiplier += 0.01

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_object()
            self.spawn_timer = self.np_random.integers(15, 30)

    def _spawn_object(self):
        obj_type = 'bomb' if self.np_random.random() < 0.2 else self.np_random.choice(list(self.FRUIT_TYPES.keys()))
        pos = np.array([self.np_random.uniform(50, self.WIDTH - 50), -30.0])
        vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(2, 4) * self.fall_speed_multiplier])
        
        if obj_type == 'bomb':
            size = self.np_random.uniform(18, 22)
            self.objects.append({'type': 'bomb', 'pos': pos, 'vel': vel, 'size': size, 'angle': 0})
        else:
            fruit_info = self.FRUIT_TYPES[obj_type]
            size = self.np_random.uniform(*fruit_info['size_range'])
            self.objects.append({
                'type': obj_type, 'pos': pos, 'vel': vel, 'size': size, 
                'color': fruit_info['color'], 'points': fruit_info['base_points'],
                'angle': 0, 'rot_speed': self.np_random.uniform(-0.1, 0.1)
            })

    def _update_objects(self):
        for obj in self.objects:
            obj['pos'] += obj['vel']
            if obj['type'] != 'bomb':
                obj['angle'] += obj['rot_speed']

    def _update_effects(self):
        # Update slice trails
        for trail in self.slice_trails:
            trail['life'] -= 1
        self.slice_trails = [t for t in self.slice_trails if t['life'] > 0]
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _perform_slice(self):
        slice_reward = 0
        sliced_something = False
        
        # Define slice area
        slice_rect = pygame.Rect(self.cursor_pos[0] - 50, self.cursor_pos[1] - 50, 100, 100)
        
        for obj in self.objects[:]:
            if slice_rect.collidepoint(obj['pos']):
                sliced_something = True
                if obj['type'] == 'bomb':
                    # SFX: explosion.wav
                    self.bombs_hit += 1
                    slice_reward -= 10
                    self.score = max(0, self.score - 10)
                    self._create_explosion(obj['pos'])
                else:
                    # SFX: fruit_squish.wav
                    points = obj['points'] + (1 if obj['size'] > 20 else 0)
                    self.score += points
                    slice_reward += points
                    self._create_fruit_particles(obj)
                self.objects.remove(obj)
        
        return slice_reward

    def _cull_offscreen_objects(self):
        reward = 0
        for obj in self.objects[:]:
            if obj['pos'][1] > self.HEIGHT + obj['size']:
                if obj['type'] != 'bomb':
                    reward -= 1 # Penalty for missing fruit
                self.objects.remove(obj)
        return reward
        
    def _create_fruit_particles(self, fruit):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': fruit['pos'].copy(), 'vel': vel, 'life': self.np_random.integers(20, 40),
                'color': fruit['color'], 'size': self.np_random.uniform(2, 5)
            })

    def _create_explosion(self, pos):
        # White flash
        self.particles.append({'pos': pos.copy(), 'vel': np.zeros(2), 'life': 8, 'color': (255, 255, 255), 'size': 100, 'type': 'flash'})
        # Gray smoke
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed - 1])
            color_val = self.np_random.integers(50, 100)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(30, 60),
                'color': (color_val, color_val, color_val), 'size': self.np_random.uniform(3, 8)
            })

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "bombs_hit": self.bombs_hit}

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color = [
                self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.HEIGHT)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_game(self):
        # Render particles
        for p in self.particles:
            if p.get('type') == 'flash':
                alpha = int(255 * (p['life'] / 8))
                radius = int(p['size'] * (1 - p['life'] / 8))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], alpha))
            else:
                alpha = int(255 * (p['life'] / 40))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Render objects
        for obj in self.objects:
            if obj['type'] == 'bomb':
                self._draw_bomb(self.screen, obj['pos'], obj['size'])
            else:
                self._draw_fruit(self.screen, obj)
        
        # Render slice trails
        for trail in self.slice_trails:
            if len(trail['points']) > 1:
                alpha = int(255 * (trail['life'] / 10))
                pygame.draw.lines(self.screen, (*self.COLOR_CURSOR, alpha), False, trail['points'], width=max(1, int(trail['life']/2)))

        # Render cursor
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 8, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (int(self.cursor_pos[0]), int(self.cursor_pos[1]) - 12), (int(self.cursor_pos[0]), int(self.cursor_pos[1]) + 12), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (int(self.cursor_pos[0]) - 12, int(self.cursor_pos[1])), (int(self.cursor_pos[0]) + 12, int(self.cursor_pos[1])), 1)

    def _draw_fruit(self, surface, fruit):
        pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
        size = int(fruit['size'])
        color = fruit['color']
        
        # Main body
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], size, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], size, (0,0,0,50))
        
        # Simple shine
        shine_pos = (pos[0] - size // 3, pos[1] - size // 3)
        shine_size = size // 3
        pygame.gfxdraw.filled_circle(surface, shine_pos[0], shine_pos[1], shine_size, (255,255,255,100))

    def _draw_bomb(self, surface, pos, size):
        pos_i = (int(pos[0]), int(pos[1]))
        size_i = int(size)
        pygame.gfxdraw.filled_circle(surface, pos_i[0], pos_i[1], size_i, self.BOMB_COLOR)
        pygame.gfxdraw.aacircle(surface, pos_i[0], pos_i[1], size_i, (0,0,0))
        self._draw_skull(surface, pos_i, size_i * 0.7)

    def _draw_skull(self, surface, pos, size):
        # Skull shape
        top_y = pos[1] - size * 0.4
        bottom_y = pos[1] + size * 0.4
        width = size * 0.7
        pygame.draw.rect(surface, self.BOMB_SKULL_COLOR, (pos[0] - width/2, top_y, width, bottom_y - top_y))
        pygame.gfxdraw.filled_circle(surface, pos[0], int(top_y), int(width/2), self.BOMB_SKULL_COLOR)
        
        # Eyes
        eye_y = pos[1] - size * 0.1
        eye_offset_x = size * 0.2
        eye_size = int(size * 0.15)
        pygame.draw.circle(surface, self.BOMB_COLOR, (pos[0] - eye_offset_x, eye_y), eye_size)
        pygame.draw.circle(surface, self.BOMB_COLOR, (pos[0] + eye_offset_x, eye_y), eye_size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Bomb hits
        for i in range(self.MAX_BOMB_HITS):
            bomb_pos = (self.WIDTH - 30 - i * 40, 30)
            if i < self.bombs_hit:
                self._draw_bomb(self.screen, bomb_pos, 15)
                pygame.draw.line(self.screen, (255, 0, 0), (bomb_pos[0]-15, bomb_pos[1]-15), (bomb_pos[0]+15, bomb_pos[1]+15), 3)
                pygame.draw.line(self.screen, (255, 0, 0), (bomb_pos[0]-15, bomb_pos[1]+15), (bomb_pos[0]+15, bomb_pos[1]-15), 3)
            else:
                pygame.gfxdraw.aacircle(self.screen, bomb_pos[0], bomb_pos[1], 15, (0,0,0,100))
                pygame.gfxdraw.filled_circle(self.screen, bomb_pos[0], bomb_pos[1], 15, (0,0,0,50))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # None
        space_action = 0 # Released
        shift_action = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Run at 30 FPS
        
    env.close()