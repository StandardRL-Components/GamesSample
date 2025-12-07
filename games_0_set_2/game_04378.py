import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use ← and → to move the basket."

    # Must be a short, user-facing description of the game:
    game_description = "Catch falling gems to score points. Miss too many and you lose!"

    # Should frames auto-advance or wait for user input?
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
        
        # Visuals
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_BASKET = (255, 200, 0)
        self.COLOR_BASKET_BORDER = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.OBJECT_SPECS = {
            'green': {'color': (0, 255, 100), 'points': 1, 'reward': 0.1},
            'blue': {'color': (50, 150, 255), 'points': 2, 'reward': 0.2},
            'red': {'color': (255, 50, 50), 'points': 3, 'reward': 0.3, 'speed_mult': 1.5}
        }
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game Constants
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 10
        self.OBJECT_SIZE = 16
        self.INITIAL_FALL_SPEED = 2.0
        self.SPEED_INCREMENT = 0.05
        self.MAX_OBJECTS = 5
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 20
        self.LOSS_CONDITION = 5

        # State variables (will be initialized in reset)
        self.basket_rect = None
        self.objects = []
        self.particles = []
        self.score = 0
        self.catches = 0
        self.misses = 0
        self.steps = 0
        self.object_fall_speed = self.INITIAL_FALL_SPEED
        self.game_over = False
        self.win = False
        self.last_spawn_step = 0
        self.spawn_interval = 45

        # Initialize state
        # A seed is not passed to __init__ but to reset(), so we can't seed here.
        # We will call reset() once to set up the initial state.
        # self.reset() is called later to ensure np_random is available.

        # Run validation check - will be called after reset in a moment
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_rect = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) / 2,
            self.HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT,
        )
        self.objects = []
        self.particles = []
        
        self.score = 0
        self.catches = 0
        self.misses = 0
        self.steps = 0
        
        self.object_fall_speed = self.INITIAL_FALL_SPEED
        self.game_over = False
        self.win = False
        self.last_spawn_step = 0
        self.spawn_interval = 45

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # 1. Unpack and handle player action
        movement = action[0]
        moved = False
        if movement == 3:  # Left
            self.basket_rect.x -= self.BASKET_SPEED
            moved = True
        elif movement == 4:  # Right
            self.basket_rect.x += self.BASKET_SPEED
            moved = True
        
        if moved:
            reward -= 0.01  # Penalty for moving

        # Clamp basket to screen boundaries
        self.basket_rect.x = max(0, min(self.WIDTH - self.BASKET_WIDTH, self.basket_rect.x))

        # 2. Update game logic
        self._update_spawning()
        reward += self._update_objects()
        self._update_particles()

        # 3. Check for termination conditions
        terminated = False
        truncated = False
        if self.catches >= self.WIN_CONDITION:
            reward += 10
            terminated = True
            self.game_over = True
            self.win = True
        elif self.misses >= self.LOSS_CONDITION:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_spawning(self):
        if self.steps > self.last_spawn_step + self.spawn_interval:
            self._spawn_object()
            self.last_spawn_step = self.steps
            self.spawn_interval = self.np_random.integers(30, 60)

    def _update_objects(self):
        step_reward = 0
        objects_to_remove = []
        for obj in self.objects:
            speed_multiplier = obj.get('speed_mult', 1.0)
            obj['rect'].y += self.object_fall_speed * speed_multiplier
            
            if self.basket_rect.colliderect(obj['rect']):
                self.score += obj['points']
                self.catches += 1
                step_reward += obj['reward']
                self._create_particles(obj['rect'].center, obj['color'])
                objects_to_remove.append(obj)
                
                if self.catches > 0 and self.catches % 2 == 0:
                    self.object_fall_speed += self.SPEED_INCREMENT
            
            elif obj['rect'].top > self.HEIGHT:
                self.misses += 1
                objects_to_remove.append(obj)

        self.objects = [obj for obj in self.objects if obj not in objects_to_remove]
        return step_reward

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _spawn_object(self):
        # FIX: Ensure we do not exceed the maximum number of objects.
        if len(self.objects) >= self.MAX_OBJECTS:
            return

        rand_val = self.np_random.random()
        if rand_val < 0.15: obj_type = 'red'
        elif rand_val < 0.50: obj_type = 'blue'
        else: obj_type = 'green'
            
        spec = self.OBJECT_SPECS[obj_type]
        x_pos = self.np_random.integers(0, self.WIDTH - self.OBJECT_SIZE)
        
        obj = {
            'rect': pygame.Rect(x_pos, -self.OBJECT_SIZE, self.OBJECT_SIZE, self.OBJECT_SIZE),
            'type': obj_type,
            'color': spec['color'],
            'points': spec['points'],
            'reward': spec['reward'],
        }
        if 'speed_mult' in spec:
            obj['speed_mult'] = spec['speed_mult']
        self.objects.append(obj)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self._draw_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame and numpy have different coordinate systems.
        # Pygame: (x, y) with (0,0) at top-left.
        # Numpy for images: (height, width) with (0,0) at top-left.
        # surfarray.array3d creates (width, height, channels).
        # We need to transpose to (height, width, channels) for Gymnasium.
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _draw_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH - 1, y))

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']),
                    (*p['color'], alpha)
                )

        for obj in self.objects:
            self._draw_gem(self.screen, obj['rect'], obj['color'])

        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_BORDER, self.basket_rect, width=2, border_radius=5)

    def _draw_gem(self, surface, rect, color):
        center_x, center_y = rect.center
        size = rect.width / 2
        
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)

        points_bottom = [(center_x - size, center_y), (center_x, center_y + size), (center_x + size, center_y)]
        points_facet_left = [(center_x, center_y - size), (center_x - size, center_y), (center_x, center_y)]
        points_facet_right = [(center_x, center_y - size), (center_x + size, center_y), (center_x, center_y)]

        pygame.gfxdraw.filled_polygon(surface, points_bottom, dark_color)
        pygame.gfxdraw.aapolygon(surface, points_bottom, dark_color)
        pygame.gfxdraw.filled_polygon(surface, points_facet_left, color)
        pygame.gfxdraw.aapolygon(surface, points_facet_left, color)
        pygame.gfxdraw.filled_polygon(surface, points_facet_right, light_color)
        pygame.gfxdraw.aapolygon(surface, points_facet_right, light_color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = "Lives: " + "♥ " * (self.LOSS_CONDITION - self.misses)
        lives_surf = self.font_ui.render(lives_text, True, self.COLOR_TEXT)
        self.screen.blit(lives_surf, (self.WIDTH - lives_surf.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(msg, True, color)
            pos_x = (self.WIDTH - game_over_surf.get_width()) / 2
            pos_y = (self.HEIGHT - game_over_surf.get_height()) / 2
            self.screen.blit(game_over_surf, (pos_x, pos_y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "catches": self.catches,
            "misses": self.misses,
        }

    def close(self):
        pygame.quit()