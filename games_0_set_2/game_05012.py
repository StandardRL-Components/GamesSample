
# Generated: 2025-08-28T03:41:30.295783
# Source Brief: brief_05012.md
# Brief Index: 5012

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to select a grid cell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find 10 hidden geometric shapes in a procedurally generated field before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_W = 16
        self.GRID_H = 10
        self.CELL_SIZE = 40
        
        # Game constants
        self.NUM_OBJECTS = 10
        self.MAX_STEPS = 600 # 60 seconds at 10 steps/sec

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (50, 60, 70, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_INCORRECT = (255, 50, 50)
        self.OBJECT_PALETTE = [
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 200, 50),  # Yellow
            (255, 100, 200), # Pink
            (180, 100, 255), # Purple
            (255, 150, 50),  # Orange
            (50, 255, 255),  # Cyan
        ]

        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.objects = []
        self.objects_found = 0
        self.incorrect_guesses = set()
        self.prev_space_held = False
        self.background_surface = None
        self.particles = []
        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.objects_found = 0
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.incorrect_guesses = set()
        self.prev_space_held = False
        self.particles = []
        
        self._generate_background()
        self._generate_objects()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        reward = 0.0
        self.steps += 1

        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Wrap cursor around edges
        self.cursor_pos[0] %= self.GRID_W
        self.cursor_pos[1] %= self.GRID_H
        
        # 2. Handle selection on space press (rising edge)
        if space_held and not self.prev_space_held:
            cursor_tuple = tuple(self.cursor_pos)
            
            # Check if this cell has already been selected
            if cursor_tuple not in self.incorrect_guesses and not any(o['found'] and tuple(o['pos']) == cursor_tuple for o in self.objects):
                found_obj = False
                for obj in self.objects:
                    if not obj['found'] and tuple(obj['pos']) == cursor_tuple:
                        obj['found'] = True
                        self.objects_found += 1
                        reward = 1.0
                        # Sound: Correct guess
                        self._create_particles(self.cursor_pos)
                        found_obj = True
                        break
                
                if not found_obj:
                    self.incorrect_guesses.add(cursor_tuple)
                    reward = -0.1
                    # Sound: Incorrect guess

        self.prev_space_held = space_held
        self.score += reward

        # 3. Check for termination
        terminated = False
        if self.objects_found == self.NUM_OBJECTS:
            terminated = True
            terminal_reward = 100.0
            reward += terminal_reward
            self.score += terminal_reward
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            terminal_reward = -100.0
            reward = terminal_reward # Overwrite any small reward/penalty
            self.score += terminal_reward
            self.game_over = True
            # Sound: Time up failure

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Blit pre-rendered background
        self.screen.blit(self.background_surface, (0, 0))
        
        self._render_game_elements()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _generate_background(self):
        self.background_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.background_surface.fill(self.COLOR_BG)
        
        # Draw many faint, overlapping shapes for texture
        for _ in range(150):
            shape_type = self.np_random.choice(['circle', 'rect', 'line'])
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            color_val = self.np_random.integers(25, 45)
            color = (color_val, color_val + 5, color_val + 15, self.np_random.integers(10, 30))
            
            if shape_type == 'circle':
                radius = self.np_random.integers(20, 150)
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.background_surface.blit(temp_surf, (x - radius, y - radius), special_flags=pygame.BLEND_RGBA_ADD)
            elif shape_type == 'rect':
                w = self.np_random.integers(50, 200)
                h = self.np_random.integers(50, 200)
                temp_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                temp_surf.fill(color)
                self.background_surface.blit(temp_surf, (x - w//2, y - h//2), special_flags=pygame.BLEND_RGBA_ADD)
    
    def _generate_objects(self):
        self.objects = []
        possible_positions = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        chosen_indices = self.np_random.choice(len(possible_positions), self.NUM_OBJECTS, replace=False)
        chosen_positions = [possible_positions[i] for i in chosen_indices]

        shapes = ['circle', 'square', 'triangle', 'star']
        for pos in chosen_positions:
            self.objects.append({
                'pos': list(pos),
                'found': False,
                'color': self.OBJECT_PALETTE[self.np_random.integers(0, len(self.OBJECT_PALETTE))],
                'shape': self.np_random.choice(shapes),
                'size_mod': self.np_random.uniform(0.7, 1.0)
            })

    def _render_game_elements(self):
        # Draw grid lines
        grid_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for x in range(1, self.GRID_W):
            pygame.draw.line(grid_surf, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(1, self.GRID_H):
            pygame.draw.line(grid_surf, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_WIDTH, y * self.CELL_SIZE))
        self.screen.blit(grid_surf, (0,0))
        
        # Draw revealed objects, incorrect guesses, and cursor
        for obj in self.objects:
            if obj['found']:
                self._draw_object(obj)

        for pos in self.incorrect_guesses:
            self._draw_incorrect_marker(pos)

        self._update_and_draw_particles()
        self._draw_cursor()

    def _draw_object(self, obj):
        center_x = int(obj['pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(obj['pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        size = int(self.CELL_SIZE * 0.35 * obj['size_mod'])
        color = obj['color']
        outline_color = (255, 255, 255)

        if obj['shape'] == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, size, outline_color)
        elif obj['shape'] == 'square':
            rect = pygame.Rect(center_x - size, center_y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, outline_color, rect, 2)
        elif obj['shape'] == 'triangle':
            points = [
                (center_x, center_y - size),
                (center_x - size, center_y + size),
                (center_x + size, center_y + size)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)
        elif obj['shape'] == 'star':
            points = []
            for i in range(10):
                angle = math.pi / 5 * i - math.pi / 2
                r = size * 1.5 if i % 2 == 0 else size * 0.7
                points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_incorrect_marker(self, pos):
        center_x = int(pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        size = int(self.CELL_SIZE * 0.3)
        pygame.draw.line(self.screen, self.COLOR_INCORRECT, (center_x - size, center_y - size), (center_x + size, center_y + size), 4)
        pygame.draw.line(self.screen, self.COLOR_INCORRECT, (center_x - size, center_y + size), (center_x + size, center_y - size), 4)

    def _draw_cursor(self):
        rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE,
            self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Pulsing alpha for the fill
        alpha = 100 + 40 * math.sin(pygame.time.get_ticks() * 0.005)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_CURSOR, alpha))
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

    def _create_particles(self, grid_pos):
        # Sound: Particle burst
        center_x = int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            color = self.OBJECT_PALETTE[self.np_random.integers(len(self.OBJECT_PALETTE))]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(
                {'pos': [center_x, center_y], 'vel': vel, 'color': color, 'life': lifetime}
            )

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = max(0, int(p['life'] * 0.2))
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], radius)

    def _render_ui(self):
        # UI Background
        ui_bg_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 35)
        s = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        s.fill((10, 15, 25, 200))
        self.screen.blit(s, (0, 0))
        pygame.draw.line(self.screen, (80, 90, 110), (0, 35), (self.SCREEN_WIDTH, 35))

        # Objects Found Text
        found_text = f"Found: {self.objects_found} / {self.NUM_OBJECTS}"
        text_surf = self.font_small.render(found_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 8))

        # Timer Text
        time_left = max(0, (self.MAX_STEPS - self.steps) / 10)
        timer_text = f"Time: {time_left:.1f}"
        timer_color = self.COLOR_INCORRECT if time_left < 10 else self.COLOR_TEXT
        text_surf = self.font_small.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 8))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.objects_found == self.NUM_OBJECTS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_INCORRECT
                
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "objects_found": self.objects_found,
            "time_left": max(0, self.MAX_STEPS - self.steps)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to call reset first to initialize rng
        self.reset(seed=123)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert self.NUM_OBJECTS == len(self.objects)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset(seed=random.randint(0, 10000))
    
    # Override render_mode for human play
    env.metadata["render_modes"].append("human")
    pygame.display.set_caption("Hidden Objects")
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    terminated = False
    
    # Store previous action to handle held keys
    action = env.action_space.sample()
    action.fill(0)
    
    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        # Human controls
        movement = 0 # no-op
        space = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        action = np.array([movement, space, 0]) # shift is unused

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the display
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Found: {info['objects_found']}")

        # Since auto_advance is False, we need to control the pace
        env.clock.tick(15) # Limit human play speed

    print(f"Game Over! Final Score: {info['score']:.2f}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(3000)

    env.close()