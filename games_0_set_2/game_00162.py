
# Generated: 2025-08-27T12:47:31.394298
# Source Brief: brief_00162.md
# Brief Index: 162

        
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
    """
    A Gymnasium environment for a fast-paced arcade fruit-slicing game.
    The player controls a cursor and must slice falling fruit to score points
    against a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space for a vertical slice and shift for a horizontal slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit in this fast-paced arcade game to reach a target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SCORE = 50
    TIME_LIMIT_SECONDS = 60
    
    # Step rate is assumed to be 100Hz for timing calculations
    MAX_STEPS = TIME_LIMIT_SECONDS * 100

    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_OUTLINE = (0, 0, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SLICE_LINE = (255, 255, 255)

    FRUIT_TYPES = {
        'apple': {'color': (220, 20, 60), 'radius': 18},
        'orange': {'color': (255, 140, 0), 'radius': 20},
        'lemon': {'color': (255, 235, 59), 'radius': 16},
        'lime': {'color': (50, 205, 50), 'radius': 15},
    }

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
        self.font_small = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 64, bold=True)

        # Game state variables are initialized in reset()
        self.cursor_pos = None
        self.fruits = None
        self.particles = None
        self.sliced_fruits = None
        self.slice_effects = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.fruit_spawn_rate = None
        self.fruit_fall_speed = None
        self.spawn_rate_milestone = None
        self.fall_speed_milestone = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.rng = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.fruits = []
        self.particles = []
        self.sliced_fruits = []
        self.slice_effects = []
        
        self.score = 0
        self.steps = 0
        self.game_over = False

        # Difficulty settings
        self.fruit_spawn_rate = 0.02
        self.fruit_fall_speed = 1.5
        self.spawn_rate_milestone = 0
        self.fall_speed_milestone = 0

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement)
        reward += self._process_slices(space_held, shift_held)

        self._update_fruits()
        self._update_effects()
        self._spawn_fruit()
        self._update_difficulty()

        self.steps += 1
        
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        cursor_speed = 6
        if movement == 1: self.cursor_pos.y -= cursor_speed
        if movement == 2: self.cursor_pos.y += cursor_speed
        if movement == 3: self.cursor_pos.x -= cursor_speed
        if movement == 4: self.cursor_pos.x += cursor_speed

        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

    def _process_slices(self, space_held, shift_held):
        slice_reward = 0
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed: # Vertical Slice
            # sfx: whoosh_vertical.wav
            self.slice_effects.append({'type': 'vertical', 'pos': self.cursor_pos.x, 'life': 10})
            sliced_indices = [i for i, fruit in enumerate(self.fruits) if abs(fruit['pos'].x - self.cursor_pos.x) < fruit['radius']]
            if sliced_indices:
                slice_reward += self._slice_fruits(sliced_indices)

        if shift_pressed: # Horizontal Slice
            # sfx: whoosh_horizontal.wav
            self.slice_effects.append({'type': 'horizontal', 'pos': self.cursor_pos.y, 'life': 10})
            sliced_indices = [i for i, fruit in enumerate(self.fruits) if abs(fruit['pos'].y - self.cursor_pos.y) < fruit['radius']]
            if sliced_indices:
                slice_reward += self._slice_fruits(sliced_indices)
        
        return slice_reward

    def _slice_fruits(self, indices):
        # sfx: squish.wav
        num_sliced = 0
        for i in sorted(indices, reverse=True):
            fruit = self.fruits.pop(i)
            self.score += 1
            num_sliced += 1
            self._create_slice_visuals(fruit)
        return num_sliced

    def _create_slice_visuals(self, fruit):
        for _ in range(2):
            self.sliced_fruits.append({
                'pos': fruit['pos'].copy(),
                'vel': pygame.Vector2(self.rng.uniform(-3, 3), self.rng.uniform(-4, -1)),
                'angle': self.rng.uniform(0, 360),
                'rot_speed': self.rng.uniform(-10, 10),
                'radius': fruit['radius'],
                'color': fruit['color'],
                'life': 60
            })
        for _ in range(20):
            self.particles.append({
                'pos': fruit['pos'].copy(),
                'vel': pygame.Vector2(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4)),
                'color': fruit['color'],
                'radius': self.rng.uniform(1, 4),
                'life': 30
            })

    def _update_fruits(self):
        for fruit in self.fruits:
            fruit['pos'] += fruit['vel']
        self.fruits = [f for f in self.fruits if f['pos'].y < self.SCREEN_HEIGHT + f['radius']]

    def _update_effects(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        for sf in self.sliced_fruits:
            sf['pos'] += sf['vel']
            sf['vel'].y += 0.2
            sf['angle'] += sf['rot_speed']
            sf['life'] -= 1
        self.sliced_fruits = [sf for sf in self.sliced_fruits if sf['life'] > 0]
        
        for sl in self.slice_effects:
            sl['life'] -= 1
        self.slice_effects = [sl for sl in self.slice_effects if sl['life'] > 0]

    def _spawn_fruit(self):
        if self.rng.random() < self.fruit_spawn_rate:
            fruit_key = self.rng.choice(list(self.FRUIT_TYPES.keys()))
            spec = self.FRUIT_TYPES[fruit_key]
            radius = spec['radius']
            new_fruit = {
                'pos': pygame.Vector2(self.rng.uniform(radius, self.SCREEN_WIDTH - radius), -radius),
                'vel': pygame.Vector2(0, self.fruit_fall_speed + self.rng.uniform(-0.5, 0.5)),
                'radius': radius,
                'color': spec['color'],
                'type': fruit_key
            }
            self.fruits.append(new_fruit)

    def _update_difficulty(self):
        current_spawn_milestone = self.score // 25
        if current_spawn_milestone > self.spawn_rate_milestone:
            self.fruit_spawn_rate += 0.005
            self.spawn_rate_milestone = current_spawn_milestone

        current_speed_milestone = self.score // 50
        if current_speed_milestone > self.fall_speed_milestone:
            self.fruit_fall_speed += 0.2
            self.fall_speed_milestone = current_speed_milestone

    def _get_observation(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for sf in self.sliced_fruits:
            self._draw_aa_circle(self.screen, sf['color'], (int(sf['pos'].x), int(sf['pos'].y)), sf['radius'])

        for fruit in self.fruits:
            self._draw_aa_circle(self.screen, fruit['color'], (int(fruit['pos'].x), int(fruit['pos'].y)), fruit['radius'])
            highlight_pos = (int(fruit['pos'].x + fruit['radius']*0.3), int(fruit['pos'].y - fruit['radius']*0.3))
            pygame.gfxdraw.aacircle(self.screen, highlight_pos[0], highlight_pos[1], int(fruit['radius']*0.2), (255, 255, 255, 150))
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], int(fruit['radius']*0.2), (255, 255, 255, 150))

        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 30)))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        for sl in self.slice_effects:
            alpha = max(0, int(255 * (sl['life'] / 10)))
            color = (*self.COLOR_SLICE_LINE, alpha)
            if sl['type'] == 'vertical':
                pygame.draw.line(self.screen, color, (int(sl['pos']), 0), (int(sl['pos']), self.SCREEN_HEIGHT), 3)
            else:
                pygame.draw.line(self.screen, color, (0, int(sl['pos'])), (self.SCREEN_WIDTH, int(sl['pos'])), 3)

        x, y = int(self.cursor_pos.x), int(self.cursor_pos.y)
        size = 12
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (x - size, y), (x + size, y), 5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (x, y - size), (x, y + size), 5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - size, y), (x + size, y), 3)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - size), (x, y + size), 3)
        
    def _draw_aa_circle(self, surface, color, center, radius):
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}/{self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left_sec = (self.MAX_STEPS - self.steps) / 100
        time_text = self.font_small.render(f"TIME: {time_left_sec:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.TARGET_SCORE else "TIME'S UP!"
            # sfx: win_jingle.wav or lose_sound.wav
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_steps": (self.MAX_STEPS - self.steps)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        exit()

    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Fruit Slicer")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(100)

    env.close()