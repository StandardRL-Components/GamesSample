
# Generated: 2025-08-28T04:57:41.462493
# Source Brief: brief_05422.md
# Brief Index: 5422

        
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

    user_guide = (
        "Controls: Use ← and → to select a column. Press Space to slice."
    )

    game_description = (
        "Slice falling fruit while dodging bombs in this fast-paced arcade game. "
        "Slice 30 fruits to win, but hitting 3 bombs means game over!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game constants
        self.NUM_COLUMNS = 8
        self.COLUMN_WIDTH = self.WIDTH // self.NUM_COLUMNS
        self.MAX_STEPS = 1000
        self.WIN_FRUITS = 30
        self.MAX_LIVES = 3
        
        # Colors
        self.COLOR_BG_TOP = (16, 16, 32)
        self.COLOR_BG_BOTTOM = (32, 16, 48)
        self.COLOR_FRUITS = [(255, 65, 54), (255, 133, 27), (255, 220, 0)] # Red, Orange, Yellow
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_FUSE = (255, 70, 70)
        self.COLOR_CURSOR = (255, 255, 255, 90)
        self.COLOR_SLICE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BOMB_ICON = (50, 50, 50)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.lives = 0
        self.fruits_sliced = 0
        self.slicer_column = 0
        self.objects = []
        self.particles = []
        self.slice_effects = []
        self.fall_speed = 0.0
        self.last_space_held = False
        self.last_move_action = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.fruits_sliced = 0
        
        self.slicer_column = self.NUM_COLUMNS // 2
        self.objects = []
        self.particles = []
        self.slice_effects = []
        
        self.fall_speed = 2.0
        self.last_space_held = False
        self.last_move_action = 0 # To register single presses
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Movement: register on new press, not hold
        if movement in [3, 4] and movement != self.last_move_action:
            if movement == 3: # Left
                self.slicer_column = max(0, self.slicer_column - 1)
            elif movement == 4: # Right
                self.slicer_column = min(self.NUM_COLUMNS - 1, self.slicer_column + 1)
        self.last_move_action = movement

        # Slicing: register on press (rising edge)
        slice_triggered = space_held and not self.last_space_held
        self.last_space_held = space_held

        # 2. Update Game State
        self._update_difficulty()
        self._update_objects()
        self._update_effects()

        # 3. Process Slicing Action
        if slice_triggered:
            # sfx: slice_whoosh.wav
            self.slice_effects.append({'column': self.slicer_column, 'life': 10})
            reward += self._perform_slice()

        # 4. Check for Termination
        is_win = self.fruits_sliced >= self.WIN_FRUITS
        is_loss = self.lives <= 0
        is_timeout = self.steps >= self.MAX_STEPS
        
        terminated = is_win or is_loss or is_timeout
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if is_win:
                reward += 100
            elif is_loss:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_difficulty(self):
        # Increase fall speed every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.fall_speed += 0.2

    def _update_objects(self):
        # Spawn new objects
        if self.np_random.random() < 0.08: # Chance to spawn an object
            self._spawn_object()

        # Move and remove off-screen objects
        for obj in self.objects[:]:
            obj['y'] += self.fall_speed
            if obj['y'] - obj['radius'] > self.HEIGHT:
                self.objects.remove(obj)

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update slice visual effects
        for effect in self.slice_effects[:]:
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.slice_effects.remove(effect)

    def _perform_slice(self):
        step_reward = 0
        slice_x_center = (self.slicer_column * self.COLUMN_WIDTH) + (self.COLUMN_WIDTH / 2)
        
        sliced_something = False
        for obj in self.objects[:]:
            if abs(obj['x'] - slice_x_center) < self.COLUMN_WIDTH / 2:
                if obj['type'] == 'fruit':
                    # sfx: fruit_squish.wav
                    step_reward += 10
                    self.score += 10
                    self.fruits_sliced += 1
                    self._create_particles(obj['x'], obj['y'], obj['color'])
                    self.objects.remove(obj)
                    sliced_something = True
                elif obj['type'] == 'bomb':
                    # sfx: explosion.wav
                    step_reward -= 50
                    self.lives -= 1
                    self._create_particles(obj['x'], obj['y'], self.COLOR_FUSE, 30) # Bomb explosion
                    self.objects.remove(obj)
                    sliced_something = True
        return step_reward

    def _spawn_object(self):
        obj_type = 'bomb' if self.np_random.random() < 0.25 else 'fruit'
        column = self.np_random.integers(0, self.NUM_COLUMNS)
        x = (column * self.COLUMN_WIDTH) + (self.COLUMN_WIDTH / 2)
        y = -20
        radius = self.np_random.integers(15, 25)
        
        if obj_type == 'fruit':
            color = self.np_random.choice(self.COLOR_FRUITS)
            self.objects.append({'type': 'fruit', 'x': x, 'y': y, 'radius': radius, 'color': color})
        else:
            self.objects.append({'type': 'bomb', 'x': x, 'y': y, 'radius': radius})

    def _create_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            life = self.np_random.integers(20, 40)
            radius = self.np_random.random() * 3 + 2
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'color': color, 'radius': radius})

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Slicer cursor
        cursor_rect = pygame.Rect(self.slicer_column * self.COLUMN_WIDTH, 0, self.COLUMN_WIDTH, self.HEIGHT)
        s = pygame.Surface((self.COLUMN_WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (cursor_rect.x, cursor_rect.y))

        # Falling Objects
        for obj in self.objects:
            pos = (int(obj['x']), int(obj['y']))
            radius = int(obj['radius'])
            if obj['type'] == 'fruit':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, obj['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, obj['color'])
                # Add a little shine
                shine_pos = (pos[0] + radius // 3, pos[1] - radius // 3)
                pygame.gfxdraw.aacircle(self.screen, shine_pos[0], shine_pos[1], radius // 4, (255,255,255,100))
                pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], radius // 4, (255,255,255,100))
            else: # Bomb
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
                # Fuse
                pygame.draw.line(self.screen, (200,200,200), (pos[0], pos[1] - radius), (pos[0]+2, pos[1] - radius - 5), 2)
                pygame.gfxdraw.filled_circle(self.screen, pos[0]+2, pos[1] - radius - 6, 3, self.COLOR_FUSE)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Slice effects
        for effect in self.slice_effects:
            alpha = int(255 * (effect['life'] / 10.0))
            color = (*self.COLOR_SLICE, alpha)
            x = effect['column'] * self.COLUMN_WIDTH + self.COLUMN_WIDTH // 2
            start_pos = (x, 0)
            end_pos = (x, self.HEIGHT)
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(s, color, start_pos, end_pos, 4)
            self.screen.blit(s, (0,0))

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        for i in range(self.lives):
            pos = (self.WIDTH - 30 - (i * 35), 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_BOMB_ICON)
            pygame.gfxdraw.filled_circle(self.screen, pos[0]+1, pos[1] - 13, 3, self.COLOR_FUSE)

        # Game Over Message
        if self.game_over:
            is_win = self.fruits_sliced >= self.WIN_FRUITS
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fruits_sliced": self.fruits_sliced,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # 30 FPS

    env.close()