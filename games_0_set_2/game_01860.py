
# Generated: 2025-08-27T18:31:30.457052
# Source Brief: brief_01860.md
# Brief Index: 1860

        
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
        "Controls: Arrow keys to move the slicer. Press space to slice. Avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding bombs in this fast-paced, grid-based arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_SLICER = (220, 255, 255)
        self.COLOR_SLICER_GLOW = (150, 200, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_FUSE = (255, 80, 0)
        self.FRUIT_TYPES = {
            "apple": {"color": (80, 200, 80), "value": 1, "radius": 0.35},
            "banana": {"color": (255, 220, 50), "value": 2, "radius": 0.3},
            "grape": {"color": (150, 80, 200), "value": 3, "radius": 0.25},
        }

        # Game parameters
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.INITIAL_LIVES = 3
        self.INITIAL_FALL_SPEED = 1.0  # cells per second
        self.SPEED_INCREASE_INTERVAL = 50
        self.SPEED_INCREASE_AMOUNT = 0.05
        self.FPS = 30.0
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.slicer_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE - 1])
        self.objects = []
        self.particles = []
        self.effects = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.object_spawn_timer = 0
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1
            
            # --- Handle Actions ---
            self._handle_movement(movement)
            slice_reward = self._handle_slicing(space_held)
            reward += slice_reward

            # --- Update Game State ---
            self._update_objects()
            self._update_particles()
            self._update_effects()
            self._spawn_objects()

            # --- Difficulty Scaling ---
            if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
                self.fall_speed += self.SPEED_INCREASE_AMOUNT

            # --- Check Termination ---
            if self.lives <= 0 or self.steps >= self.MAX_STEPS:
                terminated = True
            elif self.score >= self.WIN_SCORE:
                terminated = True
                reward += 100 # Goal-oriented reward for winning
            
            if terminated:
                self.game_over = True
                self._add_game_over_effect()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.slicer_pos[1] -= 1
        elif movement == 2: # Down
            self.slicer_pos[1] += 1
        elif movement == 3: # Left
            self.slicer_pos[0] -= 1
        elif movement == 4: # Right
            self.slicer_pos[0] += 1
        
        self.slicer_pos = np.clip(self.slicer_pos, 0, self.GRID_SIZE - 1)

    def _handle_slicing(self, space_held):
        slice_reward = 0
        slice_performed = False

        if space_held and not self.prev_space_held:
            # Sfx: Slice_whoosh.wav
            self._add_slice_effect(self.slicer_pos)
            sliced_object = False
            
            # Iterate backwards to allow safe removal
            for obj in reversed(self.objects):
                obj_grid_pos = [int(obj['pos'][0]), int(obj['pos'][1])]
                if np.array_equal(obj_grid_pos, self.slicer_pos):
                    if obj['type'] == 'bomb':
                        # Sfx: Bomb_explosion.wav
                        self.lives -= 1
                        slice_reward = -10
                        self._add_bomb_explosion_effect(obj['pos'])
                    else: # Fruit
                        # Sfx: Fruit_slice.wav
                        fruit_info = self.FRUIT_TYPES[obj['type']]
                        self.score += fruit_info['value']
                        slice_reward = 1
                        self._add_fruit_particles(obj['pos'], fruit_info['color'])
                    
                    self.objects.remove(obj)
                    sliced_object = True
                    break # Only slice one object per action
            
            if not sliced_object:
                slice_reward = -0.2 # Penalty for slicing empty space

        self.prev_space_held = space_held
        return slice_reward

    def _update_objects(self):
        fall_delta = self.fall_speed / self.FPS
        for obj in self.objects:
            obj['pos'][1] += fall_delta
        
        # Remove objects that have fallen off-screen
        self.objects = [obj for obj in self.objects if obj['pos'][1] < self.GRID_SIZE + 1]

    def _spawn_objects(self):
        self.object_spawn_timer -= 1
        if self.object_spawn_timer <= 0:
            spawn_x = self.np_random.integers(0, self.GRID_SIZE)
            
            # 20% chance to spawn a bomb
            if self.np_random.random() < 0.2:
                obj_type = 'bomb'
            else:
                obj_type = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
            
            self.objects.append({'type': obj_type, 'pos': np.array([spawn_x, -0.5], dtype=float)})
            
            # Reset timer with some randomness
            self.object_spawn_timer = self.np_random.integers(int(self.FPS * 0.5), int(self.FPS * 1.5))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_objects()
        self._render_particles()
        self._render_effects()
        self._render_slicer()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, grid_pos, center=True):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = grid_pos[1] * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_grid(self):
        for x in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, 0)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X, y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_slicer(self):
        pos_px = self._grid_to_pixel(self.slicer_pos)
        size = self.CELL_SIZE // 2

        # Glow effect
        glow_radius = int(size * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_SLICER_GLOW, 30), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos_px[0] - glow_radius, pos_px[1] - glow_radius))

        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_SLICER, (pos_px[0] - size, pos_px[1]), (pos_px[0] + size, pos_px[1]), 3)
        pygame.draw.line(self.screen, self.COLOR_SLICER, (pos_px[0], pos_px[1] - size), (pos_px[0], pos_px[1] + size), 3)
        pygame.draw.circle(self.screen, self.COLOR_SLICER, pos_px, size, 2)

    def _render_objects(self):
        for obj in self.objects:
            pos_px = self._grid_to_pixel(obj['pos'])
            
            if obj['type'] == 'bomb':
                radius = int(self.CELL_SIZE * 0.4)
                pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_BOMB)
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_BOMB)
                # Fuse
                pygame.draw.line(self.screen, self.COLOR_BOMB_FUSE, (pos_px[0] + radius - 2, pos_px[1] - radius + 2), (pos_px[0] + radius + 2, pos_px[1] - radius - 6), 3)
                # Shine
                pygame.gfxdraw.arc(self.screen, pos_px[0] - radius//2, pos_px[1] - radius//2, radius//3, 120, 160, (200,200,200))
            else: # Fruit
                info = self.FRUIT_TYPES[obj['type']]
                radius = int(self.CELL_SIZE * info['radius'])
                pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, info['color'])
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, info['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(score_text, score_rect)

        # Lives
        for i in range(self.INITIAL_LIVES):
            pos = (self.WIDTH - 40 - i * 35, 30)
            color = (255, 80, 80) if i < self.lives else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, color)
        
        # Game Over / Win message
        if self.game_over:
            message = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            color = (255, 80, 80) if self.lives <= 0 else (80, 255, 80)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _add_fruit_particles(self, pos, color):
        pos_px = self._grid_to_pixel(pos)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': np.array(pos_px, dtype=float), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], max(0, min(255, alpha)))
            size = max(1, int(p['lifespan'] / 5))
            rect = pygame.Rect(int(p['pos'][0] - size//2), int(p['pos'][1] - size//2), size, size)
            pygame.draw.rect(self.screen, color, rect)

    def _add_slice_effect(self, grid_pos):
        pos_px = self._grid_to_pixel(grid_pos)
        self.effects.append({'type': 'slice', 'pos': pos_px, 'radius': 0, 'max_radius': self.CELL_SIZE * 0.8, 'lifespan': 8})

    def _add_bomb_explosion_effect(self, grid_pos):
        pos_px = self._grid_to_pixel(grid_pos)
        self.effects.append({'type': 'explosion', 'pos': pos_px, 'radius': 0, 'max_radius': self.CELL_SIZE * 2, 'lifespan': 15})

    def _add_game_over_effect(self):
        pass # The UI rendering handles the text overlay

    def _update_effects(self):
        for e in self.effects:
            e['lifespan'] -= 1
            e['radius'] += e['max_radius'] / (e['lifespan'] + 1)
        self.effects = [e for e in self.effects if e['lifespan'] > 0]

    def _render_effects(self):
        for e in self.effects:
            alpha = 200 * (e['lifespan'] / 15.0)
            if e['type'] == 'slice':
                color = (*self.COLOR_SLICER, int(alpha))
                pygame.gfxdraw.aacircle(self.screen, e['pos'][0], e['pos'][1], int(e['radius']), color)
            elif e['type'] == 'explosion':
                color1 = (*self.COLOR_BOMB_FUSE, int(alpha))
                color2 = (*(255, 200, 0), int(alpha * 0.7))
                pygame.gfxdraw.filled_circle(self.screen, e['pos'][0], e['pos'][1], int(e['radius']), color1)
                pygame.gfxdraw.filled_circle(self.screen, e['pos'][0], e['pos'][1], int(e['radius'] * 0.6), color2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # To run and visualize the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for visualization ---
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            # Optional: auto-reset after a delay
            # pygame.time.wait(2000)
            # obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control frame rate ---
        clock.tick(env.FPS)
        
    env.close()