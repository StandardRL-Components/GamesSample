
# Generated: 2025-08-27T21:50:37.546677
# Source Brief: brief_02926.md
# Brief Index: 2926

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string, corrected for the game brief
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to slice."
    )

    # User-facing description of the game, corrected for the game brief
    game_description = (
        "Slice the falling fruit and avoid the bombs to get the highest score!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CURSOR_SPEED = 10
    MAX_STEPS = 3600 # 60 seconds at 60 FPS
    FRUITS_TO_WIN = 30
    INITIAL_LIVES = 3
    
    # Colors
    COLOR_BG_TOP = (40, 40, 60)
    COLOR_BG_BOTTOM = (10, 10, 20)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_BOMB = (30, 30, 30)
    COLOR_BOMB_FUSE = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    FRUIT_PALETTE = [
        (255, 80, 80),   # Red (Apple)
        (255, 180, 50),  # Orange (Orange)
        (100, 220, 100), # Green (Pear)
        (255, 255, 100), # Yellow (Banana-like)
    ]

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
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = deque(maxlen=10)

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.fruits_sliced = 0
        self.game_over = False
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.fruits.clear()
        self.bombs.clear()
        self.particles.clear()
        self.slice_trails.clear()
        
        self.base_fall_speed = 2.0
        self.spawn_timer = 0
        self.spawn_rate = 60 # Spawn an object every 60 steps initially

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, _ = action
        
        self.steps += 1
        reward = 0.01  # Small reward for surviving

        # 1. Handle Player Input
        self._handle_input(movement, space_pressed)

        # 2. Update Game State
        self._update_objects()
        self._spawn_objects()

        # 3. Handle Slicing
        if space_pressed:
            slice_reward, slice_hit_bomb = self._perform_slice()
            reward += slice_reward
            if slice_hit_bomb:
                self.lives -= 1
                # sfx: bomb_explode.wav

        # 4. Check Termination Conditions
        terminated = self._check_termination()
        if self.fruits_sliced >= self.FRUITS_TO_WIN and not self.game_over:
            reward += 50  # Bonus for winning
            self.game_over = True
        
        if self.lives <= 0 and not self.game_over:
            self.game_over = True
            # No extra penalty here, the -10 from bomb hit is sufficient

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED # Up
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED # Down
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED # Left
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED # Right

        # Clamp cursor to screen bounds
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        # Add to slice trail if space is pressed
        if space_pressed:
            self.slice_trails.append((self.cursor_pos.copy(), 15)) # pos, lifetime

    def _perform_slice(self):
        reward = 0
        hit_bomb = False
        sliced_something = False

        # Prioritize slicing fruits
        for fruit in self.fruits[:]:
            dist = np.linalg.norm(self.cursor_pos - fruit['pos'])
            if dist < fruit['radius']:
                reward += 1
                self.score += 100
                self.fruits_sliced += 1
                self._create_fruit_splash(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)
                sliced_something = True
                # sfx: fruit_slice.wav

        # If no fruit was sliced, check for bombs
        if not sliced_something:
            for bomb in self.bombs[:]:
                dist = np.linalg.norm(self.cursor_pos - bomb['pos'])
                if dist < bomb['radius']:
                    reward -= 10
                    hit_bomb = True
                    self._create_bomb_explosion(bomb['pos'])
                    self.bombs.remove(bomb)
                    break # Only hit one bomb per slice
        
        return reward, hit_bomb

    def _update_objects(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_fall_speed += 0.1
        
        # Update fruits
        for fruit in self.fruits:
            fruit['pos'][1] += fruit['vel']
            fruit['angle'] += fruit['rot_speed']
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.SCREEN_HEIGHT + f['radius']]
        
        # Update bombs
        for bomb in self.bombs:
            bomb['pos'][1] += bomb['vel']
            # Fuse particle effect
            if self.np_random.random() < 0.5:
                fuse_pos = bomb['pos'] + np.array([0, -bomb['radius']])
                self.particles.append(self._create_particle(
                    pos=fuse_pos, color=self.COLOR_BOMB_FUSE, lifespan=5, size=3, velocity_spread=0.5
                ))
        self.bombs = [b for b in self.bombs if b['pos'][1] < self.SCREEN_HEIGHT + b['radius']]

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
        # Update slice trails
        new_trails = deque(maxlen=10)
        for pos, lifetime in self.slice_trails:
            if lifetime > 0:
                new_trails.append((pos, lifetime - 1))
        self.slice_trails = new_trails


    def _spawn_objects(self):
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0
            
            # Decrease spawn timer for faster spawns as game progresses
            if self.fruits_sliced > 0 and self.fruits_sliced % 10 == 0:
                 self.spawn_rate = max(15, self.spawn_rate - 4)

            x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            speed = self.base_fall_speed + self.np_random.uniform(-0.5, 0.5)
            
            if self.np_random.random() < 0.25 and len(self.bombs) < 3: # 25% chance for a bomb
                self.bombs.append({
                    'pos': np.array([x, -30], dtype=np.float32),
                    'vel': speed,
                    'radius': 20,
                })
            else:
                self.fruits.append({
                    'pos': np.array([x, -30], dtype=np.float32),
                    'vel': speed,
                    'radius': self.np_random.uniform(20, 30),
                    'color': random.choice(self.FRUIT_PALETTE),
                    'angle': 0,
                    'rot_speed': self.np_random.uniform(-0.1, 0.1)
                })

    def _check_termination(self):
        return self.lives <= 0 or self.fruits_sliced >= self.FRUITS_TO_WIN or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fruits_sliced": self.fruits_sliced,
        }

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render slice trails
        if len(self.slice_trails) > 1:
            for i in range(len(self.slice_trails) - 1):
                start_pos, start_life = self.slice_trails[i]
                end_pos, end_life = self.slice_trails[i+1]
                alpha = int(255 * (start_life / 15))
                color = (*self.COLOR_TRAIL, alpha)
                pygame.draw.aaline(self.screen, color, start_pos, end_pos, 1)

        # Render bombs
        for bomb in self.bombs:
            pos_int = bomb['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], bomb['radius'], self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], bomb['radius'], (60, 60, 60))

        # Render fruits
        for fruit in self.fruits:
            pos_int = fruit['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(fruit['radius']), fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(fruit['radius']), tuple(c*0.8 for c in fruit['color']))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            if color[3] > 0:
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
                self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

        # Render cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 8, cy), (cx + 8, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 8), (cx, cy + 8), 2)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Score
        draw_text(f"Score: {self.score}", self.font_small, self.COLOR_TEXT, (10, 10))
        # Fruits Sliced
        draw_text(f"Fruits: {self.fruits_sliced}/{self.FRUITS_TO_WIN}", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10))
        # Lives
        draw_text(f"Lives: {self.lives}", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 35))

        # Game Over / Win Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.fruits_sliced >= self.FRUITS_TO_WIN else "GAME OVER"
            color = (100, 255, 100) if self.fruits_sliced >= self.FRUITS_TO_WIN else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
            self.screen.blit(score_surf, score_rect)

    def _create_particle(self, pos, color, lifespan, size, velocity_spread):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed * velocity_spread
        return {
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'size': size,
        }

    def _create_fruit_splash(self, pos, color):
        for _ in range(15):
            p = self._create_particle(pos, color, lifespan=20, size=self.np_random.integers(2, 5), velocity_spread=2)
            p['vel'][1] += 0.5 # Add a little gravity
            self.particles.append(p)

    def _create_bomb_explosion(self, pos):
        for i in range(50):
            angle = (i / 50) * 2 * math.pi
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(20, 40)
            color = random.choice([(255, 100, 0), (255, 50, 0), (200, 200, 0)])
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color, 
                'lifespan': lifespan, 'max_lifespan': lifespan, 'size': 3
            })

    def close(self):
        pygame.quit()

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

# Example usage to test the environment visually
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # --- Game Loop ---
    running = True
    while running:
        # --- Action mapping for human ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS

    env.close()