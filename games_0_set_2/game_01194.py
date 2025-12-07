
# Generated: 2025-08-27T16:19:54.554375
# Source Brief: brief_01194.md
# Brief Index: 1194

        
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
        "Controls: Use arrow keys to move the cursor. Hold space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding the bombs. The game ends when you hit 3 bombs or reach 100 points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CURSOR_SPEED = 15
        self.GRAVITY = 0.05
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time to reach score goal
        self.WIN_SCORE = 100
        self.MAX_BOMBS = 3
        
        # Colors
        self.COLOR_BG_TOP = (15, 20, 30)
        self.COLOR_BG_BOTTOM = (40, 50, 60)
        self.COLOR_FRUIT_APPLE = (50, 205, 50)
        self.COLOR_FRUIT_ORANGE = (255, 165, 0)
        self.COLOR_BOMB = (50, 50, 50)
        self.COLOR_BOMB_FUSE = (255, 69, 0)
        self.COLOR_TRAIL = (135, 206, 250)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BOMB = (120, 120, 120)
        self.COLOR_UI_BOMB_ACTIVE = (255, 69, 0)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.bombs_hit = None
        self.game_over = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.slice_trail = None
        self.base_spawn_prob = None
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trail = []
        
        self.base_spawn_prob = 0.03

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            
            # --- Game Logic ---
            self._handle_input(movement)
            self._update_objects()
            self._update_particles()
            
            reward += self._handle_slicing(space_held)
            
            self._spawn_objects()

            # --- Check Termination ---
            if self.score >= self.WIN_SCORE:
                reward += 100
                terminated = True
            if self.bombs_hit >= self.MAX_BOMBS:
                reward -= 100
                terminated = True
            if self.steps >= self.MAX_STEPS:
                terminated = True
            
            if terminated:
                self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: # Down
            self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: # Left
            self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: # Right
            self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)
        
        # Update slice trail
        self.slice_trail.append([self.cursor_pos.copy(), 15]) # pos, lifetime
        self.slice_trail = [t for t in self.slice_trail if t[1] > 0]
        for t in self.slice_trail:
            t[1] -= 1

    def _update_objects(self):
        for fruit in self.fruits[:]:
            fruit['vel'].y += self.GRAVITY
            fruit['pos'] += fruit['vel']
            if fruit['pos'].y > self.HEIGHT + fruit['radius']:
                self.fruits.remove(fruit)

        for bomb in self.bombs[:]:
            bomb['vel'].y += self.GRAVITY
            bomb['pos'] += bomb['vel']
            if bomb['pos'].y > self.HEIGHT + bomb['radius']:
                self.bombs.remove(bomb)

    def _handle_slicing(self, space_held):
        step_reward = 0
        sliced_this_frame = 0

        if space_held:
            # SFX: Whoosh sound
            for fruit in self.fruits[:]:
                if self.cursor_pos.distance_to(fruit['pos']) < fruit['radius']:
                    step_reward += 0.1
                    self.score += 1
                    sliced_this_frame += 1
                    self._create_slice_particles(fruit['pos'], fruit['color'])
                    self.fruits.remove(fruit)
                    # SFX: Fruit slice sound

            for bomb in self.bombs[:]:
                if self.cursor_pos.distance_to(bomb['pos']) < bomb['radius']:
                    step_reward -= 1.0
                    self.bombs_hit += 1
                    self._create_explosion_particles(bomb['pos'])
                    self.bombs.remove(bomb)
                    # SFX: Explosion sound
            
            if sliced_this_frame > 1:
                step_reward += 1.0 # Multi-slice bonus
        
        return step_reward

    def _spawn_objects(self):
        spawn_prob = self.base_spawn_prob + (self.score // 25) * 0.01
        if self.np_random.random() < spawn_prob:
            x = self.np_random.uniform(50, self.WIDTH - 50)
            speed_y = self.np_random.uniform(1, 3)
            speed_x = self.np_random.uniform(-1, 1)
            radius = self.np_random.uniform(15, 25)
            
            if self.np_random.random() > 0.2: # 80% chance of fruit
                fruit_type = self.np_random.choice(['apple', 'orange'])
                color = self.COLOR_FRUIT_APPLE if fruit_type == 'apple' else self.COLOR_FRUIT_ORANGE
                self.fruits.append({
                    'pos': pygame.Vector2(x, -radius),
                    'vel': pygame.Vector2(speed_x, speed_y),
                    'radius': radius,
                    'color': color
                })
            else: # 20% chance of bomb
                self.bombs.append({
                    'pos': pygame.Vector2(x, -radius),
                    'vel': pygame.Vector2(speed_x, speed_y),
                    'radius': 20,
                })

    def _create_slice_particles(self, pos, color):
        for _ in range(20):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifetime': lifetime, 'color': color, 'type': 'slice'})

    def _create_explosion_particles(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(20, 40)
            color = random.choice([(255, 69, 0), (255, 165, 0), (255, 255, 0)])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifetime': lifetime, 'color': color, 'type': 'explosion'})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            if p['type'] == 'explosion':
                p['vel'] *= 0.95 # friction
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self._draw_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40))
            if p['type'] == 'slice':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 3, (*p['color'], alpha))
            else: # explosion
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 4, (*p['color'], alpha))

        # Draw fruits and bombs
        for obj_list in [self.fruits, self.bombs]:
            for obj in obj_list:
                pos_x, pos_y = int(obj['pos'].x), int(obj['pos'].y)
                radius = int(obj['radius'])
                if 'color' in obj: # Fruit
                    pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, obj['color'])
                    pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, obj['color'])
                else: # Bomb
                    pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, self.COLOR_BOMB)
                    pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, self.COLOR_BOMB)
                    pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 5, self.COLOR_BOMB_FUSE)

        # Draw slice trail
        for i, (pos, life) in enumerate(self.slice_trail):
            if i > 0:
                prev_pos, _ = self.slice_trail[i-1]
                alpha = int(255 * (life / 15))
                color = (*self.COLOR_TRAIL, alpha)
                width = int(life / 1.5)
                pygame.draw.line(self.screen, color, (int(prev_pos.x), int(prev_pos.y)), (int(pos.x), int(pos.y)), width)
        
        # Draw cursor
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_TRAIL)
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_TRAIL)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Bomb counter
        for i in range(self.MAX_BOMBS):
            color = self.COLOR_UI_BOMB_ACTIVE if i < self.MAX_BOMBS - self.bombs_hit else self.COLOR_UI_BOMB
            pos_x = self.WIDTH - 30 - i * 25
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 30, 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 30, 8, color)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_hit": self.bombs_hit,
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

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
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()