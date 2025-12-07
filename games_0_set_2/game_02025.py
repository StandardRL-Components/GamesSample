
# Generated: 2025-08-28T03:27:09.172669
# Source Brief: brief_02025.md
# Brief Index: 2025

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move your blade. Slice the falling fruit to score points."
    )

    # Short, user-facing description of the game
    game_description = (
        "Slice falling fruit with timed swipes to reach a target score before too many fruits are missed."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_SCORE = 100
    MAX_MISSES = 5
    BLADE_SPEED = 20
    BLADE_TRAIL_LENGTH = 15

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_TRAIL = (220, 220, 255)
    COLOR_UI_TEXT = (230, 230, 230)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red (Apple/Strawberry)
        (255, 200, 60),  # Yellow (Banana/Lemon)
        (80, 220, 80),   # Green (Pear/Lime)
        (255, 140, 50),  # Orange (Orange)
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.sliced_count_for_difficulty = 0
        self.game_over = False
        
        self.blade_pos = None
        self.prev_blade_pos = None
        self.blade_trail = None
        
        self.fruits = None
        self.particles = None
        
        self.fruit_speed = 0
        self.spawn_rate = 0
        
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call to verify during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.sliced_count_for_difficulty = 0
        self.game_over = False
        
        self.blade_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.prev_blade_pos = self.blade_pos.copy()
        self.blade_trail = deque(maxlen=self.BLADE_TRAIL_LENGTH)
        
        self.fruits = []
        self.particles = []
        
        # Initial difficulty
        self.fruit_speed = 2.0
        self.spawn_rate = 0.02 # Probability per frame
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for existing, encouraging quick action

        # 1. Handle player action
        self._handle_input(action)

        # 2. Update game state
        sliced_this_frame = self._update_fruits()
        self._update_particles()
        self._spawn_new_fruits()

        # 3. Calculate rewards for this step
        if sliced_this_frame > 0:
            reward += sliced_this_frame
            # sfx: slice
            self.score += sliced_this_frame
            self.sliced_count_for_difficulty += sliced_this_frame
            self._update_difficulty()

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            elif self.missed_fruits >= self.MAX_MISSES:
                reward -= 100  # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        self.prev_blade_pos = self.blade_pos.copy()

        if movement == 1:  # Up
            self.blade_pos.y -= self.BLADE_SPEED
        elif movement == 2:  # Down
            self.blade_pos.y += self.BLADE_SPEED
        elif movement == 3:  # Left
            self.blade_pos.x -= self.BLADE_SPEED
        elif movement == 4:  # Right
            self.blade_pos.x += self.BLADE_SPEED
        
        # Clamp blade position to screen bounds
        self.blade_pos.x = max(0, min(self.SCREEN_WIDTH, self.blade_pos.x))
        self.blade_pos.y = max(0, min(self.SCREEN_HEIGHT, self.blade_pos.y))

        # Update blade trail if moving
        if self.blade_pos != self.prev_blade_pos:
            self.blade_trail.append(self.blade_pos.copy())
        elif len(self.blade_trail) > 0:
             # Fade trail when stationary
            self.blade_trail.popleft()

    def _update_fruits(self):
        sliced_this_frame = 0
        fruits_to_remove = []
        for i, fruit in enumerate(self.fruits):
            # Move fruit
            fruit['pos'].y += self.fruit_speed

            # Check for slice
            if self._line_circle_collision(self.prev_blade_pos, self.blade_pos, fruit['pos'], fruit['radius']):
                sliced_this_frame += 1
                self._create_splash(fruit['pos'], fruit['color'])
                fruits_to_remove.append(i)
                continue

            # Check for miss
            if fruit['pos'].y > self.SCREEN_HEIGHT + fruit['radius']:
                self.missed_fruits += 1
                # sfx: miss
                fruits_to_remove.append(i)

        # Remove fruits in reverse order to avoid index errors
        for i in sorted(fruits_to_remove, reverse=True):
            del self.fruits[i]
            
        return sliced_this_frame

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'].y += 0.1  # Gravity on particles
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _spawn_new_fruits(self):
        if self.np_random.random() < self.spawn_rate:
            x_pos = self.np_random.integers(50, self.SCREEN_WIDTH - 50)
            radius = self.np_random.integers(15, 25)
            color = self.FRUIT_COLORS[self.np_random.integers(0, len(self.FRUIT_COLORS))]
            
            self.fruits.append({
                'pos': pygame.Vector2(x_pos, -radius),
                'radius': radius,
                'color': color,
            })

    def _update_difficulty(self):
        # Increase fruit speed every 10 fruits sliced
        if self.sliced_count_for_difficulty % 10 == 0 and self.sliced_count_for_difficulty > 0:
            self.fruit_speed += 0.05
        # Increase spawn rate every 20 fruits sliced
        if self.sliced_count_for_difficulty % 20 == 0 and self.sliced_count_for_difficulty > 0:
            self.spawn_rate = min(0.1, self.spawn_rate + 0.01) # Cap spawn rate

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.missed_fruits >= self.MAX_MISSES or
            self.steps >= self.MAX_STEPS
        )
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], color)
            
        # Render fruits
        for fruit in self.fruits:
            x, y = int(fruit['pos'].x), int(fruit['pos'].y)
            r = fruit['radius']
            color = fruit['color']
            # Main fruit body
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
            # Simple highlight for 3D effect
            highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
            pygame.gfxdraw.pixel(self.screen, x-r//3, y-r//3, highlight_color)
            pygame.gfxdraw.pixel(self.screen, x-r//3+1, y-r//3, highlight_color)
            pygame.gfxdraw.pixel(self.screen, x-r//3, y-r//3+1, highlight_color)

        # Render blade trail
        if len(self.blade_trail) > 1:
            for i in range(len(self.blade_trail) - 1):
                start_pos = self.blade_trail[i]
                end_pos = self.blade_trail[i+1]
                
                alpha = (i / self.BLADE_TRAIL_LENGTH) * 255
                width = int(2 + (i / self.BLADE_TRAIL_LENGTH) * 10)
                
                color = (*self.COLOR_TRAIL, int(alpha))
                
                # Custom line drawing for alpha support
                self._draw_thick_line_alpha(self.screen, start_pos, end_pos, width, color)

    def _render_ui(self):
        # Score display
        score_surf = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Misses display
        miss_text = f"Missed: {self.missed_fruits}/{self.MAX_MISSES}"
        miss_surf = self.font_small.render(miss_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_surf, (self.SCREEN_WIDTH - miss_surf.get_width() - 20, 15))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            end_surf = self.font_large.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
        }
        
    # --- Helper Functions ---
    
    def _create_splash(self, pos, color):
        # sfx: splash
        num_particles = self.np_random.integers(15, 25)
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            lifespan = self.np_random.integers(20, 40)
            radius = self.np_random.integers(2, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'radius': radius
            })

    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Check if either endpoint is inside the circle
        if p1.distance_to(circle_center) < circle_radius or p2.distance_to(circle_center) < circle_radius:
            return True

        # Check for intersection with the line segment
        d = p2 - p1
        if d.length() == 0:  # The blade didn't move
            return False
            
        f = p1 - circle_center
        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - circle_radius**2
        discriminant = b*b - 4*a*c

        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True
        return False
        
    def _draw_thick_line_alpha(self, surface, p1, p2, width, color):
        """Draws a thick line with alpha on a surface."""
        line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.line(line_surface, color, p1, p2, max(1, int(width)))
        surface.blit(line_surface, (0, 0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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

if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Run at 30 FPS

    env.close()