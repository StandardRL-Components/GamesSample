
# Generated: 2025-08-27T14:42:32.527332
# Source Brief: brief_00766.md
# Brief Index: 766

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the slicer. Press space to slice fruit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to reach a target score before too many fall off-screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.CURSOR_SPEED = 15
        self.TRAIL_LENGTH = 15
        self.WIN_SCORE = 25
        self.LOSE_MISSES = 5
        self.MAX_STEPS = 2000
        
        # Difficulty scaling
        self.BASE_FRUIT_SPEED = 2.0
        self.FRUIT_SPEED_INCREASE = 0.4
        self.FRUIT_SCORE_INTERVAL = 5 # Speed increases every 5 fruits sliced

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 70)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)
        self.FRUIT_COLORS = {
            "apple": (220, 40, 40),
            "orange": (240, 140, 20),
            "lemon": (255, 230, 50),
            "lime": (150, 220, 30),
        }
        self.COLOR_HIGHLIGHT = (255, 255, 255, 100)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 28, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 60, bold=True)

        # Initialize state variables
        self.cursor_pos = None
        self.fruits = None
        self.particles = None
        self.trail = None
        self.steps = None
        self.score = None
        self.misses = None
        self.game_over = None
        self.spawn_timer = None
        self.current_fruit_speed = None
        self.last_space_press = None
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.fruits = []
        self.particles = []
        self.trail = deque(maxlen=self.TRAIL_LENGTH)
        
        self.spawn_timer = self.np_random.integers(30, 60)
        self.current_fruit_speed = self.BASE_FRUIT_SPEED
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        reward = 0
        
        # Update game logic
        self._handle_input(movement, space_held)
        
        sliced_fruit_this_step, slice_pos, slice_color = self._check_slicing(space_held)
        if sliced_fruit_this_step:
            # Sound effect placeholder: Slicing sound
            reward += 1.0 * len(sliced_fruit_this_step)
            self.score += len(sliced_fruit_this_step)
            for fruit in sliced_fruit_this_step:
                self._create_particles(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)

        misses_this_step = self._update_fruits()
        if misses_this_step == 0:
            reward += 0.1 # Survival reward
        else:
            # Sound effect placeholder: Miss sound
            self.misses += misses_this_step

        self._update_particles()
        self._spawn_fruit()
        self._update_difficulty()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
            elif self.misses >= self.LOSE_MISSES:
                reward -= 100.0 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        # Clamp cursor to screen bounds
        self.cursor_pos.x = max(0, min(self.SCREEN_WIDTH, self.cursor_pos.x))
        self.cursor_pos.y = max(0, min(self.SCREEN_HEIGHT, self.cursor_pos.y))
        
        self.trail.append(self.cursor_pos.copy())

    def _check_slicing(self, space_held):
        sliced_fruits = []
        slice_pos = None
        slice_color = None

        # Detect a new press of the space key (edge-triggered)
        is_slicing = space_held and not self.last_space_press
        self.last_space_press = space_held
        
        if is_slicing:
            for fruit in self.fruits:
                distance = self.cursor_pos.distance_to(fruit['pos'])
                if distance < fruit['radius'] + 20: # Generous slice hitbox
                    sliced_fruits.append(fruit)
        
        return sliced_fruits, slice_pos, slice_color

    def _update_fruits(self):
        misses_this_step = 0
        for fruit in self.fruits[:]:
            fruit['pos'] += fruit['vel']
            fruit['angle'] += fruit['rot_speed']
            if fruit['pos'].y > self.SCREEN_HEIGHT + fruit['radius']:
                self.fruits.remove(fruit)
                misses_this_step += 1
        return misses_this_step

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_fruit(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            x_pos = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            x_vel = self.np_random.uniform(-1, 1)
            y_vel = self.current_fruit_speed + self.np_random.uniform(-0.5, 0.5)
            
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            
            new_fruit = {
                'pos': pygame.Vector2(x_pos, -30),
                'vel': pygame.Vector2(x_vel, y_vel),
                'radius': self.np_random.integers(20, 30),
                'color': self.FRUIT_COLORS[fruit_type],
                'angle': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-3, 3)
            }
            self.fruits.append(new_fruit)
            self.spawn_timer = self.np_random.integers(
                max(10, 45 - self.score), max(20, 75 - self.score * 2)
            )

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_difficulty(self):
        self.current_fruit_speed = self.BASE_FRUIT_SPEED + \
            (self.score // self.FRUIT_SCORE_INTERVAL) * self.FRUIT_SPEED_INCREASE

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.misses >= self.LOSE_MISSES or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_trail()
        self._render_particles()
        self._render_fruits()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_fruits(self):
        for fruit in self.fruits:
            x, y = int(fruit['pos'].x), int(fruit['pos'].y)
            r = fruit['radius']
            
            # Main body with anti-aliasing
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit['color'])
            
            # Simple highlight for 3D effect
            highlight_pos = (x - r // 3, y - r // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], r // 3, self.COLOR_HIGHLIGHT)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_trail(self):
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                start_pos = self.trail[i]
                end_pos = self.trail[i+1]
                alpha = int(255 * (i / self.TRAIL_LENGTH))
                width = int(1 + 15 * (i / self.TRAIL_LENGTH))
                
                # Create a temporary surface for the line segment to handle alpha
                line_rect = pygame.Rect(min(start_pos.x, end_pos.x) - width, min(start_pos.y, end_pos.y) - width, abs(start_pos.x - end_pos.x) + 2*width, abs(start_pos.y - end_pos.y) + 2*width)
                if line_rect.width > 0 and line_rect.height > 0:
                    line_surf = pygame.Surface(line_rect.size, pygame.SRCALPHA)
                    pygame.draw.line(line_surf, (255, 255, 255, alpha), (start_pos.x - line_rect.left, start_pos.y - line_rect.top), (end_pos.x - line_rect.left, end_pos.y - line_rect.top), width)
                    self.screen.blit(line_surf, line_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (12, 12))
        self.screen.blit(text_surf, (10, 10))
        
        # Misses
        miss_text = f"MISSES: {self.misses}/{self.LOSE_MISSES}"
        text_surf = self.font_ui.render(miss_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_ui.render(miss_text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        shadow_rect = shadow_surf.get_rect(topright=(self.SCREEN_WIDTH - 8, 12))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text = "YOU WIN!"
                color = (100, 255, 100)
            else:
                end_text = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_game_over.render(end_text, True, color)
            shadow_surf = self.font_game_over.render(end_text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()