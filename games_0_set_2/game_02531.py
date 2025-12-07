
# Generated: 2025-08-28T05:09:15.339066
# Source Brief: brief_02531.md
# Brief Index: 2531

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to perform slices. Up/Down for vertical slices, Left/Right for horizontal slices."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding bombs. Reach 100 points to win, but slicing 3 bombs ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Constants
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 100
    MAX_BOMBS_SLICED = 3
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_BOMB = (50, 50, 60)
    COLOR_BOMB_FLASH = (255, 50, 50)
    COLOR_SHADOW = (0, 0, 0, 50)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (255, 255, 80),  # Yellow
        (255, 165, 0),   # Orange
        (180, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_big = pygame.font.SysFont("Arial", 60, bold=True)
        
        self.render_mode = render_mode
        
        self.reset()
        
        # This will run once and ensures the implementation is correct.
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bombs_sliced = 0
        self.game_over = False
        
        self.falling_objects = []
        self.particles = []
        self.slice_effects = []
        
        # Initial spawn rates (objects per second)
        self.base_spawn_rate_fruit = 1.0
        self.base_spawn_rate_bomb = 0.1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        
        if not self.game_over:
            self.steps += 1
            
            # 1. Handle player action (slicing)
            slice_reward = self._handle_slicing(action)
            reward += slice_reward
            
            # 2. Update game state
            self._update_objects()
            self._spawn_objects()
            self._update_effects()

            # 3. Check for termination conditions
            if self.score >= self.WIN_SCORE:
                reward += 10
                terminated = True
                self.game_over = True
            elif self.bombs_sliced >= self.MAX_BOMBS_SLICED:
                reward -= 100
                terminated = True
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_slicing(self, action):
        movement = action[0]
        if movement == 0:  # No-op
            return 0

        reward = 0
        slice_line = None
        
        # Define slice lines based on action
        if movement == 1:  # Up -> Vertical slice 1
            slice_line = ((self.WIDTH // 3, 0), (self.WIDTH // 3, self.HEIGHT))
        elif movement == 2:  # Down -> Vertical slice 2
            slice_line = ((2 * self.WIDTH // 3, 0), (2 * self.WIDTH // 3, self.HEIGHT))
        elif movement == 3:  # Left -> Horizontal slice 1
            slice_line = ((0, self.HEIGHT // 3), (self.WIDTH, self.HEIGHT // 3))
        elif movement == 4:  # Right -> Horizontal slice 2
            slice_line = ((0, 2 * self.HEIGHT // 3), (self.WIDTH, 2 * self.HEIGHT // 3))

        if not slice_line:
            return 0

        # Check for collisions with falling objects
        sliced_objects = []
        for obj in self.falling_objects:
            obj_rect = pygame.Rect(obj['pos'][0] - obj['size'], obj['pos'][1] - obj['size'], obj['size'] * 2, obj['size'] * 2)
            if obj_rect.clipline(slice_line):
                sliced_objects.append(obj)
        
        for obj in sliced_objects:
            self.falling_objects.remove(obj)
            if obj['type'] == 'fruit':
                # SFX: Fruit slice
                reward += 1
                self.score += 1
                self._create_fruit_particles(obj['pos'], obj['color'])
                self.slice_effects.append({'start': slice_line[0], 'end': slice_line[1], 'color': obj['color'], 'width': 5, 'lifetime': 10})
            elif obj['type'] == 'bomb':
                # SFX: Explosion
                reward -= 5
                self.bombs_sliced += 1
                self._create_bomb_particles(obj['pos'])
                self.slice_effects.append({'start': slice_line[0], 'end': slice_line[1], 'color': self.COLOR_BOMB_FLASH, 'width': 8, 'lifetime': 15})
        
        return reward

    def _update_objects(self):
        # Move falling objects and remove those off-screen
        for obj in self.falling_objects[:]:
            obj['pos'][1] += obj['vel_y']
            if obj['pos'][1] - obj['size'] > self.HEIGHT:
                self.falling_objects.remove(obj)

    def _spawn_objects(self):
        # Difficulty scaling
        rate_increase = (self.steps // 100) * 0.01
        current_spawn_rate_fruit = self.base_spawn_rate_fruit + rate_increase
        current_spawn_rate_bomb = self.base_spawn_rate_bomb + rate_increase
        
        # Spawn fruit
        if self.np_random.random() < current_spawn_rate_fruit / self.FPS:
            size = self.np_random.integers(15, 25)
            self.falling_objects.append({
                'pos': [self.np_random.integers(size, self.WIDTH - size), -size],
                'type': 'fruit',
                'color': random.choice(self.FRUIT_COLORS),
                'size': size,
                'vel_y': self.np_random.uniform(2, 4)
            })
            
        # Spawn bomb
        if self.np_random.random() < current_spawn_rate_bomb / self.FPS:
            size = 20
            self.falling_objects.append({
                'pos': [self.np_random.integers(size, self.WIDTH - size), -size],
                'type': 'bomb',
                'color': self.COLOR_BOMB,
                'size': size,
                'vel_y': self.np_random.uniform(3, 5)
            })

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
                
        # Update slice trails
        for s in self.slice_effects[:]:
            s['lifetime'] -= 1
            s['width'] = max(0, s['width'] - 0.5)
            if s['lifetime'] <= 0:
                self.slice_effects.remove(s)

    def _create_fruit_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.integers(2, 5),
                'lifetime': self.np_random.integers(15, 30)
            })

    def _create_bomb_particles(self, pos):
        # Initial flash
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': random.choice([(255,255,255), (255,255,0), self.COLOR_BOMB_FLASH]),
                'size': self.np_random.integers(2, 4),
                'lifetime': self.np_random.integers(10, 20)
            })
        # Lingering smoke
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': self.COLOR_BOMB,
                'size': self.np_random.integers(5, 10),
                'lifetime': self.np_random.integers(20, 40)
            })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a simple gradient for a nice background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render shadows first for pseudo-3D effect
        for obj in sorted(self.falling_objects, key=lambda o: o['pos'][1]):
            shadow_pos = (int(obj['pos'][0]), int(obj['pos'][1] + obj['size'] * 0.7))
            shadow_size = (int(obj['size']), int(obj['size'] * 0.5))
            shadow_surf = pygame.Surface((shadow_size[0]*2, shadow_size[1]*2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, shadow_surf.get_rect())
            self.screen.blit(shadow_surf, (shadow_pos[0] - shadow_size[0], shadow_pos[1] - shadow_size[1]))

        # Render falling objects
        for obj in sorted(self.falling_objects, key=lambda o: o['pos'][1]):
            pos_int = (int(obj['pos'][0]), int(obj['pos'][1]))
            if obj['type'] == 'fruit':
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], obj['size'], obj['color'])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], obj['size'], obj['color'])
            elif obj['type'] == 'bomb':
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], obj['size'], self.COLOR_BOMB)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.COLOR_BOMB)
                # Flashing indicator
                flash_size = int(obj['size'] * 0.4 * (0.75 + 0.25 * math.sin(self.steps * 0.3)))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], flash_size, self.COLOR_BOMB_FLASH)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render slice effects
        for s in self.slice_effects:
            if s['width'] > 0:
                pygame.draw.line(self.screen, s['color'], s['start'], s['end'], int(s['width']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Bombs sliced
        bomb_icon_size = 20
        for i in range(self.MAX_BOMBS_SLICED):
            pos_x = self.WIDTH - (i + 1) * (bomb_icon_size + 5) - 10
            pos_y = 10 + bomb_icon_size // 2
            color = self.COLOR_BOMB if i >= self.bombs_sliced else self.COLOR_BOMB_FLASH
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, bomb_icon_size // 2, color)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, bomb_icon_size // 2, color)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_BOMB_FLASH
                
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_sliced": self.bombs_sliced,
        }
        
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
    import os
    # Set the video driver to a dummy one if not playing interactively
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # To render the game window
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    while not done:
        # Map keyboard inputs to actions
        action = [0, 0, 0] # Default no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Info: {info}")
    env.close()