
# Generated: 2025-08-28T01:49:54.017150
# Source Brief: brief_04246.md
# Brief Index: 4246

        
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
        "Controls: ←→ to move the slicer. Press space to slice."
    )

    game_description = (
        "Slice falling fruit to score points while avoiding bombs. "
        "Slicing multiple fruits at once grants a bonus!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.MAX_BOMBS = 3
        
        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_SLICER = (26, 188, 156)
        self.COLOR_BOMB = (52, 73, 94)
        self.COLOR_FUSE = (231, 76, 60)
        self.FRUIT_COLORS = {
            "apple": (231, 76, 60),
            "orange": (230, 126, 34),
            "lemon": (241, 196, 15),
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables
        self.slicer_x = 0
        self.slicer_speed = 12
        self.fall_speed = 0.0
        self.objects = []
        self.particles = []
        self.slice_effect = None
        self.screen_shake = 0
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        
        self.reset()
        
        # This is a self-check to ensure the implementation follows the spec
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        
        self.slicer_x = self.WIDTH // 2
        self.fall_speed = 2.0
        
        self.objects = []
        self.particles = []
        self.slice_effect = None
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- UPDATE GAME LOGIC ---

        # 1. Handle Slicer Movement
        if movement == 3:  # Left
            self.slicer_x -= self.slicer_speed
        elif movement == 4:  # Right
            self.slicer_x += self.slicer_speed
        self.slicer_x = np.clip(self.slicer_x, 0, self.WIDTH)

        # 2. Update Object Positions & Handle Off-screen
        for obj in list(self.objects):
            obj['pos'][1] += self.fall_speed
            if obj['pos'][1] > self.HEIGHT + 50:
                if obj['type'] == 'fruit':
                    reward -= 1  # Penalty for missed fruit
                self.objects.remove(obj)

        # 3. Handle Slicing
        if space_held:
            # sound: slice.wav
            self.slice_effect = {'x': self.slicer_x, 'timer': 4}
            sliced_fruits = 0
            
            sliced_objects = []
            for obj in self.objects:
                if abs(obj['pos'][0] - self.slicer_x) < obj['radius'] + 8:
                    sliced_objects.append(obj)
            
            for obj in sliced_objects:
                if obj['type'] == 'fruit':
                    # sound: fruit_squish.wav
                    sliced_fruits += 1
                    reward += 1
                    self.score += obj['value']
                    self._create_particles(obj['pos'], self.FRUIT_COLORS[obj['subtype']], 30)
                elif obj['type'] == 'bomb':
                    # sound: explosion.wav
                    reward -= 5
                    self.bombs_hit += 1
                    self.screen_shake = 10
                    self._create_particles(obj['pos'], self.COLOR_FUSE, 50, 'explosion')
                
                if obj in self.objects:
                    self.objects.remove(obj)

            if sliced_fruits > 1:
                reward += 5  # Combo bonus

        # 4. Spawn New Objects
        spawn_chance = 4 + self.steps // 150
        if self.np_random.integers(0, 100) < spawn_chance:
            self._spawn_object()

        # 5. Update Particles & Effects
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
        
        if self.slice_effect and self.slice_effect['timer'] > 0:
            self.slice_effect['timer'] -= 1
        else:
            self.slice_effect = None

        if self.screen_shake > 0:
            self.screen_shake -= 1

        # 6. Update Difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.fall_speed = min(8.0, self.fall_speed + 0.2)
        
        self.steps += 1
        
        # --- CHECK TERMINATION ---
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.bombs_hit >= self.MAX_BOMBS:
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

    def _spawn_object(self):
        x = self.np_random.integers(50, self.WIDTH - 50)
        y = -50
        radius = self.np_random.integers(15, 25)
        
        if self.np_random.random() < 0.25: # 25% chance of bomb
            obj_type = 'bomb'
            subtype = None
            value = 0
        else:
            obj_type = 'fruit'
            subtype = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            value = 1

        self.objects.append({
            'pos': [x, y],
            'type': obj_type,
            'subtype': subtype,
            'radius': radius,
            'value': value,
            'rotation': self.np_random.random() * 360,
            'rot_speed': self.np_random.uniform(-2, 2)
        })

    def _create_particles(self, pos, color, count, p_type='splatter'):
        for _ in range(count):
            if p_type == 'splatter':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
                lifetime = self.np_random.integers(20, 40)
            else: # explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifetime = self.np_random.integers(30, 50)
                color = self.np_random.choice([self.COLOR_FUSE, (243, 156, 18), self.COLOR_BOMB])
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'color': color
            })

    def _get_observation(self):
        render_surface = self.screen
        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-5, 5)
            offset_y = self.np_random.integers(-5, 5)
            render_surface = self.screen.copy()
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT))
            temp_surf.blit(self.screen, (offset_x, offset_y))
            render_surface = temp_surf

        render_surface.fill(self.COLOR_BG)
        self._render_game(render_surface)
        self._render_ui(render_surface)
        
        arr = pygame.surfarray.array3d(render_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        # Render Grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            surface.blit(temp_surf, (int(p['pos'][0]) - 2, int(p['pos'][1]) - 2))

        # Render Objects
        for obj in self.objects:
            obj['rotation'] += obj['rot_speed']
            pos_int = (int(obj['pos'][0]), int(obj['pos'][1]))
            if obj['type'] == 'fruit':
                self._draw_fruit(surface, obj)
            elif obj['type'] == 'bomb':
                self._draw_bomb(surface, pos_int, obj['radius'])

        # Render Slicer indicator
        pygame.draw.circle(surface, self.COLOR_SLICER, (self.slicer_x, 20), 8, 2)
        pygame.draw.circle(surface, self.COLOR_SLICER, (self.slicer_x, self.HEIGHT - 20), 8, 2)
        pygame.draw.line(surface, (*self.COLOR_SLICER, 50), (self.slicer_x, 20), (self.slicer_x, self.HEIGHT - 20), 1)
        
        # Render Slice Effect
        if self.slice_effect:
            alpha = int(255 * (self.slice_effect['timer'] / 4))
            x = self.slice_effect['x']
            width = 15 * (self.slice_effect['timer'] / 4)
            slice_surf = pygame.Surface((width * 2, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(slice_surf, (255, 255, 255, alpha), (width, 0), (width, self.HEIGHT), int(width))
            surface.blit(slice_surf, (x - width, 0))

    def _draw_fruit(self, surface, obj):
        pos = (int(obj['pos'][0]), int(obj['pos'][1]))
        radius = obj['radius']
        color = self.FRUIT_COLORS[obj['subtype']]
        
        # Body
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, (30, 30, 30))
        
        # Simple shine
        shine_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
        pygame.gfxdraw.filled_circle(surface, shine_pos[0], shine_pos[1], radius // 4, (255, 255, 255, 80))

    def _draw_bomb(self, surface, pos, radius):
        # Body
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, self.COLOR_BOMB)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, (10, 10, 10))

        # Fuse
        fuse_end = (pos[0] + radius // 2, pos[1] - radius // 2)
        pygame.draw.line(surface, (120, 120, 120), (pos[0], pos[1] - radius), fuse_end, 3)
        
        # Spark
        if self.np_random.random() > 0.1:
            spark_color = self.np_random.choice([self.COLOR_FUSE, (241, 196, 15)])
            pygame.draw.circle(surface, spark_color, fuse_end, 3)

    def _render_ui(self, surface):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (20, 10))

        # Bomb counter
        for i in range(self.MAX_BOMBS):
            pos = (self.WIDTH - 30 - i * 35, 25)
            if i < self.bombs_hit:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], 12, self.COLOR_FUSE)
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], 12, self.COLOR_FUSE)
            else:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], 12, self.COLOR_BOMB)
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], 12, (10,10,10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (46, 204, 113)
            else:
                msg = "GAME OVER"
                color = self.COLOR_FUSE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            surface.blit(overlay, (0, 0))
            surface.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_hit": self.bombs_hit,
            "fall_speed": self.fall_speed,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        # In a human-playable context, we map keys to a single step
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()