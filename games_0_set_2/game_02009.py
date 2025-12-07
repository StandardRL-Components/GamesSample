
# Generated: 2025-08-28T03:24:57.556445
# Source Brief: brief_02009.md
# Brief Index: 2009

        
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
        "Controls: Use ↑ and ↓ to move the slicer. Press Space to slice."
    )

    game_description = (
        "Slice the falling fruit while dodging the bombs! Slice 30 fruits to win, but slicing 3 bombs ends the game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.render_mode = render_mode
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.SLICER_SPEED = 15
        self.MAX_STEPS = 1000
        self.TOTAL_FRUITS = 30
        self.BOMB_LIMIT = 3
        self.OBJECT_BASE_SIZE = 20
        self.SPAWN_INTERVAL = 25
        self.BOMB_CHANCE = 0.2

        self._setup_colors()
        self._setup_fonts()
        self._create_object_surfaces()
        
        # Initialize state variables to be populated in reset()
        self.slicer_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bombs_sliced = 0
        self.fruits_sliced = 0
        self.fruits_to_spawn = 0
        self.fall_speed = 0.0
        self.objects = []
        self.particles = []
        self.slice_effects = []
        self.steps_since_spawn = 0
        
        self.reset()
        self.validate_implementation()
    
    def _setup_colors(self):
        self.COLOR_BG_TOP = (40, 40, 80)
        self.COLOR_BG_BOTTOM = (10, 10, 30)
        self.COLOR_APPLE = (220, 30, 30)
        self.COLOR_ORANGE = (255, 150, 20)
        self.COLOR_BANANA = (255, 230, 50)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_FUSE = (200, 50, 50)
        self.COLOR_SPARK = (255, 255, 100)
        self.COLOR_STEM = (139, 69, 19)
        self.COLOR_SLICER = (220, 220, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.EXPLOSION_COLORS = [(255, 50, 50), (255, 150, 0), (255, 255, 0)]

    def _setup_fonts(self):
        try:
            self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 24)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)

    def _create_object_surfaces(self):
        self.object_surfaces = {}
        s = self.OBJECT_BASE_SIZE * 2.5 # Surface size
        c = s / 2 # Center
        
        # Apple
        apple_surf = pygame.Surface((s, s), pygame.SRCALPHA)
        pygame.draw.circle(apple_surf, self.COLOR_APPLE, (c, c), self.OBJECT_BASE_SIZE)
        pygame.draw.line(apple_surf, self.COLOR_STEM, (c, c - self.OBJECT_BASE_SIZE), (c, c - self.OBJECT_BASE_SIZE + 8), 4)
        self.object_surfaces['apple'] = apple_surf

        # Orange
        orange_surf = pygame.Surface((s, s), pygame.SRCALPHA)
        pygame.draw.circle(orange_surf, self.COLOR_ORANGE, (c, c), self.OBJECT_BASE_SIZE)
        self.object_surfaces['orange'] = orange_surf

        # Banana
        banana_surf = pygame.Surface((s, s), pygame.SRCALPHA)
        rect = pygame.Rect(c - self.OBJECT_BASE_SIZE, c - self.OBJECT_BASE_SIZE / 2, self.OBJECT_BASE_SIZE * 2, self.OBJECT_BASE_SIZE)
        pygame.draw.arc(banana_surf, self.COLOR_BANANA, rect, math.pi, 2*math.pi, int(self.OBJECT_BASE_SIZE/1.5))
        self.object_surfaces['banana'] = banana_surf

        # Bomb
        bomb_surf = pygame.Surface((s, s), pygame.SRCALPHA)
        pygame.draw.circle(bomb_surf, self.COLOR_BOMB, (c, c), self.OBJECT_BASE_SIZE)
        fuse_end = (c + 8, c - self.OBJECT_BASE_SIZE - 4)
        pygame.draw.line(bomb_surf, self.COLOR_FUSE, (c, c - self.OBJECT_BASE_SIZE + 5), fuse_end, 5)
        pygame.gfxdraw.filled_circle(bomb_surf, int(fuse_end[0]), int(fuse_end[1]), 4, self.COLOR_SPARK)
        self.object_surfaces['bomb'] = bomb_surf

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.slicer_y = self.SCREEN_HEIGHT / 2
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bombs_sliced = 0
        self.fruits_sliced = 0
        self.fruits_to_spawn = self.TOTAL_FRUITS
        self.fall_speed = 2.0
        self.objects = []
        self.particles = []
        self.slice_effects = []
        self.steps_since_spawn = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input
        movement = action[0]
        space_held = action[1] == 1
        old_slicer_y = self.slicer_y
        self._update_slicer(movement)
        
        # 2. Calculate Movement Reward
        reward += self._calculate_movement_reward(old_slicer_y)

        # 3. Handle Slicing
        if space_held:
            slice_reward = self._perform_slice()
            reward += slice_reward

        # 4. Update Game World
        self._update_objects_and_particles()
        self._update_spawner()
        self.steps += 1
        if self.steps > 0 and self.steps % 100 == 0:
            self.fall_speed = min(8.0, self.fall_speed + 0.05)
        
        # 5. Check Termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.fruits_sliced >= self.TOTAL_FRUITS:
                reward += 100
                # SFX: victory.wav
        
        # Update score with rewards
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_slicer(self, movement):
        if movement == 1:  # Up
            self.slicer_y -= self.SLICER_SPEED
        elif movement == 2:  # Down
            self.slicer_y += self.SLICER_SPEED
        self.slicer_y = np.clip(self.slicer_y, 0, self.SCREEN_HEIGHT)

    def _calculate_movement_reward(self, old_slicer_y):
        fruits = [obj for obj in self.objects if obj['type'] != 'bomb']
        if not fruits or self.slicer_y == old_slicer_y:
            return 0

        closest_fruit = min(fruits, key=lambda f: abs(f['y'] - self.slicer_y))
        
        old_dist = abs(closest_fruit['y'] - old_slicer_y)
        new_dist = abs(closest_fruit['y'] - self.slicer_y)

        if new_dist < old_dist:
            return 1.0
        elif new_dist > old_dist:
            return -0.1
        return 0

    def _perform_slice(self):
        # SFX: whoosh.wav
        self.slice_effects.append({'y': self.slicer_y, 'timer': 8, 'alpha': 255})
        for _ in range(20): # Spark particles
            self.particles.append({
                'x': self.np_random.uniform(0, self.SCREEN_WIDTH), 'y': self.slicer_y,
                'vx': self.np_random.uniform(-1, 1), 'vy': self.np_random.uniform(-1, 1),
                'lifespan': self.np_random.integers(10, 20), 'color': self.COLOR_SLICER, 'size': self.np_random.integers(1, 4)
            })

        reward = 0
        hit_objects = [obj for obj in self.objects if abs(obj['y'] - self.slicer_y) < obj['size']]
        
        for obj in hit_objects:
            if obj['type'] == 'bomb':
                # SFX: explosion.wav
                self.bombs_sliced += 1
                reward -= 50
                self._create_explosion(obj['x'], obj['y'])
            else: # Fruit
                # SFX: slice.wav
                self.fruits_sliced += 1
                reward += 10
                self._create_sliced_fruit_particles(obj)
            self.objects.remove(obj)
        return reward

    def _create_explosion(self, x, y):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            self.particles.append({
                'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(20, 40),
                'color': random.choice(self.EXPLOSION_COLORS),
                'size': self.np_random.integers(2, 5)
            })

    def _create_sliced_fruit_particles(self, obj):
        for i in range(2):
            angle = self.np_random.uniform(-math.pi/2, math.pi/2) + (i * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': obj['x'], 'y': obj['y'], 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed - 2,
                'lifespan': 60, 'type': obj['type'], 'rotation': obj['rotation'],
                'rotation_speed': obj['rotation_speed'] * (1 if i == 0 else -1) * 2, 'size': obj['size']
            })

    def _update_objects_and_particles(self):
        # Update falling objects
        self.objects = [obj for obj in self.objects if obj['y'] < self.SCREEN_HEIGHT + obj['size'] * 2]
        for obj in self.objects:
            obj['y'] += self.fall_speed
            obj['rotation'] = (obj['rotation'] + obj['rotation_speed']) % 360

        # Update particles
        next_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            if 'type' in p: # Sliced fruit
                p['vy'] += 0.15 # Gravity
                p['rotation'] = (p['rotation'] + p['rotation_speed']) % 360
            if p['lifespan'] > 0:
                next_particles.append(p)
        self.particles = next_particles
        
        # Update slice visual effects
        self.slice_effects = [s for s in self.slice_effects if s['timer'] > 0]
        for s in self.slice_effects:
            s['timer'] -= 1
            s['alpha'] = max(0, s['alpha'] - 32)

    def _update_spawner(self):
        self.steps_since_spawn += 1
        if self.steps_since_spawn >= self.SPAWN_INTERVAL and self.fruits_to_spawn > 0:
            self.steps_since_spawn = 0
            is_bomb = self.np_random.random() < self.BOMB_CHANCE
            
            if not is_bomb:
                obj_type = self.np_random.choice(['apple', 'orange', 'banana'])
                self.fruits_to_spawn -= 1
            else:
                obj_type = 'bomb'

            self.objects.append({
                'x': self.np_random.uniform(self.OBJECT_BASE_SIZE, self.SCREEN_WIDTH - self.OBJECT_BASE_SIZE),
                'y': -self.OBJECT_BASE_SIZE,
                'type': obj_type,
                'size': self.OBJECT_BASE_SIZE,
                'rotation': self.np_random.uniform(0, 360),
                'rotation_speed': self.np_random.uniform(-2, 2)
            })

    def _check_termination(self):
        return (
            self.bombs_sliced >= self.BOMB_LIMIT or
            self.fruits_sliced >= self.TOTAL_FRUITS or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self._render_background()
        self._render_objects_and_particles()
        self._render_slicer_and_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_sliced": self.fruits_sliced,
            "bombs_sliced": self.bombs_sliced,
        }

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.SCREEN_HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.SCREEN_HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.SCREEN_HEIGHT
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_objects_and_particles(self):
        # Render full objects
        for obj in self.objects:
            self._draw_rotated_surface(obj)
        
        # Render particles
        for p in self.particles:
            if 'type' in p: # Sliced fruit
                self._draw_rotated_surface(p)
            else: # Explosion/spark
                size = p['size'] * (p['lifespan'] / 20) # Fade out size
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), max(0, int(size)))

    def _draw_rotated_surface(self, obj):
        surf = self.object_surfaces[obj['type']]
        rotated_surf = pygame.transform.rotate(surf, obj['rotation'])
        w, h = rotated_surf.get_size()
        self.screen.blit(rotated_surf, (int(obj['x'] - w / 2), int(obj['y'] - h / 2)))

    def _render_slicer_and_effects(self):
        # Draw slicer guide
        guide_surf = pygame.Surface((self.SCREEN_WIDTH, 5), pygame.SRCALPHA)
        guide_surf.fill((*self.COLOR_SLICER, 50))
        self.screen.blit(guide_surf, (0, int(self.slicer_y - 2)))
        
        # Draw slice flash effects
        for effect in self.slice_effects:
            surf = pygame.Surface((self.SCREEN_WIDTH, 3), pygame.SRCALPHA)
            surf.fill((*self.COLOR_SLICER, effect['alpha']))
            self.screen.blit(surf, (0, int(effect['y'] - 1)))

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        draw_text(f"Score: {int(self.score)}", self.font_medium, self.COLOR_TEXT, (10, 10))
        bombs_text = f"Bombs: {self.bombs_sliced}/{self.BOMB_LIMIT}"
        bombs_surf = self.font_medium.render(bombs_text, True, self.COLOR_TEXT)
        draw_text(bombs_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - bombs_surf.get_width() - 10, 10))
        fruits_text = f"Fruits: {self.fruits_sliced}/{self.TOTAL_FRUITS}"
        draw_text(fruits_text, self.font_medium, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            if self.fruits_sliced >= self.TOTAL_FRUITS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            draw_text(msg, self.font_large, self.COLOR_TEXT, text_rect.topleft)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the game and play it
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Human controls
        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # 30 FPS
        
    env.close()